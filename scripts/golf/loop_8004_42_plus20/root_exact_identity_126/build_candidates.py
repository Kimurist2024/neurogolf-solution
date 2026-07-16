#!/usr/bin/env python3
"""Remove one or two exact internal Identity nodes from six 8009.46 members."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "submission.zip"
TASKS = (269, 289, 262, 214, 102, 353)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve(aliases: dict[str, str], name: str) -> str:
    while name in aliases:
        name = aliases[name]
    return name


def main() -> int:
    (HERE / "current").mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            source = HERE / "current" / f"task{task:03d}.onnx"
            source.write_bytes(data)
            model = onnx.load_from_string(data)
            aliases = {
                node.output[0]: node.input[0]
                for node in model.graph.node
                if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
            }
            kept = []
            for node in model.graph.node:
                if node.op_type == "Identity" and node.output[0] in aliases:
                    continue
                item = copy.deepcopy(node)
                for index, name in enumerate(item.input):
                    item.input[index] = resolve(aliases, name)
                kept.append(item)
            del model.graph.node[:]
            model.graph.node.extend(kept)
            removed = set(aliases)
            vi = [item for item in model.graph.value_info if item.name not in removed]
            del model.graph.value_info[:]
            model.graph.value_info.extend(vi)
            row = {
                "task": task,
                "authority_sha256": sha(data),
                "removed": aliases,
                "source_nodes": len(kept) + len(aliases),
                "candidate_nodes": len(kept),
            }
            try:
                onnx.checker.check_model(model, full_check=True)
                onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
                out = HERE / "candidates" / f"task{task:03d}_identity_removed.onnx"
                onnx.save(model, out)
                row["checker"] = "pass"
                row["candidate_sha256"] = sha(out.read_bytes())
                row["path"] = str(out.relative_to(ROOT))
            except Exception as exc:
                row["checker"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    (HERE / "build.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
