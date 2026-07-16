#!/usr/bin/env python3
"""Cost-triage exact collapses of consecutive unary/shape operators."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
IDEMPOTENT = {"Abs", "Ceil", "Floor", "Relu", "Round", "Sign"}
INVOLUTIONS = {"Neg", "Not", "BitwiseNot"}


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"unarychain167_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def attrs(node: onnx.NodeProto) -> bytes:
    holder = onnx.NodeProto()
    holder.attribute.extend(node.attribute)
    return holder.SerializeToString()


def perm(node: onnx.NodeProto, rank: int | None = None) -> list[int] | None:
    for attribute in node.attribute:
        if attribute.name == "perm":
            return [int(value) for value in attribute.ints]
    if rank is not None:
        return list(reversed(range(rank)))
    return None


def build_variant(
    model: onnx.ModelProto, inner_index: int, outer_index: int, kind: str
) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    inner = candidate.graph.node[inner_index]
    outer = candidate.graph.node[outer_index]
    source = inner.input[0]
    if kind in {"idempotent", "same_cast", "reshape_chain"}:
        outer.input[0] = source
    elif kind == "involution":
        replacement = helper.make_node(
            "Identity", [source], list(outer.output), name=outer.name
        )
        outer.CopyFrom(replacement)
    elif kind == "transpose_chain":
        first = perm(inner)
        second = perm(outer)
        if first is None or second is None or len(first) != len(second):
            raise ValueError("transpose rank/perm unavailable")
        composed = [first[index] for index in second]
        replacement = helper.make_node(
            "Transpose", [source], list(outer.output), name=outer.name, perm=composed
        )
        outer.CopyFrom(replacement)
    else:
        raise ValueError(kind)
    del candidate.graph.node[inner_index]
    return candidate


def opportunities(model: onnx.ModelProto) -> list[dict[str, Any]]:
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_outputs = {item.name for item in model.graph.output}
    rows: list[dict[str, Any]] = []
    for outer_index, outer in enumerate(model.graph.node):
        if not outer.input:
            continue
        inner_index = producer.get(outer.input[0])
        if inner_index is None or inner_index >= outer_index:
            continue
        inner = model.graph.node[inner_index]
        if len(inner.input) != 1 or len(inner.output) != 1:
            continue
        if uses[inner.output[0]] != 1 or inner.output[0] in graph_outputs:
            continue
        kind = None
        if inner.op_type == outer.op_type and inner.op_type in IDEMPOTENT:
            kind = "idempotent"
        elif inner.op_type == outer.op_type and inner.op_type in INVOLUTIONS:
            kind = "involution"
        elif inner.op_type == outer.op_type == "Cast" and attrs(inner) == attrs(outer):
            kind = "same_cast"
        elif inner.op_type == outer.op_type == "Reshape":
            kind = "reshape_chain"
        elif inner.op_type == outer.op_type == "Transpose":
            kind = "transpose_chain"
        if kind:
            rows.append({
                "inner_index": inner_index,
                "outer_index": outer_index,
                "inner_op": inner.op_type,
                "outer_op": outer.op_type,
                "kind": kind,
            })
    return rows


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            sites = opportunities(model)
            if not sites:
                continue
            baseline = profile(model, task)
            for site in sites:
                row = {"task": task, **site, "baseline": baseline}
                try:
                    candidate = build_variant(
                        model, site["inner_index"], site["outer_index"], site["kind"]
                    )
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    current = profile(candidate, task)
                    row["candidate"] = current
                    row["strict_lower"] = current["cost"] < baseline["cost"]
                    if row["strict_lower"]:
                        path = CANDIDATES / (
                            f"task{task:03d}_{site['inner_index']:04d}_{site['outer_index']:04d}_{site['kind']}.onnx"
                        )
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "profiles": len(rows),
        "strict_lower_count": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": len([row for row in rows if "error" in row]),
    }, indent=2))


if __name__ == "__main__":
    main()
