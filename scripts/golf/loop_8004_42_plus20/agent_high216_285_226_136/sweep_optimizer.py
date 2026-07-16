#!/usr/bin/env python3
"""Conservative ONNX optimizer identity-pass sweep on current authority members."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import onnx
import onnxoptimizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (216, 285, 226)
PASSES = (
    "eliminate_identity", "eliminate_deadend", "eliminate_unused_initializer",
    "eliminate_duplicate_initializer", "eliminate_nop_cast", "eliminate_nop_concat",
    "eliminate_nop_reshape", "eliminate_nop_transpose", "eliminate_nop_with_unit",
    "eliminate_consecutive_idempotent_ops", "fuse_consecutive_concats",
    "fuse_consecutive_slices", "fuse_consecutive_transposes",
    "fuse_consecutive_unsqueezes", "fuse_consecutive_reduce_unsqueeze",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high136_optimizer_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    authority = json.loads((HERE / "authority_audit.json").read_text())
    report: dict[str, object] = {"tasks": {}, "preliminary_lower": []}
    outdir = HERE / "optimizer_candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        source_data = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        source = onnx.load_model_from_string(source_data)
        base_cost = authority["tasks"][str(task)]["official_profile"]["cost"]
        variants = [(name, [name]) for name in PASSES]
        variants.extend((
            ("safe_eliminate", list(PASSES[:8])),
            ("safe_all", list(PASSES)),
        ))
        seen = {digest(source_data)}
        rows = []
        for label, passes in variants:
            row: dict[str, object] = {"task": task, "label": label, "passes": passes}
            try:
                model = onnxoptimizer.optimize(copy.deepcopy(source), passes)
                data = model.SerializeToString()
                sha = digest(data)
                row["sha256"] = sha
                if sha in seen:
                    row["status"] = "UNCHANGED_OR_DUPLICATE"
                    rows.append(row)
                    continue
                seen.add(sha)
                structural = SCAN.structural(copy.deepcopy(model))
                row["structural"] = structural
                if not structural.get("pass", False):
                    row["status"] = "STRUCTURAL_REJECT"
                else:
                    profile = SCAN.official_cost(data, f"high136_optimizer_{task:03d}_{label}")
                    row["declared_profile"] = profile
                    row["strict_lower"] = profile["cost"] < base_cost
                    row["status"] = "PRELIMINARY_LOWER" if row["strict_lower"] else "NOT_LOWER"
                    if row["strict_lower"]:
                        path = outdir / f"task{task:03d}_{label}_{sha[:12]}.onnx"
                        path.write_bytes(data)
                        row["path"] = str(path.relative_to(ROOT))
                        report["preliminary_lower"].append(row)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        report["tasks"][str(task)] = {
            "baseline_cost": base_cost,
            "unique_changed": len(seen) - 1,
            "rows": rows,
        }
        print(
            f"task{task:03d} changed={len(seen)-1} "
            f"lower={sum(1 for row in rows if row.get('strict_lower'))}", flush=True
        )
    (HERE / "optimizer_sweep.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
