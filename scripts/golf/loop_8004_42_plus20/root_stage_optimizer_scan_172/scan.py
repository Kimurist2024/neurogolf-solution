#!/usr/bin/env python3
"""Re-golf staged SOUND candidates with semantics-preserving ONNX passes."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxoptimizer

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

STAGE = ROOT / "others/71407"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
PASSES = [
    name for name in onnxoptimizer.get_available_passes()
    if name.startswith("eliminate_") or name.startswith("fuse_consecutive_")
]


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"stageopt172_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(STAGE.glob("task*.onnx")):
        task = int(path.stem.removeprefix("task"))
        model = onnx.load(path)
        baseline = profile(model, task)
        variants = [(name, [name]) for name in PASSES]
        variants.append(("all_eliminate_fuse", PASSES))
        for label, passes in variants:
            row: dict[str, Any] = {"task": task, "pass": label, "baseline": baseline}
            try:
                candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                current = profile(candidate, task)
                row["candidate"] = current
                row["strict_lower"] = current["cost"] < baseline["cost"]
                if row["strict_lower"]:
                    output = CANDIDATES / f"task{task:03d}_{label}.onnx"
                    onnx.save(candidate, output)
                    row["path"] = str(output.relative_to(ROOT))
                    row["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    payload = {
        "stage": str(STAGE.relative_to(ROOT)),
        "passes": PASSES,
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
