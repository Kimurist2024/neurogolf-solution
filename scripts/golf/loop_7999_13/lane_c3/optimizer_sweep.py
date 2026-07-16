#!/usr/bin/env python3
"""Try every single ONNX optimizer pass on each exact C3 baseline."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxoptimizer
import onnxruntime

onnxruntime.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402
import scan_existing as scan  # noqa: E402

TASK_COSTS = {36: 325, 44: 1087, 90: 1050, 125: 1050, 205: 1042, 319: 1023, 361: 968}


def digest(model: onnx.ModelProto) -> str:
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def main() -> int:
    root = HERE / "optimizer_variants"
    root.mkdir(parents=True, exist_ok=True)
    output = {"tasks": {}}
    for task, base_cost in TASK_COSTS.items():
        model = onnx.load(HERE / "base" / f"task{task:03d}.onnx")
        base_hash = digest(model)
        seen = {base_hash}
        rows = []
        for pass_name in onnxoptimizer.get_available_passes():
            row = {"pass": pass_name}
            try:
                candidate = onnxoptimizer.optimize(model, [pass_name])
                sha = digest(candidate)
                row["sha256"] = sha
                if sha in seen:
                    row["status"] = "duplicate_or_base"
                    rows.append(row)
                    continue
                seen.add(sha)
                task_dir = root / f"task{task:03d}"
                task_dir.mkdir(parents=True, exist_ok=True)
                path = task_dir / f"{pass_name}.onnx"
                onnx.save(candidate, path)
                row["path"] = str(path.relative_to(ROOT))
                ok, reason, checked = scan.static_check(path.read_bytes())
                row["static"] = reason
                if not ok or checked is None:
                    row["status"] = "static_reject"
                    rows.append(row)
                    continue
                floor, memory_floor, params = scan.scanner.static_cost_floor(checked)
                row.update(static_cost_floor=floor, static_memory_floor=memory_floor, param_floor=params)
                with tempfile.TemporaryDirectory(prefix=f"c3_opt_{task:03d}_") as workdir:
                    scored = scoring.score_and_verify(
                        checked, task, workdir, label=pass_name, require_correct=False
                    )
                row["score"] = scored
                if scored is None:
                    row["status"] = "unscorable"
                elif scored["correct"] and scored["cost"] < base_cost:
                    row["status"] = "cheaper_visible_correct"
                elif scored["correct"]:
                    row["status"] = "not_cheaper"
                else:
                    row["status"] = "visible_wrong"
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        output["tasks"][str(task)] = {
            "base_cost": base_cost,
            "base_sha256": base_hash,
            "unique_outputs_including_base": len(seen),
            "rows": rows,
        }
        print(
            f"task{task:03d}: unique={len(seen)} "
            f"cheaper={sum(r.get('status') == 'cheaper_visible_correct' for r in rows)}",
            flush=True,
        )
    (HERE / "optimizer_sweep.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
