#!/usr/bin/env python3
"""Re-run all safe optimizer profiles on the newly promoted task158 payload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import onnx
import onnxoptimizer


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "others" / "71407" / "task158.onnx"
SHARED = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_stage_fusion_latest_216/scan.py"
)


def load_shared():
    spec = importlib.util.spec_from_file_location("stage_fusion_216", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared scan")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    shared = load_shared()
    raw = SOURCE.read_bytes()
    model = onnx.load_model_from_string(raw)
    baseline = shared.profile(model, 158, "root222_base")
    rows = []
    candidates = HERE / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    for label, passes in shared.PASS_SETS.items():
        row = {"pass_set": label, "passes": passes, "baseline": baseline}
        try:
            candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
            encoded = candidate.SerializeToString()
            row["changed"] = encoded != raw
            if row["changed"]:
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(
                    candidate, strict_mode=True, data_prop=True
                )
                current = shared.profile(candidate, 158, f"root222_{label}")
                row["candidate"] = current
                row["strict_lower"] = (
                    "cost" in baseline
                    and "cost" in current
                    and current["cost"] < baseline["cost"]
                )
                if row["strict_lower"]:
                    output = candidates / f"task158_{label}.onnx"
                    onnx.save(candidate, output)
                    row["path"] = str(output.relative_to(ROOT))
                    row["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "baseline": baseline,
        "profiles": len(rows),
        "changed": sum(bool(row.get("changed")) for row in rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "source_sha256", "baseline", "profiles", "changed", "strict_lower"
    )}, indent=2))


if __name__ == "__main__":
    main()
