#!/usr/bin/env python3
"""Independent Wave15 identity, structure, runtime-shape, and known audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WAVE15 = HERE.parent / "submission_7999.13_wave15_candidate_meta.zip"
EXPECTED = {
    226: ("342ff4b0df090df3cb1fdea435049e05f9e317f4775af82a14ded63b2a490c13", 399),
    338: ("edcac049616e90e42b848d1a719b3af7a4a078b5d1180a3cdf0ecf60e340a01d", 426),
}
PATHS = {
    226: HERE.parent / "lane_a5" / "baseline" / "task226.onnx",
    338: HERE.parent / "lane_a5" / "baseline" / "task338.onnx",
}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def load_shared_audit():
    path = HERE.parent / "lane_b15" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("b21_shared_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    shared = load_shared_audit()
    rows = []
    with zipfile.ZipFile(WAVE15) as archive:
        for task in (226, 338):
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            expected_hash, expected_cost = EXPECTED[task]
            path = PATHS[task]
            if sha256_bytes(data) != expected_hash or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError(f"task{task:03d} Wave15 identity mismatch")
            model = onnx.load(path)
            structure = shared.structural(copy.deepcopy(model))
            try:
                runtime_shape = shared.trace_runtime_shapes(copy.deepcopy(model), task)
            except Exception as exc:  # noqa: BLE001 - fail-closed audit evidence
                runtime_shape = {"shape_cloak": None, "trace_error": f"{type(exc).__name__}: {exc}"}
            known = shared.known_dual(copy.deepcopy(model), task)
            memory, params, cost = cost_of(str(path))
            if int(cost) != expected_cost:
                raise RuntimeError(f"task{task:03d} cost mismatch: {cost}")
            rows.append(
                {
                    "task": task,
                    "wave15_member": member,
                    "sha256": expected_hash,
                    "cost": int(cost),
                    "memory": int(memory),
                    "params": int(params),
                    "structure": structure,
                    "runtime_shape": runtime_shape,
                    "known_dual": known,
                }
            )
    payload = {
        "wave15": str(WAVE15.relative_to(ROOT)),
        "wave15_sha256": hashlib.sha256(WAVE15.read_bytes()).hexdigest(),
        "rows": rows,
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": row["task"],
                        "cost": row["cost"],
                        "shape_cloak": row["runtime_shape"].get("shape_cloak"),
                        "shape_mismatches": len(
                            row["runtime_shape"].get("declared_runtime_mismatches", [])
                        ),
                        "trace_error": row["runtime_shape"].get("trace_error"),
                        "known": row["known_dual"],
                    }
                    for row in rows
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
