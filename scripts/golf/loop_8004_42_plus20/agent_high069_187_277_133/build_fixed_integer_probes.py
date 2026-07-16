#!/usr/bin/env python3
"""Build current-only exact fixed-integer probes for the two Shape/Size chains."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high133_fixed_cost_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high133_fixed_runtime_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_producer_with_initializer(
    model: onnx.ModelProto,
    output: str,
    value: np.ndarray,
) -> None:
    producers = [node for node in model.graph.node if output in node.output]
    if len(producers) != 1:
        raise RuntimeError(f"expected one producer for {output}, got {len(producers)}")
    model.graph.node.remove(producers[0])
    model.graph.initializer.append(numpy_helper.from_array(value, name=output))


def safe_profile(data: bytes, label: str) -> dict[str, object]:
    try:
        return SCAN.official_cost(data, label)
    except Exception as exc:  # noqa: BLE001
        return {"memory": -1, "params": -1, "cost": -1, "error": f"{type(exc).__name__}: {exc}"}


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    outdir = HERE / "rejected_probes"
    outdir.mkdir(parents=True, exist_ok=True)
    specs = []

    task69 = onnx.load(HERE / "current/task069.onnx")
    size_source = next(node.input[0] for node in task69.graph.node if "c10_dyn" in node.output)
    source_init = next(init for init in task69.graph.initializer if init.name == size_source)
    source_array = numpy_helper.to_array(source_init)
    if source_array.size != 10:
        raise RuntimeError(f"unexpected codes size: {source_array.size}")
    replace_producer_with_initializer(task69, "c10_dyn", np.asarray(10, dtype=np.int64))
    specs.append((69, "size_codes_exact10", task69, {
        "proof": "Size(codes_i8)=10 because authority initializer shape is [1,10,1,1]",
        "range": [10, 10],
        "overflow": "none: exact int64 scalar replacement",
        "rounding": "none: integer Size",
    }))

    task187 = onnx.load(HERE / "current/task187.onnx")
    shape_source = next(node.input[0] for node in task187.graph.node if "seed_shape" in node.output)
    shape_init = next(init for init in task187.graph.initializer if init.name == shape_source)
    shape_array = numpy_helper.to_array(shape_init)
    replace_producer_with_initializer(
        task187, "seed_shape", np.asarray(shape_array.shape, dtype=np.int64)
    )
    specs.append((187, "shape_seed_exact_rank4", task187, {
        "proof": "Shape(shape_seed)=[4] because authority initializer shape is [4]",
        "range": [4, 4],
        "overflow": "none: exact int64 vector replacement",
        "rounding": "none: integer Shape",
    }))

    rows = []
    for task, label, model, proof in specs:
        data = model.SerializeToString()
        digest = sha(data)
        path = outdir / f"task{task:03d}_{label}_{digest[:12]}.onnx"
        path.write_bytes(data)
        static = SCAN.structural(copy.deepcopy(model))
        profile = safe_profile(data, f"high133_task{task:03d}_{label}") if static.get("pass") else None
        trace = safe_trace(task, data) if static.get("pass") else None
        row = {
            "task": task,
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "proof": proof,
            "structural": static,
            "official_profile": profile,
            "runtime_shape_trace": trace,
            "decision": "REJECT",
        }
        rows.append(row)
        print(
            f"task{task:03d} {label} structural={static.get('pass')} "
            f"cost={None if profile is None else profile.get('cost')} "
            f"truthful={None if trace is None else trace.get('truthful')}",
            flush=True,
        )
    (HERE / "fixed_integer_probes.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
