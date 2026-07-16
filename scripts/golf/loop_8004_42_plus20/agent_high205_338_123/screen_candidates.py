#!/usr/bin/env python3
"""Static, cost, runtime-shape, and four-config known screening."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(
    "high205338_audit_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
SCAN = load_module(
    "high205338_scan_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
CONV = load_module("high205338_conv", ROOT / "scripts/golf/check_conv_bias.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    entries = [
        (205, "current", HERE / "current/task205.onnx"),
        (338, "current", HERE / "current/task338.onnx"),
        (205, "rowpow_selu", HERE / "candidates/task205_rowpow_selu.onnx"),
        (205, "boxcast_gain01902", HERE / "probes/task205_boxcast_gain01902.onnx"),
        (205, "rowpow_selu_boxcast_gain01902", HERE / "probes/task205_rowpow_selu_boxcast_gain01902.onnx"),
        (205, "rowpow_selu_cmneg", HERE / "probes/task205_rowpow_selu_cmneg.onnx"),
        (205, "rowpow_selu_gaincolq", HERE / "probes/task205_rowpow_selu_gaincolq.onnx"),
        (338, "cast_truthful_probe", HERE / "probes/task338_cast_truthful_probe.onnx"),
    ]
    bases = {task: (HERE / f"current/task{task:03d}.onnx").read_bytes() for task in (205, 338)}
    report: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "entries": [],
    }
    base_costs: dict[int, dict[str, int]] = {}
    for task in (205, 338):
        base_costs[task] = SCAN.official_cost(bases[task], f"high123_task{task:03d}_current")
    for task, label, path in entries:
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        static = SCAN.structural(copy.deepcopy(model))
        conv = CONV.check_model(copy.deepcopy(model))
        try:
            cost = SCAN.official_cost(data, f"high123_task{task:03d}_{label}")
        except Exception as exc:  # noqa: BLE001
            cost = {"memory": -1, "params": -1, "cost": -1, "error": f"{type(exc).__name__}: {exc}"}
        trace = AUDIT.direct_trace(task, data)
        row = {
            "task": task,
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "static": static,
            "conv_bias_findings": [list(item) for item in conv],
            "official_profile": cost,
            "authority_profile": base_costs[task],
            "strict_lower": cost["cost"] >= 0 and cost["cost"] < base_costs[task]["cost"],
            "runtime_shape_trace": trace,
            "known_four_configs": {},
        }
        if label != "current" and row["strict_lower"]:
            for disable, threads, config_label in CONFIGS:
                row["known_four_configs"][config_label] = AUDIT.known_config(
                    task, bases[task], data, disable, threads
                )
        report["entries"].append(row)
        print(
            f"task{task:03d} {label} cost={cost['cost']} lower={row['strict_lower']} "
            f"truthful={trace.get('truthful')} configs={len(row['known_four_configs'])}",
            flush=True,
        )
    (HERE / "screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
