#!/usr/bin/env python3
"""Record current known behavior in the four required ORT configurations."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
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
    "high133_current_audit_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def main() -> int:
    report: dict[str, object] = {"tasks": {}}
    for task in (69, 187, 277):
        data = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        configs = {}
        for disable, threads, label in CONFIGS:
            configs[label] = AUDIT.known_config(task, data, data, disable, threads)
            print(
                f"task{task:03d} {label} perfect={configs[label].get('perfect')} "
                f"right={configs[label].get('candidate_right')} "
                f"session_error={bool(configs[label].get('session_error'))}",
                flush=True,
            )
        report["tasks"][str(task)] = configs
    (HERE / "current_known_four_configs.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
