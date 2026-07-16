#!/usr/bin/env python3
"""Extract and audit task216/285/226 from immutable 8009.46 authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (216, 285, 226)
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
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
    "high136_scan_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high136_audit_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def official_profile(task: int, model: onnx.ModelProto) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"high136_{task:03d}_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir,
            label=f"high136_task{task:03d}_authority", require_correct=False,
        )
    if result is None:
        raise RuntimeError(f"task{task:03d}: score_and_verify returned None")
    return result


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    current = HERE / "current"
    current.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": {},
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    for task, data in members.items():
        path = current / f"task{task:03d}.onnx"
        path.write_bytes(data)
        model = onnx.load_model_from_string(data)
        known = {
            label: AUDIT.known_config(task, data, data, disable, threads)
            for disable, threads, label in CONFIGS
        }
        row = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "serialized_bytes": len(data),
            "official_profile": official_profile(task, model),
            "declared_shape_cost_profile": SCAN.official_cost(
                data, f"high136_task{task:03d}_declared"
            ),
            "structural": SCAN.structural(copy.deepcopy(model)),
            "runtime_shape_trace": safe_trace(task, data),
            "inventory": SCAN.graph_inventory(copy.deepcopy(model)),
            "known_four_configs": known,
        }
        report["tasks"][str(task)] = row
        (HERE / f"task{task:03d}_graph.txt").write_text(
            onnx.printer.to_text(model) + "\n", encoding="utf-8"
        )
        print(
            f"task{task:03d} sha={row['sha256'][:12]} "
            f"official={row['official_profile']['cost']} "
            f"structural={row['structural'].get('pass')} "
            f"truthful={row['runtime_shape_trace'].get('truthful')} "
            f"known4={all(item.get('perfect', False) for item in known.values())}",
            flush=True,
        )
    (HERE / "authority_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
