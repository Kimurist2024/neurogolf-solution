#!/usr/bin/env python3
"""Independent official-profile/runtime audit for strict-lower scan155 rows."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (71, 133, 216, 285, 388)
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


AUDIT = load_module(
    "attr155_exactb_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
SCAN = load_module(
    "attr155_exactb_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(task: int, data: bytes, label: str) -> dict:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"attr155_{task:03d}_", dir="/tmp") as wd:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, wd, label=label, require_correct=False
        )
    if result is None:
        raise RuntimeError(f"{label}: score_and_verify returned None")
    return result


def candidate_config(task: int, data: bytes, disable: bool, threads: int) -> dict:
    examples = scoring.load_examples(task)
    ordered = [
        (split, index, example)
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[split])
    ]
    row = {
        "total": len(ordered), "right": 0, "runtime_errors": 0,
        "nonfinite_values": 0, "shapes": [], "first_failure": None,
    }
    try:
        session = AUDIT.make_session(data, disable, threads)
    except Exception as exc:
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["perfect"] = False
        return row
    for split, index, example in ordered:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            row["first_failure"] = row["first_failure"] or {
                "split": split, "index": index, "error": "convert_to_numpy returned None",
            }
            continue
        try:
            value = AUDIT.run(session, benchmark)
        except Exception as exc:
            row["runtime_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "split": split, "index": index,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        shape = list(value.shape)
        if shape not in row["shapes"]:
            row["shapes"].append(shape)
        row["nonfinite_values"] += int(value.size - np.count_nonzero(np.isfinite(value)))
        correct = np.array_equal(value > 0, benchmark["output"].astype(bool))
        row["right"] += int(correct)
        if not correct:
            row["first_failure"] = row["first_failure"] or {
                "split": split, "index": index, "error": "threshold mismatch",
            }
    row["perfect"] = (
        row["right"] == row["total"]
        and row["runtime_errors"] == 0
        and row["nonfinite_values"] == 0
    )
    return row


def safe_trace(task: int, data: bytes) -> dict:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    if sha(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    scan = json.loads((HERE / "scan.json").read_text())
    leads = {row["task"]: row for row in scan["rows"] if row.get("strict_lower")}
    result = {"authority_sha256": AUTHORITY_SHA256, "tasks": {}}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            base = archive.read(f"task{task:03d}.onnx")
            lead = leads[task]
            cand_path = ROOT / lead["path"]
            cand = cand_path.read_bytes()
            row = {
                "candidate_path": lead["path"],
                "authority_sha256": sha(base),
                "candidate_sha256": sha(cand),
                "declared_profiles": {
                    "authority": SCAN.official_cost(base, f"attr155_decl_base_{task:03d}"),
                    "candidate": SCAN.official_cost(cand, f"attr155_decl_cand_{task:03d}"),
                },
                "official_profiles": {
                    "authority": official(task, base, f"attr155_base_{task:03d}"),
                    "candidate": official(task, cand, f"attr155_cand_{task:03d}"),
                },
                "structural": SCAN.structural(copy.deepcopy(onnx.load_model_from_string(cand))),
                "runtime_shape_trace": safe_trace(task, cand),
                "known_candidate": {
                    label: candidate_config(task, cand, disable, threads)
                    for disable, threads, label in CONFIGS
                },
                "known_differential": {
                    label: AUDIT.known_config(task, base, cand, disable, threads)
                    for disable, threads, label in CONFIGS
                },
            }
            row["official_strict_lower"] = (
                row["official_profiles"]["candidate"]["cost"]
                < row["official_profiles"]["authority"]["cost"]
            )
            result["tasks"][str(task)] = row
            print(
                f"task{task:03d} official "
                f"{row['official_profiles']['authority']['cost']}->"
                f"{row['official_profiles']['candidate']['cost']} "
                f"truthful={row['runtime_shape_trace'].get('truthful')} "
                f"known4={all(x.get('perfect', False) for x in row['known_candidate'].values())}",
                flush=True,
            )
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
