#!/usr/bin/env python3
"""Inventory standalone 71406/71409 ONNX candidates against immutable 8009.46."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71406"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASK_RE = re.compile(r"task(\d{3})")
KNOWN_BLACK = {23, 185, 198, 201, 208, 396}
PRIVATE_ZERO = {
    9, 15, 35, 44, 48, 66, 72, 77, 86, 90, 96, 101, 102, 133, 134,
    138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187, 192, 196,
    202, 205, 209, 216, 219, 222, 233, 246, 255, 277, 285, 286, 302,
    325, 346, 361, 365, 366, 372, 377, 379, 393, 396,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOLS = load_module(
    "residual117_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_payload(data: bytes) -> str:
    model = onnx.load_model_from_string(data)
    model.producer_name = ""
    model.producer_version = ""
    model.domain = ""
    model.model_version = 0
    model.doc_string = ""
    del model.metadata_props[:]
    model.graph.name = ""
    model.graph.doc_string = ""
    del model.graph.value_info[:]
    for node in model.graph.node:
        node.name = ""
        node.doc_string = ""
    for value in list(model.graph.input) + list(model.graph.output):
        value.doc_string = ""
    return digest(model.SerializeToString(deterministic=True))


def params_floor(data: bytes) -> int:
    model = onnx.load_model_from_string(data)
    total = 0
    for item in model.graph.initializer:
        total += int(np.asarray(numpy_helper.to_array(item)).size)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                total += int(np.asarray(numpy_helper.to_array(attribute.t)).size)
            elif attribute.name == "value_floats":
                total += len(attribute.floats)
            elif attribute.name == "value_ints":
                total += len(attribute.ints)
    return total


def all_scores() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = TASK_RE.fullmatch(row["task"])
            if match:
                rows[int(match.group(1))] = {
                    "hash8": row["hash"],
                    "cost": int(row["cost"]),
                    "score": float(row["score"]),
                }
    return rows


def main() -> int:
    authority_zip = AUTHORITY.read_bytes()
    if digest(authority_zip) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 hash changed")
    score_rows = all_scores()
    paths = sorted(
        path
        for path in SOURCE.rglob("*.onnx")
        if "submission_task343_cost175" not in path.parts
    )
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    invalid_names: list[str] = []
    for path in paths:
        match = TASK_RE.search(path.name)
        if match is None:
            invalid_names.append(str(path.relative_to(ROOT)))
            continue
        task = int(match.group(1))
        data = path.read_bytes()
        sha = digest(data)
        key = (task, sha)
        grouped.setdefault(key, {"task": task, "sha256": sha, "data": data, "sources": []})
        grouped[key]["sources"].append(str(path.relative_to(ROOT)))

    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = {task: archive.read(f"task{task:03d}.onnx") for task, _ in grouped}
    authority_sha = {task: digest(data) for task, data in authority.items()}
    authority_canonical = {task: canonical_payload(data) for task, data in authority.items()}

    records: list[dict[str, Any]] = []
    for (task, sha), item in sorted(grouped.items()):
        data = item["data"]
        row: dict[str, Any] = {
            "task": task,
            "sha256": sha,
            "sources": item["sources"],
            "source_count": len(item["sources"]),
            "serialized_bytes": len(data),
            "authority_sha256": authority_sha[task],
            "same_sha_as_authority": sha == authority_sha[task],
            "same_canonical_graph_as_authority": canonical_payload(data) == authority_canonical[task],
            "known_black": task in KNOWN_BLACK,
            "private_zero_history": task in PRIVATE_ZERO,
            "all_scores_authority": score_rows.get(task),
            "params_floor": None,
            "structural": None,
            "candidate_profile": None,
            "authority_profile": None,
            "strict_lower": False,
            "status": None,
        }
        if row["same_sha_as_authority"]:
            row["status"] = "EXCLUDE_ALREADY_FIXED_SAME_SHA"
            records.append(row)
            continue
        if row["known_black"]:
            row["status"] = "EXCLUDE_KNOWN_BLACK"
            records.append(row)
            continue
        try:
            model = onnx.load_model_from_string(data)
            structural = TOOLS.structural(model)
        except Exception as exc:  # noqa: BLE001
            structural = {"pass": False, "reasons": ["load"], "error": f"{type(exc).__name__}: {exc}"}
        row["structural"] = structural
        if not structural.get("pass", False):
            row["status"] = "EXCLUDE_STATIC_UB_OR_SHAPE"
            records.append(row)
            continue
        floor = params_floor(data)
        row["params_floor"] = floor
        current_hint = score_rows.get(task, {}).get("cost")
        if current_hint is not None and floor >= current_hint:
            row["status"] = "EXCLUDE_PARAM_FLOOR_NOT_LOWER"
            records.append(row)
            continue
        try:
            candidate_profile = TOOLS.official_cost(data, f"res117_task{task:03d}_candidate")
        except Exception as exc:  # noqa: BLE001
            row["profile_error"] = f"{type(exc).__name__}: {exc}"
            row["status"] = "EXCLUDE_PROFILE_RUNTIME"
            records.append(row)
            continue
        row["candidate_profile"] = candidate_profile
        if candidate_profile["cost"] < 0:
            row["status"] = "EXCLUDE_PROFILE_RUNTIME"
            records.append(row)
            continue
        # Reprofile the immutable member for every candidate that appears lower
        # than the CSV hint or for which no hint exists.
        if current_hint is None or candidate_profile["cost"] < current_hint:
            try:
                base_profile = TOOLS.official_cost(authority[task], f"res117_task{task:03d}_authority")
            except Exception as exc:  # noqa: BLE001
                row["authority_profile_error"] = f"{type(exc).__name__}: {exc}"
                row["status"] = "EXCLUDE_AUTHORITY_PROFILE_RUNTIME"
                records.append(row)
                continue
            row["authority_profile"] = base_profile
            row["strict_lower"] = candidate_profile["cost"] < base_profile["cost"]
            row["cost_reduction"] = base_profile["cost"] - candidate_profile["cost"]
            row["status"] = "STRICT_LOWER_NEEDS_RUNTIME" if row["strict_lower"] else "EXCLUDE_NOT_LOWER_REPROFILED"
        else:
            row["status"] = "EXCLUDE_NOT_LOWER_PROFILED"
        records.append(row)
        if row["status"] == "STRICT_LOWER_NEEDS_RUNTIME":
            print(
                f"lower task{task:03d} {sha[:12]} candidate={candidate_profile['cost']} "
                f"authority={row['authority_profile']['cost']} sources={len(item['sources'])}",
                flush=True,
            )

    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[row["status"]] += 1
    strict = [row for row in records if row["strict_lower"]]
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "source_root": str(SOURCE.relative_to(ROOT)),
        "discovery": {
            "standalone_onnx_files": len(paths),
            "unique_task_sha_pairs": len(records),
            "duplicate_source_files": len(paths) - len(records),
            "invalid_names": invalid_names,
        },
        "policy": {
            "known_black": sorted(KNOWN_BLACK),
            "private_zero_history": sorted(PRIVATE_ZERO),
        },
        "status_counts": dict(sorted(counts.items())),
        "strict_lower": strict,
        "records": records,
    }
    (HERE / "inventory_screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"discovery": report["discovery"], "status_counts": report["status_counts"], "strict_lower": len(strict)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
