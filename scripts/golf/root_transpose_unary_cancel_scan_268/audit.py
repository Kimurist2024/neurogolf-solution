#!/usr/bin/env python3
"""Independent, fail-closed audit of the transpose/unary scan artifact."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
STAGE = ROOT / "others/71407"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_digest(rows: list[dict[str, Any]]) -> str:
    encoded = "\n".join(
        f"{row['path']}\0{row['sha256']}" for row in rows
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_stage_snapshot() -> list[dict[str, Any]]:
    return [
        {
            "task": int(path.stem.removeprefix("task")),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for path in sorted(STAGE.glob("task*.onnx"))
    ]


def count_broad_sandwiches(model: onnx.ModelProto) -> int:
    """Independently count T -> any 1-in/1-out node -> T raw graph paths."""
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers[name].append(index)
    count = 0
    for first in model.graph.node:
        if first.op_type != "Transpose" or len(first.output) != 1:
            continue
        first_output = first.output[0]
        for middle_index in consumers.get(first_output, []):
            middle = model.graph.node[middle_index]
            if not (
                len(middle.input) == 1
                and middle.input[0] == first_output
                and len(middle.output) == 1
                and middle.output[0]
            ):
                continue
            for second_index in consumers.get(middle.output[0], []):
                second = model.graph.node[second_index]
                if (
                    second.op_type == "Transpose"
                    and second.input
                    and second.input[0] == middle.output[0]
                ):
                    count += 1
    return count


def load_collections() -> dict[str, dict[int, onnx.ModelProto]]:
    authority: dict[int, onnx.ModelProto] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for name in archive.namelist():
            if name.endswith(".onnx"):
                task = int(Path(name).stem.removeprefix("task"))
                authority[task] = onnx.load_model_from_string(archive.read(name))
    stage = {
        int(path.stem.removeprefix("task")): onnx.load(path)
        for path in STAGE.glob("task*.onnx")
    }
    composite = dict(authority)
    composite.update(stage)
    return {"authority": authority, "stage": stage, "composite_best": composite}


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    scan_path = HERE / "scan.json"
    scan = json.loads(scan_path.read_text())
    candidates_path = HERE / "candidates.json"
    candidates = json.loads(candidates_path.read_text())
    checks: list[str] = []
    current_stage = current_stage_snapshot()
    current_manifest_sha = sha256(STAGE / "MANIFEST.json")

    require(
        scan["authority"]["sha256"] == sha256(AUTHORITY),
        "authority SHA-256 matches the scanned submission.zip",
        checks,
    )
    require(scan["authority"]["models"] == 400, "authority census covers 400 models", checks)
    require(
        scan["stage"]["snapshot"] == current_stage,
        "staged root ONNX snapshot matches scan.json byte-for-byte",
        checks,
    )
    require(
        scan["stage"]["snapshot_digest"] == snapshot_digest(current_stage),
        "staged snapshot digest recomputes exactly",
        checks,
    )
    require(
        scan["stage"]["manifest_sha256"] == current_manifest_sha,
        "others/71407 manifest SHA-256 matches",
        checks,
    )
    require(len(current_stage) == 19, "staged census covers 19 active root ONNX files", checks)
    require(scan["composite_best"]["models"] == 400, "composite-best census covers 400 models", checks)
    require(
        scan["composite_best"]["staged_overrides"] == [row["task"] for row in current_stage],
        "composite-best staged overrides equal the live staged task set",
        checks,
    )

    independently_loaded = load_collections()
    for collection in ("authority", "stage", "composite_best"):
        summary = scan["summaries"][collection]
        independent_paths = sum(
            count_broad_sandwiches(model)
            for model in independently_loaded[collection].values()
        )
        require(
            len(independently_loaded[collection]) == summary["models"],
            f"{collection}: independent model count matches scanner summary",
            checks,
        )
        require(
            independent_paths == summary["any_one_input_output_sandwiches"] == 0,
            f"{collection}: independent raw-graph rescan confirms zero broad sandwiches",
            checks,
        )
        require(summary["checker_failures"] == 0, f"{collection}: all sources pass full ONNX checker", checks)
        require(
            summary["any_one_input_output_sandwiches"] == 0,
            f"{collection}: no Transpose-oneInputOneOutput-Transpose path exists",
            checks,
        )
        require(summary["sandwiches"] == 0, f"{collection}: no safe unary sandwich exists", checks)
        require(summary["eligible"] == 0, f"{collection}: no eligible cancellation exists", checks)

    require(scan["sandwiches"] == [], "composite sandwich detail list is empty", checks)
    require(scan["eligible_patterns"] == [], "eligible pattern list is empty", checks)
    require(scan["candidate_rows"] == [], "no in-memory candidate was produced", checks)
    require(scan["strict_lower_candidates"] == [], "strict-lower candidate list is empty", checks)
    require(scan["winner"] is None, "winner is null", checks)
    require(
        candidates["structural_candidates"] == []
        and candidates["strict_lower_candidates"] == []
        and candidates["winner"] is None,
        "standalone candidate ledger records an empty result",
        checks,
    )
    require(
        candidates["execution_gate"] == scan["candidate_gate"]
        and candidates["policy"] == scan["policy"],
        "standalone candidate ledger matches scan gates and policy",
        checks,
    )
    require(
        not any(scan["candidate_gate"][key] for key in (
            "official_known_four_config_raw",
            "fresh_independent_2000",
            "full_strict_profile_ub0",
        )),
        "runtime/cost gates are explicitly skipped because the structural candidate set is empty",
        checks,
    )
    require(
        scan["policy"] == {
            "private_zero_candidate": False,
            "runtime_shape_cloak_candidate": False,
            "lookup_candidate": False,
            "root_or_others71407_modified": False,
        },
        "policy record rejects private-zero, runtime cloak, lookup, and protected-path writes",
        checks,
    )

    payload = {
        "audit": "root_transpose_unary_cancel_scan_268",
        "pass": True,
        "decision": "NO_TRANSPOSE_UNARY_TRANSPOSE_SANDWICH",
        "scan_sha256": sha256(scan_path),
        "scanner_sha256": sha256(HERE / "scan.py"),
        "candidates_sha256": sha256(candidates_path),
        "authority_sha256": sha256(AUTHORITY),
        "stage_snapshot_digest": snapshot_digest(current_stage),
        "stage_manifest_sha256": current_manifest_sha,
        "checks": checks,
        "candidate_execution_gate": {
            "required": False,
            "reason": scan["candidate_gate"]["skip_reason"],
        },
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
