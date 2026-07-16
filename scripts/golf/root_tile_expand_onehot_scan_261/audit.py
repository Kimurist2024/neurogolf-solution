#!/usr/bin/env python3
"""Fail-closed consistency audit for the Tile/Expand/OneHot census."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCAN = HERE / "scan.json"
AUTHORITY = ROOT / "submission.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(SCAN.read_text())
    failures: list[str] = []
    if data["authority"]["sha256"] != sha256(AUTHORITY):
        failures.append("authority SHA changed after scan")
    if data["authority"]["onnx_member_count"] != 400:
        failures.append("authority is not a complete 400-member ZIP")
    if data["census"] != {"Expand": 1, "OneHot": 4, "Tile": 4}:
        failures.append(f"unexpected census: {data['census']}")
    if data["occurrence_count"] != 9:
        failures.append("occurrence count is not nine")
    if data["target_tasks"] != [66, 133, 200, 233, 247, 300, 388]:
        failures.append("target task set changed")

    authority_gate = {}
    for task, row in data["rows"].items():
        profile = row["known_official_profile"]
        structure = row["structural"]
        exact = profile["right"] == profile["total"] and profile["wrong"] == 0
        clean_runtime = (
            profile["errors"] == 0
            and profile["nonfinite"] == 0
            and profile["output_shape_errors"] == 0
        )
        clean_structure = (
            structure["checker_full"]
            and structure["strict_shape_inference_data_prop"]
            and structure["standard_domains"]
            and not structure["functions"]
            and not structure["sparse_initializers"]
            and not structure["nested_graph_ops"]
            and not structure["banned_ops"]
            and not structure["conv_bias_findings"]
        )
        authority_gate[task] = {
            "known_exact": exact,
            "runtime_error_nonfinite_output_shape_clean": clean_runtime,
            "checker_strict_domain_ub_clean": clean_structure,
            "runtime_shapes_truthful": structure["runtime_shapes_truthful"],
            "runtime_shape_mismatch_count": structure["runtime_shape_mismatch_count"],
        }
        if not exact:
            failures.append(f"task{int(task):03d} authority known mismatch")
        if not clean_runtime:
            failures.append(f"task{int(task):03d} authority runtime fault")
        if not clean_structure:
            failures.append(f"task{int(task):03d} authority structural/UB fault")

    rewrites = [node for row in data["rows"].values() for node in row["special_nodes"]]
    onehot = [node for node in rewrites if node["op"] == "OneHot"]
    tile = [node for node in rewrites if node["op"] == "Tile"]
    expand = [node for node in rewrites if node["op"] == "Expand"]
    if any(node["equal_to_range_optimistic_cost_delta"] <= 0 for node in onehot):
        failures.append("a OneHot Equal lower bound is non-positive")
    if any(node["all_repeats_one"] or node["singleton_repeat_axes"] for node in tile):
        failures.append("a Tile unexpectedly meets a direct exact rewrite rule")
    if any(node["identity"] or node["duplicate_shape_initializers"] for node in expand):
        failures.append("Expand unexpectedly has an alias/shared-shape rewrite")
    if data["lower_survivors"] or data["candidate_files"] or data["winner"] is not None:
        failures.append("scan emitted a candidate despite an empty strict-lower set")

    result = {
        "decision": "NO_STRICT_LOWER_EXACT_REWRITE",
        "winner": None,
        "authority_sha256": data["authority"]["sha256"],
        "census": data["census"],
        "authority_gate": authority_gate,
        "onehot_optimistic_cost_deltas": {
            str(node["task"]): node["equal_to_range_optimistic_cost_delta"]
            for node in onehot
        },
        "tile_all_ones": 0,
        "tile_singleton_repeat_axes": 0,
        "expand_identity_or_shared_shape": 0,
        "lower_survivor_count": 0,
        "known_four_config_raw_gate_run": False,
        "fresh_two_seed_x1000_run": False,
        "gate_skip_reason": (
            "The cost-and-semantics screen has no strictly lower exact survivor to audit or fresh-test."
        ),
        "candidate_policy": {
            "approximation": False,
            "lookup": False,
            "shape_cloak": False,
            "candidate_created": False,
        },
        "failures": failures,
        "pass": not failures,
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
