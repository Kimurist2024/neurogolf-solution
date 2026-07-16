#!/usr/bin/env python3
"""Independent byte/metadata/model audit of Wave1 -> Wave3."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOOP = HERE.parent
BASE = ROOT / "submission_base_8003.40.zip"
WAVE1 = LOOP / "submission_8003.40_wave1_policy95_meta.zip"
WAVE3 = LOOP / "submission_8003.40_wave3_safe_meta.zip"
TASK109_FILE = LOOP / "agent_archive_resume/candidates/task109.onnx"
MAX_MODEL_BYTES = int(1.44 * 1024 * 1024)
TASK_RE = re.compile(r"task[_-]?(\d{1,3})\.onnx$", re.IGNORECASE)

sys.path.insert(0, str(ROOT))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def task_number(name: str) -> int | None:
    match = TASK_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def invariant_zipinfo(info: zipfile.ZipInfo) -> dict[str, object]:
    """Metadata fields independent of replacement payload/physical offset."""
    return {
        "filename": info.filename,
        "date_time": list(info.date_time),
        "compress_type": info.compress_type,
        "comment_hex": info.comment.hex(),
        "extra_hex": info.extra.hex(),
        "create_system": info.create_system,
        "create_version": info.create_version,
        "extract_version": info.extract_version,
        "reserved": info.reserved,
        "flag_bits": info.flag_bits,
        "volume": info.volume,
        "internal_attr": info.internal_attr,
        "external_attr": info.external_attr,
    }


def compressed_payload(path: Path, info: zipfile.ZipInfo) -> bytes:
    blob = path.read_bytes()
    offset = info.header_offset
    header = struct.unpack_from("<IHHHHHIIIHH", blob, offset)
    if header[0] != 0x04034B50:
        raise ValueError(f"bad local header for {info.filename}")
    filename_length, extra_length = header[-2:]
    start = offset + 30 + filename_length + extra_length
    return blob[start : start + info.compress_size]


def value_info_map(model: onnx.ModelProto) -> dict[str, dict[str, object]]:
    result = {}
    for item in model.graph.value_info:
        tensor = item.type.tensor_type
        result[item.name] = {
            "elem_type": tensor.elem_type,
            "shape": [
                int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
                for dim in tensor.shape.dim
            ],
            "doc_string": item.doc_string,
        }
    return result


def deterministic_without_value_info(model: onnx.ModelProto) -> bytes:
    clone = copy.deepcopy(model)
    del clone.graph.value_info[:]
    return clone.SerializeToString(deterministic=True)


def main() -> None:
    expected_shas = {
        "base": "9bb795b4a2945882e98071350ee8333a914f27552ac8935570a20e2a57afd36f",
        "wave1": "c829af1a2928fbbcdae3d13a38a8f38777e4c38c6a08c2c5ae51a3c4f0bd2c49",
        "wave3": "338b6c968bb345780a570ec849f17b2fc0c1233c5bd0a000c67a035aeafb0cd7",
    }
    actual_shas = {
        "base": sha256_file(BASE),
        "wave1": sha256_file(WAVE1),
        "wave3": sha256_file(WAVE3),
    }

    with zipfile.ZipFile(WAVE1) as left, zipfile.ZipFile(WAVE3) as right:
        left_infos = left.infolist()
        right_infos = right.infolist()
        left_names = [item.filename for item in left_infos]
        right_names = [item.filename for item in right_infos]
        left_by_name = {item.filename: item for item in left_infos}
        right_by_name = {item.filename: item for item in right_infos}
        if set(left_by_name) != set(right_by_name):
            raise RuntimeError("Wave1 and Wave3 member-name sets differ")

        changed = []
        unchanged_payload_count = 0
        unchanged_compressed_count = 0
        metadata_mismatches = []
        member_records = []
        for name in left_names:
            left_info = left_by_name[name]
            right_info = right_by_name[name]
            before = left.read(name)
            after = right.read(name)
            payload_equal = before == after
            compressed_equal = compressed_payload(WAVE1, left_info) == compressed_payload(WAVE3, right_info)
            metadata_equal = invariant_zipinfo(left_info) == invariant_zipinfo(right_info)
            task = task_number(name)
            if payload_equal:
                unchanged_payload_count += 1
                unchanged_compressed_count += int(compressed_equal)
            else:
                changed.append({
                    "name": name,
                    "task": task,
                    "wave1_sha256": sha256_bytes(before),
                    "wave3_sha256": sha256_bytes(after),
                    "wave1_size": len(before),
                    "wave3_size": len(after),
                })
            if not metadata_equal:
                metadata_mismatches.append(name)
            member_records.append((name, task, right_info, after))

        right_tasks = [task for _, task, _, _ in member_records if task is not None]
        over_limit = [
            {"name": name, "bytes": len(payload)}
            for name, task, _, payload in member_records
            if task is not None and len(payload) > MAX_MODEL_BYTES
        ]
        max_member = max(
            ({"name": name, "bytes": len(payload)} for name, task, _, payload in member_records if task is not None),
            key=lambda item: item["bytes"],
        )
        integrity_error = right.testzip()
        archive_comment_equal = left.comment == right.comment

        bias_issues = []
        parse_errors = []
        for name, task, _, payload in member_records:
            if task is None:
                continue
            try:
                model = onnx.load_model_from_string(payload)
                issues = check_conv_bias(model)
                if issues:
                    bias_issues.append({"task": task, "name": name, "issues": [list(x) for x in issues]})
            except Exception as exc:  # noqa: BLE001
                parse_errors.append({"task": task, "name": name, "error": repr(exc)})

        task109_names = [name for name, task, _, _ in member_records if task == 109]
        if len(task109_names) != 1:
            raise RuntimeError(f"expected exactly one task109: {task109_names}")
        task109_name = task109_names[0]
        before109 = left.read(task109_name)
        after109 = right.read(task109_name)

    baseline109 = onnx.load_model_from_string(before109)
    candidate109 = onnx.load_model_from_string(after109)
    onnx.checker.check_model(candidate109, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(candidate109, strict_mode=True, data_prop=True)
    inferred_static = all(
        dim.HasField("dim_value") and dim.dim_value > 0
        for item in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        for dim in item.type.tensor_type.shape.dim
    )
    baseline_vi = value_info_map(baseline109)
    candidate_vi = value_info_map(candidate109)
    differing_vi = []
    for name in sorted(set(baseline_vi) | set(candidate_vi)):
        if baseline_vi.get(name) != candidate_vi.get(name):
            differing_vi.append({"name": name, "wave1": baseline_vi.get(name), "wave3": candidate_vi.get(name)})
    computational_payload_equal = (
        deterministic_without_value_info(baseline109)
        == deterministic_without_value_info(candidate109)
    )

    independent_external = json.loads((HERE / "independent_external500.json").read_text())
    fresh = json.loads((HERE / "fresh_review.json").read_text())
    wave1_compare = json.loads((LOOP / "wave1_compare.json").read_text())
    independently_summed_wave1_gain = sum(
        math.log(item["baseline"]["cost"] / item["candidate"]["cost"])
        for item in wave1_compare["decisions"]
    )
    external_decision = independent_external["decision"]
    task109_gain = math.log(
        external_decision["baseline"]["cost"] / external_decision["candidate"]["cost"]
    )
    projected = 8003.40 + independently_summed_wave1_gain + task109_gain

    gates = {
        "sha_matches_expected": actual_shas == expected_shas,
        "wave1_to_wave3_changed_only_task109": [item["task"] for item in changed] == [109],
        "other_399_uncompressed_payloads_equal": unchanged_payload_count == 399,
        "other_399_compressed_payloads_equal": unchanged_compressed_count == 399,
        "member_order_equal": left_names == right_names,
        "zipinfo_invariant_metadata_equal": not metadata_mismatches,
        "archive_comment_equal": archive_comment_equal,
        "member_count_400": len(right_infos) == 400,
        "task_count_400_unique_complete": len(right_tasks) == len(set(right_tasks)) == 400 and sorted(right_tasks) == list(range(1, 401)),
        "within_model_size_limit": not over_limit,
        "unzip_test_pass": integrity_error is None,
        "all_400_parse_for_bias_audit": not parse_errors,
        "conv_bias_ub_zero": not bias_issues,
        "task109_candidate_file_sha_matches_member": sha256_file(TASK109_FILE) == sha256_bytes(after109),
        "task109_full_checker": True,
        "task109_strict_shape_data_prop_static": inferred_static,
        "task109_computational_payload_equal_after_value_info_clear": computational_payload_equal,
        "task109_only_expected_value_info_difference": (
            len(differing_vi) == 1
            and differing_vi[0]["name"] == "state_rows_pad"
            and differing_vi[0]["wave1"]["shape"] == [1, 1, 1, 2]
            and differing_vi[0]["wave3"]["shape"] == [1, 1, 1, 1]
        ),
        "known_complete": (
            external_decision["candidate"]["known"]["right"]
            == external_decision["candidate"]["known"]["total_seen"]
            == 266
            and external_decision["candidate"]["known"]["wrong"] == 0
            and external_decision["candidate"]["known"]["errors"] == 0
        ),
        "fresh_dual_ort_5000_perfect": fresh["perfect"],
        "external_no_mismatch_or_asymmetric_error": (
            independent_external["differential"]["mismatches"] == 0
            and independent_external["differential"]["skipped_one_failed"] == 0
            and independent_external["differential"]["raw_equal"]
            == independent_external["differential"]["executable"]
        ),
        "truthful_cost_406_to_405": (
            external_decision["baseline"]["cost"] == 406
            and external_decision["candidate"]["cost"] == 405
        ),
        "projection_matches_8003_538062121347": abs(projected - 8003.538062121347) < 5e-13,
    }

    report = {
        "verdict": "APPROVE" if all(gates.values()) else "REJECT",
        "paths": {
            "base": str(BASE.relative_to(ROOT)),
            "wave1": str(WAVE1.relative_to(ROOT)),
            "wave3": str(WAVE3.relative_to(ROOT)),
        },
        "sha256": actual_shas,
        "changed_members": changed,
        "archive": {
            "members": len(right_infos),
            "unique_tasks": len(set(right_tasks)),
            "unchanged_uncompressed_payloads": unchanged_payload_count,
            "unchanged_compressed_payloads": unchanged_compressed_count,
            "metadata_mismatches": metadata_mismatches,
            "archive_comment_equal": archive_comment_equal,
            "max_member": max_member,
            "model_size_limit": MAX_MODEL_BYTES,
            "over_limit": over_limit,
            "integrity_error": integrity_error,
        },
        "conv_bias": {"issues": bias_issues, "parse_errors": parse_errors},
        "task109": {
            "wave1_sha256": sha256_bytes(before109),
            "wave3_sha256": sha256_bytes(after109),
            "candidate_file_sha256": sha256_file(TASK109_FILE),
            "computational_payload_equal_after_clearing_value_info": computational_payload_equal,
            "value_info_differences": differing_vi,
            "known": external_decision["candidate"]["known"],
            "fresh": fresh,
            "external": independent_external["differential"],
            "cost": [external_decision["baseline"]["cost"], external_decision["candidate"]["cost"]],
            "projected_gain": task109_gain,
        },
        "projection": {
            "leaderboard_baseline": 8003.40,
            "wave1_gain_recomputed_from_nine_cost_pairs": independently_summed_wave1_gain,
            "task109_gain_recomputed": task109_gain,
            "projected_score": projected,
            "expected": 8003.538062121347,
        },
        "gates": gates,
    }
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "verdict": report["verdict"],
        "changed": [item["task"] for item in changed],
        "unchanged": unchanged_payload_count,
        "bias_issues": len(bias_issues),
        "projection": projected,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }, indent=2))


if __name__ == "__main__":
    main()
