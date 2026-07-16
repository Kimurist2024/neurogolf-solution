#!/usr/bin/env python3
"""Full-support refutation audit for the task396 71407 probes.

The two counterexamples below are expressed only through public generator
parameters.  This makes them reproducible without relying on saved private
examples or on a random-number implementation detail.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(ROOT / "inputs/arc-gen-repo/tasks"),
]
from lib import scoring  # noqa: E402


TASK = 396
ROOT_GUARDS = {
    ROOT / "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    ROOT / "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
    # This manifest was externally updated while this isolated audit was
    # running (6f22cc20... -> b57f95fd...).  The two guarded task396 payloads
    # below did not change, and this lane never writes the manifest.
    ROOT / "others/71407/MANIFEST.json": "b57f95fd3f17e163aaa5e894bf42465e6e44504d975e9832930b355d5b5ce0d2",
    ROOT / "others/71407/PROBE_ONLY_DO_NOT_MERGE/task396_cost1017.onnx.quarantine":
        "83f7ef034709949a7d743fd8909944c11fd7b65b2cf097874b8a2957abe1d6bf",
    ROOT / "others/71407/PROBE_ONLY_DO_NOT_MERGE/task396_cost961.onnx.quarantine":
        "1806e29dbd9f6cf7e21b2bb7dcf49f02ea1f613d2f12cfbfdad04f256ab99073",
}
MODEL_PATHS = {
    "authority1019": HERE / "baseline/task396_authority.onnx",
    "candidate1017": HERE / "candidates/task396_cost1017.onnx",
    "candidate961": HERE / "candidates/task396_cost961.onnx",
}
MODEL_SHA = {
    "authority1019": "ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94",
    "candidate1017": "83f7ef034709949a7d743fd8909944c11fd7b65b2cf097874b8a2957abe1d6bf",
    "candidate961": "1806e29dbd9f6cf7e21b2bb7dcf49f02ea1f613d2f12cfbfdad04f256ab99073",
}


CASES: dict[str, dict[str, Any]] = {
    # Reproduces random.seed(92000396), valid generated case 23.  This case
    # drives s8/maxw to uint8 zero.  candidate1017 computes 0-1 in uint8,
    # producing 255; authority clamps through Max(..., 1).
    "cost1017_uint8_underflow": {
        "random_origin": {"seed": 92000396, "valid_case": 23},
        "generator_parameters": {
            "width": 12,
            "height": 13,
            "rows": [1, 3, 4, 4, 4, 5, 5, 6, 7, 7, 8, 10, 11, 11],
            "cols": [9, 5, 0, 4, 10, 5, 10, 5, 7, 8, 2, 4, 2, 6],
            "brows": [2, 3],
            "bcols": [3, 9],
            "wides": [4, 3],
            "talls": [6, 4],
            "colors": [8, 6],
        },
        "mechanism": "maxw=0; uint8 Sub(maxw,1)=255 removes the authority clamp",
    },
    # Reproduces random.seed(94000396), valid generated case 93.  The cost961
    # graph keeps only three top rows.  The fourth row is the sole row carrying
    # the correct start/width code, so it is invisible to that candidate.
    "cost961_krow3_omits_decisive_row": {
        "random_origin": {"seed": 94000396, "valid_case": 93},
        "generator_parameters": {
            "width": 14,
            "height": 12,
            "rows": [0, 1, 5, 6, 6, 6, 7, 7, 9],
            "cols": [11, 5, 4, 3, 8, 11, 3, 4, 1],
            "brows": [4, 5],
            "bcols": [2, 10],
            "wides": [4, 3],
            "talls": [5, 3],
            "colors": [2, 5],
        },
        "mechanism": (
            "krow=3 omits the decisive fourth TopK row; all retained s8 codes "
            "are zero, powshift becomes zero, and the interior mask saturates"
        ),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_guards() -> None:
    for path, expected in {**ROOT_GUARDS, **{MODEL_PATHS[k]: v for k, v in MODEL_SHA.items()}}.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"guard failed for {path}: {actual} != {expected}")


def make_authority_equivalent_repair() -> tuple[Path, str]:
    """Repair candidate1017's underflow with one byte, not a true-rule repair."""
    model = onnx.load(MODEL_PATHS["candidate1017"])
    sub_index = next(
        i for i, node in enumerate(model.graph.node)
        if node.op_type == "Sub" and list(node.input) == ["maxw", "u1"]
    )
    clamp = helper.make_node("Max", ["maxw", "u1"], ["maxw_safe"], name="maxw_safe")
    nodes = list(model.graph.node)
    nodes.insert(sub_index, clamp)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.node[sub_index + 1].input[0] = "maxw_safe"
    model.graph.value_info.append(
        helper.make_tensor_value_info("maxw_safe", TensorProto.UINT8, [1])
    )
    path = HERE / "candidates/task396_cost1018_authority_equivalent_rejected.onnx"
    onnx.save(model, path)
    return path, sha256(path)


def tensor_dims(value: onnx.ValueInfoProto) -> list[int | None]:
    tensor = value.type.tensor_type
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in tensor.shape.dim]


def structural_audit(model: onnx.ModelProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        result["checker_full"] = False
        result["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        result["shape_inference_strict_data_prop"] = True
        values = {
            value.name: value
            for value in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        }
        init_names = {item.name for item in inferred.graph.initializer}
        nonstatic = []
        for node in inferred.graph.node:
            for name in node.output:
                if not name or name in init_names or name == "output":
                    continue
                value = values.get(name)
                dims = [] if value is None else tensor_dims(value)
                if not dims or any(dim is None or dim <= 0 for dim in dims):
                    nonstatic.append(name)
        result["nonstatic_outputs"] = sorted(set(nonstatic))
    except Exception as exc:  # noqa: BLE001
        result["shape_inference_strict_data_prop"] = False
        result["shape_error"] = f"{type(exc).__name__}: {exc}"
        result["nonstatic_outputs"] = None
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
    result.update(
        {
            "standard_domains": all(node.domain in {"", "ai.onnx"} for node in model.graph.node)
            and all(item.domain in {"", "ai.onnx"} for item in model.opset_import),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "nested_graph_attributes": sum(
                attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
                for node in model.graph.node for attr in node.attribute
            ),
            "nonfinite_initializer_values": int(
                sum(np.count_nonzero(~np.isfinite(arr)) for arr in arrays if arr.dtype.kind in "fc")
            ),
            "conv_bias_ub": any(node.op_type == "Conv" and len(node.input) >= 3 for node in model.graph.node),
            "node_count": len(model.graph.node),
            "op_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "duplicate_node_names": sorted(
                name for name, count in Counter(node.name for node in model.graph.node if node.name).items()
                if count > 1
            ),
        }
    )
    return result


def profile(path: Path, label: str) -> dict[str, Any]:
    model = onnx.load(path)
    scored = scoring.score_and_verify(
        model, TASK, str(HERE / "audit/profile_tmp"), label=label, require_correct=False
    )
    if scored is None:
        raise RuntimeError(f"official profile failed: {label}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "official_profile": scored,
        "structural": structural_audit(model),
    }


def make_session(path: Path, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def run_case(
    session: ort.InferenceSession,
    benchmark: dict[str, np.ndarray],
    authority_raw: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    try:
        raw = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: benchmark["input"]},
        )[0]
    except Exception as exc:  # noqa: BLE001
        return {"runtime_error": f"{type(exc).__name__}: {exc}"}, None
    want = benchmark["output"] > 0
    mask = raw > 0
    nonfinite = int(np.count_nonzero(~np.isfinite(raw)))
    positive = np.argwhere(mask)
    spatial = positive[:, 2:] if positive.size else np.empty((0, 2), dtype=np.int64)
    return {
        "runtime_error": None,
        "raw_shape": list(raw.shape),
        "nonfinite_values": nonfinite,
        "correct": bool(np.array_equal(mask, want)),
        "different_mask_cells": int(np.count_nonzero(mask != want)),
        "positive_extent_hw": (
            [int(spatial[:, 0].max()) + 1, int(spatial[:, 1].max()) + 1]
            if spatial.size else [0, 0]
        ),
        "raw_bit_identical_to_authority": (
            None if authority_raw is None else bool(np.array_equal(raw, authority_raw))
        ),
    }, raw


def graph_delta() -> dict[str, Any]:
    authority = onnx.load(MODEL_PATHS["authority1019"])
    c1017 = onnx.load(MODEL_PATHS["candidate1017"])
    c961 = onnx.load(MODEL_PATHS["candidate961"])
    return {
        "candidate1017": {
            "removed_from_authority": [
                "Gather(s8,bestj1)->wu", "Max(wu,u1)->wu_safe"
            ],
            "replacement": "Sub(maxw,u1)->widx",
            "counterexample_intermediates": {
                "maxw_uint8": 0,
                "authority_wu_safe_uint8": 1,
                "candidate_widx_uint8": 255,
            },
        },
        "candidate961": {
            "authority_nodes": len(authority.graph.node),
            "candidate_nodes": len(c961.graph.node),
            "topk_krow": int(numpy_helper.to_array(next(
                item for item in c961.graph.initializer if item.name == "krow"
            )).reshape(-1)[0]),
            "added_false_correction": (
                "if r0>=7 and width_code<=2 and rm==1, use r0-7"
            ),
            "counterexample_intermediates": {
                "authority_ri": [5, 7, 6, 4],
                "authority_s8": [0, 0, 0, 1],
                "authority_selected_r0": 4,
                "candidate_ri": [5, 7, 6],
                "candidate_s8": [0, 0, 0],
                "candidate_selected_r0": 5,
                "candidate_powshift": 0.0,
                "shift_cond": False,
            },
        },
        "node_counts": {
            "authority1019": len(authority.graph.node),
            "candidate1017": len(c1017.graph.node),
            "candidate961": len(c961.graph.node),
        },
    }


def main() -> int:
    assert_guards()
    repair_path, repair_sha = make_authority_equivalent_repair()
    paths = {**MODEL_PATHS, "repair1018_rejected": repair_path}
    profiles = {label: profile(path, label) for label, path in paths.items()}

    generator = importlib.import_module("task_fcb5c309")
    generated_cases: dict[str, dict[str, Any]] = {}
    for case_name, case in CASES.items():
        example = generator.generate(**case["generator_parameters"])
        random.seed(case["random_origin"]["seed"])
        np.random.seed(case["random_origin"]["seed"] & 0xFFFFFFFF)
        random_example = None
        for _ in range(case["random_origin"]["valid_case"]):
            random_example = generator.generate()
        if random_example != example:
            raise RuntimeError(f"seeded provenance mismatch: {case_name}")
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"counterexample does not convert: {case_name}")
        generated_cases[case_name] = {
            **case,
            "generator_reachable": True,
            "random_origin_reproduced": True,
            "input": example["input"],
            "output": example["output"],
            "input_shape_hw": [len(example["input"]), len(example["input"][0])],
            "output_shape_hw": [len(example["output"]), len(example["output"][0])],
        }

    modes = {
        "disable_all_threads1": (True, 1),
        "disable_all_threads4": (True, 4),
        "default_threads1": (False, 1),
        "default_threads4": (False, 4),
    }
    evidence: dict[str, Any] = {}
    totals = {label: {"runs": 0, "errors": 0, "nonfinite": 0, "wrong": 0} for label in paths}
    for mode_name, (disable_all, threads) in modes.items():
        sessions = {label: make_session(path, disable_all, threads) for label, path in paths.items()}
        mode_result: dict[str, Any] = {}
        for case_name, case in generated_cases.items():
            benchmark = scoring.convert_to_numpy({"input": case["input"], "output": case["output"]})
            assert benchmark is not None
            authority_stats, authority_raw = run_case(sessions["authority1019"], benchmark, None)
            per_model = {"authority1019": authority_stats}
            for label in ("candidate1017", "candidate961", "repair1018_rejected"):
                stats, _ = run_case(sessions[label], benchmark, authority_raw)
                per_model[label] = stats
            mode_result[case_name] = per_model
            for label, stats in per_model.items():
                totals[label]["runs"] += 1
                totals[label]["errors"] += int(stats["runtime_error"] is not None)
                totals[label]["nonfinite"] += int(stats.get("nonfinite_values", 0))
                totals[label]["wrong"] += int(not stats.get("correct", False))
        evidence[mode_name] = mode_result

    # The one-byte repair must be semantically identical to authority on all
    # known examples and on the two explicit support witnesses.  This does not
    # make it true-rule sound because the authority itself fails the first case.
    known_identity = {name: {"compared": 0, "different": 0, "errors": 0} for name in modes}
    known_four = {
        label: {
            name: {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0}
            for name in modes
        }
        for label in paths
    }
    examples = scoring.load_examples(TASK)
    known = examples["train"] + examples["test"] + examples["arc-gen"]
    for mode_name, (disable_all, threads) in modes.items():
        sessions = {
            label: make_session(path, disable_all, threads)
            for label, path in paths.items()
        }
        for example in known:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            raws: dict[str, np.ndarray] = {}
            for label, session in sessions.items():
                stats = known_four[label][mode_name]
                try:
                    raw = session.run(None, {"input": benchmark["input"]})[0]
                except Exception:  # noqa: BLE001
                    stats["errors"] += 1
                    continue
                raws[label] = raw
                stats["nonfinite"] += int(np.count_nonzero(~np.isfinite(raw)))
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    stats["right"] += 1
                else:
                    stats["wrong"] += 1
            if "authority1019" not in raws or "repair1018_rejected" not in raws:
                known_identity[mode_name]["errors"] += 1
            else:
                known_identity[mode_name]["compared"] += 1
                known_identity[mode_name]["different"] += int(
                    not np.array_equal(raws["authority1019"], raws["repair1018_rejected"])
                )

    result = {
        "task": TASK,
        "task_hash": "fcb5c309",
        "decision": "REJECT_BOTH_71407_CANDIDATES",
        "root_or_stage_modified": False,
        "observed_external_manifest_drift": {
            "path": "others/71407/MANIFEST.json",
            "before_sha256": "6f22cc2024b779f37b51386b64086559621a5572575902085f833cc71f8fff28",
            "after_sha256": "b57f95fd3f17e163aaa5e894bf42465e6e44504d975e9832930b355d5b5ce0d2",
            "task396_payloads_changed": False,
        },
        "guard_sha256": {str(path.relative_to(ROOT)): expected for path, expected in ROOT_GUARDS.items()},
        "profiles": profiles,
        "graph_delta": graph_delta(),
        "known_four_mode": known_four,
        "counterexamples": generated_cases,
        "four_mode_evidence": evidence,
        "totals_over_two_counterexamples_x_four_modes": totals,
        "authority_equivalent_repair": {
            "path": str(repair_path.relative_to(ROOT)),
            "sha256": repair_sha,
            "cost": profiles["repair1018_rejected"]["official_profile"]["cost"],
            "known_raw_identity_to_authority": known_identity,
            "disposition": "rejected: cheaper authority-equivalent repair is still generator-unsound",
        },
        "known_sound_control": {
            "path": "scripts/golf/scratch_codex/task396/agent_corner_micro.onnx",
            "sha256": "f1bddd36f0c0b943fe84d500bb629159b3639997bf7ea4b2e39eb2aa2bc9da2b",
            "cost": 1245,
            "fresh_history": "5000/5000 in disable-all and default ORT",
            "delta_vs_authority": 226,
        },
        "cheaper_true_rule_repair_found": False,
    }
    out = HERE / "audit/result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for case_name, case in generated_cases.items():
        (HERE / "counterexamples").mkdir(parents=True, exist_ok=True)
        (HERE / "counterexamples" / f"{case_name}.json").write_text(
            json.dumps(case, indent=2) + "\n"
        )
    assert_guards()
    print(json.dumps({
        "decision": result["decision"],
        "costs": {label: row["official_profile"]["cost"] for label, row in profiles.items()},
        "totals": totals,
        "repair_sha256": repair_sha,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
