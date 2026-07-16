#!/usr/bin/env python3
"""Final structural and known-data audit for the task379 QV factor winner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "golf" / "loop_7999_13"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402
from dual_ort_fresh import make_session  # noqa: E402


TASK = 379
BASELINE = HERE / "baseline" / "task379.onnx"
CANDIDATE = HERE / "candidates" / "task379_qv_middle_rank2.onnx"
OUTPUT = HERE / "task379_qv_middle_rank2_final_audit.json"
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Compress", "SequenceMap"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def known_differential(
    baseline: onnx.ModelProto, candidate: onnx.ModelProto, disabled: bool
) -> dict[str, object]:
    baseline_session = make_session(baseline, disabled)
    candidate_session = make_session(candidate, disabled)
    row = {
        "total": 0,
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "max_abs_raw_difference": 0.0,
    }
    examples = scoring.load_examples(TASK)
    for subset in ("train", "test", "arc-gen"):
        for example in examples[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            row["total"] += 1
            try:
                baseline_raw = baseline_session.run(
                    ["output"], {"input": benchmark["input"]}
                )[0]
                candidate_raw = candidate_session.run(
                    ["output"], {"input": benchmark["input"]}
                )[0]
            except Exception:  # noqa: BLE001
                row["runtime_errors"] += 1
                continue
            correct = np.array_equal(candidate_raw > 0, benchmark["output"] > 0)
            row["right"] += int(correct)
            row["wrong"] += int(not correct)
            row["raw_equal"] += int(
                np.array_equal(baseline_raw, candidate_raw, equal_nan=True)
            )
            row["threshold_equal"] += int(
                np.array_equal(baseline_raw > 0, candidate_raw > 0)
            )
            difference = np.abs(
                np.nan_to_num(baseline_raw, nan=0.0, posinf=0.0, neginf=0.0)
                - np.nan_to_num(candidate_raw, nan=0.0, posinf=0.0, neginf=0.0)
            )
            row["max_abs_raw_difference"] = max(
                row["max_abs_raw_difference"], float(difference.max(initial=0.0))
            )
    row["perfect"] = bool(
        row["right"] == row["total"]
        and row["wrong"] == 0
        and row["runtime_errors"] == 0
        and row["raw_equal"] == row["total"]
        and row["threshold_equal"] == row["total"]
    )
    return row


def runtime_shapes(model: onnx.ModelProto, disabled: bool) -> dict[str, object]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runner = ort.InferenceSession(traced.SerializeToString(), options)
    benchmark = scoring.convert_to_numpy(scoring.load_examples(TASK)["train"][0])
    if benchmark is None:
        raise RuntimeError("known input was not convertible")
    values = runner.run(names, {"input": benchmark["input"]})
    mismatches = []
    for name, value in zip(names, values):
        static = dims(typed[name])
        actual = list(np.asarray(value).shape)
        if static != actual:
            mismatches.append({"name": name, "static": static, "runtime": actual})
    output_shape = list(np.asarray(values[names.index("output")]).shape)
    return {
        "mode": "disable_all" if disabled else "default",
        "traced_outputs": len(names),
        "mismatches": mismatches,
        "output_shape": output_shape,
        "truthful": not mismatches and output_shape == [1, 10, 30, 30],
    }


def main() -> int:
    baseline = onnx.load(BASELINE)
    candidate = onnx.load(CANDIDATE)
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(candidate), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    with tempfile.TemporaryDirectory(prefix="c42_task379_", dir=HERE) as workdir:
        baseline_score = scoring.score_and_verify(
            copy.deepcopy(baseline), TASK, workdir, "baseline", require_correct=True
        )
        candidate_score = scoring.score_and_verify(
            copy.deepcopy(candidate), TASK, workdir, "candidate", require_correct=True
        )

    known = {
        "disable_all": known_differential(baseline, candidate, True),
        "default": known_differential(baseline, candidate, False),
    }
    shape_rows = [runtime_shapes(candidate, True), runtime_shapes(candidate, False)]
    baseline_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item)) for item in baseline.graph.initializer
    }
    candidate_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer
    }
    qcore = candidate_arrays["QCore2"]
    expand = candidate_arrays["QExpand3x2"]
    mode = candidate_arrays["QMode3x2"]
    row1 = candidate_arrays["QRow1_2"]
    qv = baseline_arrays["QV"]
    nv = baseline_arrays["NV__mode1__from__QV"]
    rflip = baseline_arrays["Rflip__slice1_1__of__QV"]
    factor_checks = {
        "qv_reconstructed": bool(np.array_equal(np.einsum("xrg,mr->xmg", qcore, expand), qv)),
        "qv_nv_effect_reconstructed": bool(
            np.array_equal(
                np.einsum("xrg,ir->xig", qcore, mode),
                np.einsum("xmg,im->xig", qv, nv),
            )
        ),
        "qv_rflip_effect_reconstructed": bool(
            np.array_equal(
                np.einsum("orp,r->op", qcore, row1),
                np.einsum("onp,n->op", qv, rflip),
            )
        ),
        "old_parameter_count": int(qv.size + nv.size + rflip.size),
        "new_parameter_count": int(qcore.size + expand.size + mode.size + row1.size),
    }
    nested_graphs = [
        node.op_type
        for node in candidate.graph.node
        for attribute in node.attribute
        if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    structure = {
        "full_checker": True,
        "strict_shape_data_prop": True,
        "all_static_positive": all(
            all(isinstance(dim, int) and dim > 0 for dim in dims(value)) for value in values
        ),
        "runtime_shapes": shape_rows,
        "canonical_io": [value.name for value in candidate.graph.input] == ["input"]
        and [value.name for value in candidate.graph.output] == ["output"],
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in candidate.graph.node)
        and all(item.domain in ("", "ai.onnx") for item in candidate.opset_import),
        "functions": len(candidate.functions),
        "sparse_initializers": len(candidate.graph.sparse_initializer),
        "nested_graphs": nested_graphs,
        "banned_ops": [
            node.op_type
            for node in candidate.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        ],
        "conv_bias_issues": check_conv_bias(candidate),
        "node_count": len(candidate.graph.node),
        "initializer_count": len(candidate.graph.initializer),
        "op_histogram": dict(sorted(Counter(node.op_type for node in candidate.graph.node).items())),
    }
    report = {
        "task": TASK,
        "baseline": {
            "path": str(BASELINE.relative_to(ROOT)),
            "sha256": digest(BASELINE),
            "score": baseline_score,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(CANDIDATE),
            "serialized_size": CANDIDATE.stat().st_size,
            "score": candidate_score,
        },
        "cost_reduction": 2,
        "projected_gain": math.log(1951 / 1949),
        "known_dual_raw_equivalence": known,
        "structure": structure,
        "factorization": factor_checks,
        "initializer_changes": {
            "removed": sorted(set(baseline_arrays) - set(candidate_arrays)),
            "added": sorted(set(candidate_arrays) - set(baseline_arrays)),
        },
    }
    report["pass"] = bool(
        baseline_score
        and baseline_score["cost"] == 1951
        and candidate_score
        and candidate_score["cost"] == 1949
        and candidate_score["correct"]
        and all(row["perfect"] for row in known.values())
        and all(row["truthful"] for row in shape_rows)
        and structure["all_static_positive"]
        and structure["canonical_io"]
        and structure["input_shape"] == structure["output_shape"] == [1, 10, 30, 30]
        and structure["standard_domains"]
        and not structure["functions"]
        and not structure["sparse_initializers"]
        and not structure["nested_graphs"]
        and not structure["banned_ops"]
        and not structure["conv_bias_issues"]
        and all(factor_checks[key] for key in factor_checks if key.endswith("reconstructed"))
        and factor_checks["old_parameter_count"] == 24
        and factor_checks["new_parameter_count"] == 22
    )
    OUTPUT.write_text(json.dumps(report, indent=2, default=bool) + "\n")
    print(json.dumps(report, indent=2, default=bool))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
