from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "scripts" / "golf" / "loop_7999_13" / "lane_b13"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


MODELS = {
    "task254_exact_base": (254, ROOT / "scripts/golf/loop_7999_13/lane_b12/baseline/task254.onnx"),
    "task254_archive_cost68": (
        254,
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task254_r03_static68.onnx",
    ),
    "task254_precontract15_reject": (
        254,
        LANE / "candidates/task254_precontract15.onnx",
    ),
    "task267_exact_base": (267, ROOT / "scripts/golf/loop_7999_13/lane_b12/baseline/task267.onnx"),
    "task267_safe_floor_control": (267, LANE / "controls/task267_safe_cost60.onnx"),
    "task267_archive_cost30_reject": (
        267,
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task267_r01_static30.onnx",
    ),
}
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Compress", "TfIdfVectorizer"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    if not value.type.HasField("tensor_type"):
        return ["non_tensor"]
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def verify_known(task: int, model: onnx.ModelProto, disable_all: bool) -> dict[str, object]:
    runner = session(model, disable_all)
    right = wrong = runtime_errors = nonfinite = 0
    min_positive = float("inf")
    max_nonpositive = -float("inf")
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = runner.run(
                    [runner.get_outputs()[0].name],
                    {runner.get_inputs()[0].name: benchmark["input"]},
                )[0]
                matches = np.array_equal(raw > 0, benchmark["output"] > 0)
                right += int(matches)
                wrong += int(not matches)
                finite = np.isfinite(raw)
                nonfinite += int(np.count_nonzero(~finite))
                positive = raw[(raw > 0) & finite]
                nonpositive = raw[(raw <= 0) & finite]
                if positive.size:
                    min_positive = min(min_positive, float(positive.min()))
                if nonpositive.size:
                    max_nonpositive = max(max_nonpositive, float(nonpositive.max()))
            except Exception:  # noqa: BLE001
                runtime_errors += 1
    return {
        "right": right,
        "wrong": wrong,
        "runtime_errors": runtime_errors,
        "nonfinite_elements": nonfinite,
        "min_positive": min_positive if np.isfinite(min_positive) else None,
        "max_nonpositive": max_nonpositive if np.isfinite(max_nonpositive) else None,
    }


def inspect(label: str, task: int, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        strict = False
        strict_error = f"{type(exc).__name__}: {exc}"
        inferred = model
    with tempfile.TemporaryDirectory(prefix=f"b13_{label}_", dir="/tmp") as workdir:
        actual = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label, require_correct=False
        )
    initializers = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        initializers.append(
            {
                "name": item.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "elements": int(array.size),
                "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True,
            }
        )
    equations = []
    max_einsum = 0
    for node in model.graph.node:
        if node.op_type == "Einsum":
            max_einsum = max(max_einsum, len(node.input))
            for attr in node.attribute:
                if attr.name == "equation":
                    value = helper.get_attribute_value(attr)
                    equations.append(value.decode() if isinstance(value, bytes) else str(value))
    row: dict[str, object] = {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "node_count": len(model.graph.node),
        "ops": dict(Counter(node.op_type for node in model.graph.node)),
        "max_einsum_operands": max_einsum,
        "equations": equations,
        "initializers": initializers,
        "full_checker": checker,
        "checker_error": checker_error,
        "strict_shape_data_prop": strict,
        "strict_error": strict_error,
        "declared_output_shapes": [dims(item) for item in model.graph.output],
        "inferred_output_shapes": [dims(item) for item in inferred.graph.output],
        "all_static": all(
            all(isinstance(dimension, int) and dimension > 0 for dimension in dims(item))
            for item in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        ),
        "standard_domains": all(
            node.domain in ("", "ai.onnx") for node in model.graph.node
        )
        and all(item.domain in ("", "ai.onnx") for item in model.opset_import),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graph_attributes": sum(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attr in node.attribute
        ),
        "banned_ops": [node.op_type for node in model.graph.node if node.op_type in BANNED],
        "conv_bias_issues": check_conv_bias(model),
        "actual_score": actual,
        "known_disable_all": verify_known(task, model, True),
        "known_default": verify_known(task, model, False),
    }
    row["structure_safe_under_b13"] = bool(
        checker
        and strict
        and row["all_static"]
        and row["standard_domains"]
        and not row["functions"]
        and not row["sparse_initializers"]
        and not row["nested_graph_attributes"]
        and not row["banned_ops"]
        and not row["conv_bias_issues"]
        and max_einsum <= 16
        and all(item["finite"] for item in initializers)
    )
    return row


def exact_tt_floor() -> dict[str, object]:
    # Reconstruct the 7-axis coefficient tensor represented by the correct
    # cost-68 / 20-operand witness, then exhaust all feature-axis orders and all
    # contiguous four-core cuts.  Exact matrix ranks give the minimal dense TT
    # core element count in this family.
    source = onnx.load(MODELS["task254_archive_cost68"][1])
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in source.graph.initializer
    }
    v = arrays["V"]
    u = np.einsum("PZ,Pb->b", v, v)
    f = np.einsum("efXY,Xb,Yb->efb", arrays["W"], v, v)
    t = np.einsum("efj,b,efb->efjb", arrays["P"], u, f)
    k = np.einsum("ABe,CDEf,efjb->ABCDEjb", arrays["CA"], arrays["CB"], t)
    dimensions = [2] * 6 + [10]
    best: tuple[int, str, tuple[int, int, int], tuple[int, ...]] | None = None
    for permutation in itertools.permutations(range(6)):
        order = [*permutation, 6]
        arranged = np.transpose(k, order)
        ordered_dimensions = [dimensions[index] for index in order]
        for cuts in itertools.combinations(range(1, 7), 3):
            groups = []
            start = 0
            for end in (*cuts, 7):
                groups.append(int(np.prod(ordered_dimensions[start:end])))
                start = end
            ranks = [1]
            for cut in cuts:
                matrix = arranged.reshape(
                    int(np.prod(ordered_dimensions[:cut])),
                    int(np.prod(ordered_dimensions[cut:])),
                )
                ranks.append(int(np.linalg.matrix_rank(matrix, tol=1e-7)))
            ranks.append(1)
            core_elements = sum(
                ranks[index] * groups[index] * ranks[index + 1]
                for index in range(4)
            )
            names = "ABCDEj"
            candidate = (
                core_elements,
                "".join(names[index] for index in permutation) + "b",
                cuts,
                tuple(ranks),
            )
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {
        "family": "four dense TT cores + shared V, yielding exactly 16 Einsum operands",
        "minimum_core_elements_for_exact_coefficient_tensor": best[0],
        "shared_v_elements": 20,
        "minimum_total_params": best[0] + 20,
        "best_axis_order": best[1],
        "cuts": list(best[2]),
        "ranks_with_endpoints": list(best[3]),
        "incumbent_cost": 76,
        "strict_improvement_possible_in_exact_family": best[0] + 20 < 76,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    rows = {label: inspect(label, task, path) for label, (task, path) in MODELS.items()}
    tt = json.loads((LANE / "tt_search.json").read_text())
    max_tt_right = max(
        max(row["disable_all"]["right"], row["default"]["right"])
        for row in tt["rows"]
    )
    audit = {
        "immutable_base": {
            "path": "submission_base_7999.13.zip",
            "sha256": sha256(ROOT / "submission_base_7999.13.zip"),
        },
        "models": rows,
        "task254_tt_search": {
            "attempt_count": tt["attempt_count"],
            "known_winner_count": len(tt["known_winners"]),
            "max_known_right_in_either_mode": max_tt_right,
            "all_runtime_errors": sum(
                row[mode]["runtime_errors"]
                for row in tt["rows"]
                for mode in ("disable_all", "default")
            ),
            "parameter_range": [
                min(row["total_params"] for row in tt["rows"]),
                max(row["total_params"] for row in tt["rows"]),
            ],
        },
        "task254_exact_tt_floor": exact_tt_floor(),
        "decision": {
            "winners": [],
            "fresh5000_run": False,
            "fresh5000_skip_reason": "No candidate passed complete known gold in both ORT modes.",
            "cost_delta": 0,
            "score_delta": 0.0,
        },
    }
    (LANE / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    winner_manifest = {
        "immutable_base": audit["immutable_base"],
        "winners": [],
        "decision": "no adoption",
        "cost_delta": 0,
        "score_delta": 0.0,
    }
    (LANE / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2) + "\n")

    report = f"""# Lane B13 — task254/task267 safe rebuild

## Result

No candidate is eligible for adoption. The exact `submission_base_7999.13.zip`
remains unchanged; lane score delta is `0.0`.

| task | exact base | best safe result | decision |
|---:|---:|---:|---|
| 254 | 76 | no correct model below 76 with at most 16 Einsum operands | reject all |
| 267 | 60 | safe zero-rebuild control cost 60 (tie) | no adoption |

## Task254

The generator rule is fixed 9x9 gray bars: the shortest bar becomes red and the
tallest becomes blue. The exact base is cost 76 but contains Einsums with up to
49 operands, so it is not a B13-safe construction.

Three independent routes were checked:

1. Constant precontraction produced a cost-64, 15-operand model, but eliminating
   the coupled `e/f` latent state changed the function. It scored 0/265 known in
   both ORT modes and was immediately rejected.
2. A four-core tensor-train keeps the single graph node at exactly 16 operands.
   {tt['attempt_count']} candidates spanning cost/params
   {min(row['total_params'] for row in tt['rows'])}–{max(row['total_params'] for row in tt['rows'])}
   were tested. None solved any complete known case; runtime errors were zero.
   Approximation perturbed required zero logits across the threshold.
3. Exhaustive exact-rank analysis over all six feature-axis orders and all
   four-core cuts found an exact-family floor of
   {audit['task254_exact_tt_floor']['minimum_total_params']} params, above cost 76.

The correct archive cost-68 witness still needs 20 Einsum operands and is
therefore rejected by the explicit >16 rule.

## Task267

The generator rule is fixed 7x7 creature recoloring from the marker at `(6,0)`.
The from-scratch standard control uses one 5-input Einsum, finite initializers,
no intermediates, and cost 60. It passed all 264 known cases in both ORT modes
with zero runtime errors, but ties the exact base and cannot improve the score.

The archive cost-30 model remains ineligible: it uses a 37-input giant Einsum
and numerical repeated-product behavior. No lookup, UB, shape cloak, sparse
initializer, custom domain, or nonstandard operator was adopted.

## Gates and files

- `audit.json`: checker, strict shape inference with data propagation, domains,
  finite initializers, Conv bias, operand counts, actual costs, and both-ORT known results.
- `tt_search.json`: all 60 under-budget TT attempts and their complete-known results.
- `winner_manifest.json`: empty winner list.

Fresh 5000 x both ORT modes was intentionally not run because no candidate
passed the required complete-known pre-gate. Root ZIP/CSV/ledger files were not modified.
"""
    (LANE / "REPORT.md").write_text(report)
    print(json.dumps(winner_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
