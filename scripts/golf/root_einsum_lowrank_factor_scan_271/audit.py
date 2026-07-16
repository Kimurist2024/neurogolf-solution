#!/usr/bin/env python3
"""Independent fail-closed audit for exact Einsum low-rank scan 271."""

from __future__ import annotations

import hashlib
import json
import string
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import sympy as sp
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
STAGE = ROOT / "others/71407"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(value: Any) -> sp.Rational:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (int, np.integer, bool, np.bool_)):
        return sp.Rational(int(scalar))
    numerator, denominator = float(scalar).as_integer_ratio()
    return sp.Rational(numerator, denominator)


def matrix_for(array: np.ndarray, left: tuple[int, ...]) -> sp.Matrix:
    right = tuple(axis for axis in range(array.ndim) if axis not in left)
    rows = int(np.prod([array.shape[axis] for axis in left], dtype=np.int64))
    columns = int(np.prod([array.shape[axis] for axis in right], dtype=np.int64))
    flattened = np.transpose(array, left + right).reshape(rows, columns)
    return sp.Matrix(rows, columns, [rational(value) for value in flattened.reshape(-1)])


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            value = onnx.helper.get_attribute_value(attr)
            return value.decode("ascii") if isinstance(value, bytes) else str(value)
    return None


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: np.asarray(numpy_helper.to_array(initializer))
        for initializer in model.graph.initializer
    }


def load_sources() -> tuple[
    dict[int, onnx.ModelProto], dict[int, onnx.ModelProto], dict[str, Any]
]:
    authority = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            authority[task] = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
    manifest = json.loads((STAGE / "MANIFEST.json").read_text())
    stage = {}
    snapshot = []
    for row in manifest["active_candidates"]:
        path = STAGE / row["file"]
        digest = sha256(path)
        if digest != row["sha256"]:
            raise AssertionError(f"manifest SHA mismatch: {path}")
        task = int(row["task"])
        stage[task] = onnx.load(path)
        snapshot.append({"task": task, "path": str(path.relative_to(ROOT)), "sha256": digest})
    return authority, stage, {"manifest": manifest, "snapshot": snapshot}


def expected_partition_keys(models: dict[int, onnx.ModelProto]) -> set[tuple[int, str, tuple[int, ...]]]:
    keys = set()
    for task, model in models.items():
        initializer_arrays = arrays(model)
        einsum_names = {
            name
            for node in model.graph.node
            if node.op_type == "Einsum"
            for name in node.input
            if name in initializer_arrays
        }
        for name in einsum_names:
            array = initializer_arrays[name]
            if (
                2 <= array.ndim <= 8
                and array.dtype.kind in "fiu"
                and np.all(np.isfinite(array))
            ):
                for mask in range(1, 1 << array.ndim):
                    if mask & 1 and mask != (1 << array.ndim) - 1:
                        left = tuple(axis for axis in range(array.ndim) if mask & (1 << axis))
                        keys.add((task, name, left))
    return keys


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def invalid_latent_probe(label: str) -> dict[str, Any]:
    eq = f"a{label},{label}b->ab"
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Einsum", ["U", "V"], ["Y"], equation=eq)],
            "invalid_latent_probe",
            [],
            [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 2])],
            [
                numpy_helper.from_array(np.eye(2, dtype=np.float32), "U"),
                numpy_helper.from_array(np.eye(2, dtype=np.float32), "V"),
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    checker_rejected = False
    runtime_rejected = False
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        checker_rejected = True
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
    except Exception:
        runtime_rejected = True
    return {
        "label": label,
        "equation": eq,
        "checker_rejected": checker_rejected,
        "ort_disable_all_rejected": runtime_rejected,
    }


def main() -> None:
    scan_path = HERE / "scan.json"
    candidates_path = HERE / "candidates.json"
    scan = json.loads(scan_path.read_text())
    candidates = json.loads(candidates_path.read_text())
    authority, stage, stage_info = load_sources()
    checks: list[str] = []

    require(scan["authority"]["sha256"] == sha256(AUTHORITY), "authority SHA matches", checks)
    require(len(authority) == scan["authority"]["models"] == 400, "authority covers 400 models", checks)
    require(
        scan["active_stage"]["manifest_sha256"] == sha256(STAGE / "MANIFEST.json"),
        "active-stage manifest SHA matches",
        checks,
    )
    require(
        scan["active_stage"]["snapshot"] == stage_info["snapshot"],
        "active-stage byte snapshot matches manifest descendants",
        checks,
    )
    require(
        len(stage) == scan["active_stage"]["models"] == stage_info["manifest"]["active_root_onnx_count"],
        "active-stage model count matches manifest",
        checks,
    )
    invalid_label_probes = [invalid_latent_probe(label) for label in ("0", "_", "@", "α")]
    require(
        all(row["checker_rejected"] and row["ort_disable_all_rejected"] for row in invalid_label_probes),
        "digits, punctuation, and Unicode cannot extend the Einsum latent-label alphabet",
        checks,
    )

    source_models = {"authority": authority, "active_stage": stage}
    source_arrays = {
        source: {task: arrays(model) for task, model in models.items()}
        for source, models in source_models.items()
    }
    for source, models in source_models.items():
        rows = scan["partition_rows"][source]
        actual_keys = {
            (int(row["task"]), row["initializer"], tuple(row["left_axes"])) for row in rows
        }
        expected_keys = expected_partition_keys(models)
        require(actual_keys == expected_keys, f"{source}: every axis bipartition appears exactly once", checks)
        require(len(rows) == len(actual_keys), f"{source}: partition rows contain no duplicates", checks)
        for row in rows:
            array = source_arrays[source][int(row["task"])][row["initializer"]]
            exact_rank = int(matrix_for(array, tuple(row["left_axes"])).rank())
            if exact_rank != row["exact_rational_rank"]:
                raise AssertionError(
                    f"{source} task{row['task']} {row['initializer']} {row['left_axes']}: rank mismatch"
                )
        checks.append(f"{source}: all recorded exact rational ranks independently recompute")

    composite = dict(authority)
    composite.update(stage)
    require(len(composite) == scan["composite_best"]["models"] == 400, "composite covers 400 models", checks)
    rge2 = scan["composite_rge2_parameter_saving"]
    require(len(rge2) == 6, "composite has six exact R>=2 parameter-saving bipartitions", checks)
    require({int(row["task"]) for row in rge2} == {13, 107, 398}, "R>=2 rows occur only in tasks 013/107/398", checks)

    for row in rge2:
        task = int(row["task"])
        model = composite[task]
        array = arrays(model)[row["initializer"]]
        matrix = matrix_for(array, tuple(row["left_axes"]))
        factor = row["factorization"]
        left = sp.Matrix([[sp.Rational(value) for value in values] for values in factor["left_rationals"]])
        right = sp.Matrix([[sp.Rational(value) for value in values] for values in factor["right_rationals"]])
        require(left * right == matrix, f"task{task:03d} {row['initializer']}: factor coefficients reconstruct exactly", checks)
        require(
            factor["factor_dtype_fully_representable"]
            and factor["serialized_coefficient_reconstruction_exact"],
            f"task{task:03d} {row['initializer']}: factors are exactly serialized in source dtype",
            checks,
        )

        uses_per_node: dict[int, int] = defaultdict(int)
        for node_index, node in enumerate(model.graph.node):
            if node.op_type == "Einsum":
                uses_per_node[node_index] += sum(name == row["initializer"] for name in node.input)
        for budget in row["label_budget"]:
            node = model.graph.node[int(budget["node_index"])]
            eq = equation(node)
            used = set(eq.replace("->", "").replace(",", "").replace(" ", ""))
            available = len(set(string.ascii_letters) - used)
            required = uses_per_node[int(budget["node_index"])]
            require(
                available == budget["available_unused_labels"]
                and required == budget["occurrences_requiring_independent_latents"]
                and available < required,
                f"task{task:03d} {row['initializer']}: independent latent-label budget is insufficient",
                checks,
            )

    require(scan["structural_candidates"] == [], "structural candidate list is empty", checks)
    require(scan["strict_lower_candidates"] == [], "strict-lower candidate list is empty", checks)
    require(scan["winner"] is None, "winner is null", checks)
    require(
        candidates["structural_candidates"] == []
        and candidates["strict_lower_candidates"] == []
        and candidates["winner"] is None,
        "standalone candidate ledger is empty",
        checks,
    )
    require(
        not any(
            scan["candidate_gate"][key]
            for key in (
                "full_checker_strict_data_prop",
                "actual_profile_truthful_shape",
                "known_four_raw",
                "fresh_two_by_2000",
                "runtime_errors_zero",
                "nonfinite_zero",
                "conv_bias_ub0",
            )
        ),
        "runtime gates are explicitly skipped because no graph can be formed",
        checks,
    )
    require(
        not any(scan["policy"].values()),
        "policy rejects approximation, rank1 adoption, partial sharing, cloaks, private-zero, and protected writes",
        checks,
    )

    payload = {
        "audit": "root_einsum_lowrank_factor_scan_271",
        "pass": True,
        "decision": "NO_BUILDABLE_RGE2_EINSUM_FACTOR",
        "scan_sha256": sha256(scan_path),
        "scanner_sha256": sha256(HERE / "scan.py"),
        "candidates_sha256": sha256(candidates_path),
        "authority_sha256": sha256(AUTHORITY),
        "active_stage_manifest_sha256": sha256(STAGE / "MANIFEST.json"),
        "invalid_latent_label_probes": invalid_label_probes,
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
