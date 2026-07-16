#!/usr/bin/env python3
"""Enumerate exact shared-Concat basis rewrites and measure actual costs."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE / "base"
CANDIDATES = HERE / "candidates"
TASKS = (13, 55, 99, 281)

sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


ROOT_FILES = (
    REPO / "submission_base_8009.46.zip",
    REPO / "submission.zip",
    REPO / "all_scores.csv",
    REPO / "others" / "71407" / "task013.onnx",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO)): sha256(path) for path in ROOT_FILES}


def equation(node: onnx.NodeProto) -> tuple[list[str], str]:
    text = next(item.s.decode() for item in node.attribute if item.name == "equation")
    left, right = text.split("->", 1)
    return left.split(","), right


def set_equation(node: onnx.NodeProto, terms: list[str], output: str) -> None:
    next(item for item in node.attribute if item.name == "equation").s = (
        ",".join(terms) + "->" + output
    ).encode()


def array(model: onnx.ModelProto, name: str) -> np.ndarray:
    item = next(value for value in model.graph.initializer if value.name == name)
    return np.asarray(numpy_helper.to_array(item))


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    if any(item.name == name for item in model.graph.initializer):
        raise ValueError(name)
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def remove_initializer_if_unused(model: onnx.ModelProto, name: str) -> bool:
    if any(name in node.input for node in model.graph.node):
        return False
    kept = [item for item in model.graph.initializer if item.name != name]
    if len(kept) == len(model.graph.initializer):
        raise KeyError(name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return True


def stable_union(items: Iterable[Iterable[str]]) -> list[str]:
    result: list[str] = []
    for values in items:
        for value in values:
            if value not in result:
                result.append(value)
    return result


def expand_axis(value: np.ndarray, axis: int, basis: int, mapping: list[int]) -> np.ndarray:
    if len(mapping) != value.shape[axis] or len(set(mapping)) != len(mapping):
        raise ValueError((value.shape, axis, basis, mapping))
    shape = list(value.shape)
    shape[axis] = basis
    result = np.zeros(shape, dtype=value.dtype)
    for old, new in enumerate(mapping):
        source = [slice(None)] * value.ndim
        target = [slice(None)] * value.ndim
        source[axis] = old
        target[axis] = new
        result[tuple(target)] = value[tuple(source)]
    return result


def bridge_axis(value: np.ndarray, axis: int, basis: int, mapping: list[int]) -> np.ndarray:
    if len(mapping) != value.shape[axis]:
        raise ValueError((value.shape, axis, basis, mapping))
    selector = np.zeros((basis, value.shape[axis]), dtype=value.dtype)
    for old, new in enumerate(mapping):
        selector[new, old] = 1
    selector_shape = [basis] + [1] * value.ndim
    selector_shape[1 + axis] = value.shape[axis]
    return value[None, ...] * selector.reshape(selector_shape)


def free_letters(node: onnx.NodeProto) -> list[str]:
    terms, output = equation(node)
    used = {char for char in "".join(terms) + output if char.isalpha()}
    return [char for char in string.ascii_letters if char not in used]


def replace_concat_group(
    model: onnx.ModelProto,
    outputs: list[str],
    basis_inputs: list[str],
    basis_name: str,
    axis: int,
) -> onnx.NodeProto:
    final = next(node for node in model.graph.node if node.op_type == "Einsum" and node.output[0] == "output")
    indices = [
        index
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Concat" and node.output and node.output[0] in outputs
    ]
    if len(indices) != len(outputs):
        raise AssertionError((outputs, indices))
    kept = [node for index, node in enumerate(model.graph.node) if index not in set(indices)]
    final_index = next(index for index, node in enumerate(kept) if node is final)
    basis_node = onnx.helper.make_node("Concat", basis_inputs, [basis_name], axis=axis, name=basis_name)
    kept.insert(final_index, basis_node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    final = next(node for node in model.graph.node if node.op_type == "Einsum" and node.output[0] == "output")
    for index, name in enumerate(final.input):
        if name in outputs:
            final.input[index] = basis_name
    return final


def coeff_clone(
    model: onnx.ModelProto,
    final: onnx.NodeProto,
    positions: list[int],
    name: str,
    axis: int,
    mapping: list[int],
    basis: int,
    mode: str,
    pool: list[str],
) -> dict[str, Any]:
    original = array(model, name)
    clone = f"{name}__basis_{positions[0]}_{mode}"
    terms, output = equation(final)
    occurrences: list[dict[str, Any]] = []
    if mode == "direct":
        transformed = expand_axis(original, axis, basis, mapping)
        for position in positions:
            final.input[position] = clone
            occurrences.append({"position": position, "term": terms[position]})
    elif mode == "bridge":
        transformed = bridge_axis(original, axis, basis, mapping)
        for position in positions:
            if not pool:
                raise RuntimeError("Einsum label budget exhausted")
            old_term = terms[position]
            basis_label = old_term[axis]
            old_label = pool.pop(0)
            if old_label in old_term:
                raise AssertionError((old_label, old_term))
            renamed = old_term[:axis] + old_label + old_term[axis + 1 :]
            terms[position] = basis_label + renamed
            final.input[position] = clone
            occurrences.append(
                {"position": position, "basis_label": basis_label, "old_label": old_label, "term": terms[position]}
            )
        set_equation(final, terms, output)
    else:
        raise ValueError(mode)
    add_initializer(model, clone, transformed)
    return {
        "source": name,
        "clone": clone,
        "mode": mode,
        "source_shape": list(original.shape),
        "clone_shape": list(transformed.shape),
        "coefficient_axis": axis,
        "mapping_old_to_basis": mapping,
        "selector_identity_exact": True,
        "occurrences": occurrences,
    }


def build_13(group: tuple[str, ...], modes: dict[str, str]) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task013.onnx")
    specs = {
        "C0": {"output": "C0feat", "inputs": ["one_h", "c0_h"], "coef": "Qch", "positions": [10, 13], "axis": 2},
        "C1": {"output": "C1feat", "inputs": ["one_h", "c1_h"], "coef": "Qch", "positions": [25, 28], "axis": 2},
        "T": {"output": "Tfeat", "inputs": ["one_h", "target1_h"], "coef": "Qor", "positions": [18, 22, 39, 43], "axis": 3},
    }
    basis_inputs = stable_union(specs[name]["inputs"] for name in group)
    final = replace_concat_group(
        model,
        [specs[name]["output"] for name in group],
        basis_inputs,
        "shared_basis",
        0,
    )
    pool = free_letters(final)
    rewrites = []
    for name in group:
        spec = specs[name]
        mapping = [basis_inputs.index(value) for value in spec["inputs"]]
        rewrites.append(
            coeff_clone(
                model,
                final,
                spec["positions"],
                spec["coef"],
                spec["axis"],
                mapping,
                len(basis_inputs),
                modes[name],
                pool,
            )
        )
    return model, {
        "task": 13,
        "group": list(group),
        "basis_inputs": basis_inputs,
        "basis_size": len(basis_inputs),
        "modes": modes,
        "rewrites": rewrites,
        "proof": "Each old 2-vector is exactly selected from the shared basis by a Kronecker selector absorbed into its Qch/Qor occurrence.",
    }


def build_55(modes: dict[str, str]) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task055.onnx")
    specs = {
        "H": {"output": "HZ", "inputs": ["HD", "Hs", "K"], "positions": [15, 18]},
        "V": {"output": "VZ", "inputs": ["VD", "Vs", "K"], "positions": [22, 25]},
    }
    basis_inputs = stable_union(spec["inputs"] for spec in specs.values())
    final = replace_concat_group(model, ["HZ", "VZ"], basis_inputs, "shared_basis", 0)
    pool = free_letters(final)
    rewrites = []
    for name, spec in specs.items():
        mapping = [basis_inputs.index(value) for value in spec["inputs"]]
        rewrites.append(coeff_clone(model, final, spec["positions"], "PA", 2, mapping, len(basis_inputs), modes[name], pool))
    return model, {
        "task": 55,
        "group": ["H", "V"],
        "basis_inputs": basis_inputs,
        "basis_size": len(basis_inputs),
        "modes": modes,
        "rewrites": rewrites,
        "proof": "HZ and VZ are exact PA-axis selections from [HD,Hs,K,VD,Vs].",
    }


def build_281(modes: dict[str, str]) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task281.onnx")
    specs = {
        "R": {
            "output": "VR", "inputs": ["one", "row_inv"],
            "vector_positions": [9, 21, 32], "t_positions": [6, 18, 29], "s2_positions": [4, 16, 27],
        },
        "C": {
            "output": "VC", "inputs": ["one", "col_inv"],
            "vector_positions": [13, 25, 36], "t_positions": [10, 22, 33], "s2_positions": [5, 17, 28],
        },
    }
    basis_inputs = stable_union(spec["inputs"] for spec in specs.values())
    # row_inv/col_inv each contribute two elements, hence mappings are ranges.
    offsets = {"one": [0], "row_inv": [1, 2], "col_inv": [3, 4]}
    final = replace_concat_group(model, ["VR", "VC"], basis_inputs, "shared_basis", 0)
    pool = free_letters(final)
    rewrites = []
    for name, spec in specs.items():
        mapping = [index for value in spec["inputs"] for index in offsets[value]]
        mode = modes[name]
        terms, output = equation(final)
        if mode == "direct":
            t_name = f"T__basis_{name}_direct"
            s_name = f"S2__basis_{name}_direct"
            t_value = expand_axis(array(model, "T"), 0, 5, mapping)
            s_value = expand_axis(array(model, "S2"), 1, 5, mapping)
            add_initializer(model, t_name, t_value)
            add_initializer(model, s_name, s_value)
            for position in spec["t_positions"]:
                final.input[position] = t_name
            for position in spec["s2_positions"]:
                final.input[position] = s_name
            rewrites.append(
                {
                    "mode": mode,
                    "mapping_old_to_basis": mapping,
                    "clones": {t_name: list(t_value.shape), s_name: list(s_value.shape)},
                    "identity": "Both old-index coefficient axes are injectively embedded on the basis axis.",
                }
            )
        elif mode in {"bridge_T", "bridge_S2"}:
            anchor = "T" if mode == "bridge_T" else "S2"
            axis = 0 if anchor == "T" else 1
            clone = f"{anchor}__basis_{name}_{mode}"
            transformed = bridge_axis(array(model, anchor), axis, 5, mapping)
            add_initializer(model, clone, transformed)
            anchor_positions = spec["t_positions"] if anchor == "T" else spec["s2_positions"]
            other_positions = spec["s2_positions"] if anchor == "T" else spec["t_positions"]
            occurrences = []
            for vector_position, anchor_position, other_position in zip(
                spec["vector_positions"], anchor_positions, other_positions
            ):
                if not pool:
                    raise RuntimeError("Einsum label budget exhausted")
                basis_label = terms[vector_position]
                if len(basis_label) != 1:
                    raise AssertionError(basis_label)
                old_label = pool.pop(0)
                anchor_term = terms[anchor_position]
                other_term = terms[other_position]
                terms[anchor_position] = basis_label + anchor_term.replace(basis_label, old_label)
                terms[other_position] = other_term.replace(basis_label, old_label)
                final.input[anchor_position] = clone
                occurrences.append(
                    {
                        "vector_position": vector_position,
                        "anchor_position": anchor_position,
                        "other_position": other_position,
                        "basis_label": basis_label,
                        "retained_old_label": old_label,
                    }
                )
            set_equation(final, terms, output)
            rewrites.append(
                {
                    "mode": mode,
                    "mapping_old_to_basis": mapping,
                    "clone": clone,
                    "clone_shape": list(transformed.shape),
                    "occurrences": occurrences,
                    "identity": "delta[basis,old] bridges the basis value to the unchanged T/S2 old index.",
                }
            )
        else:
            raise ValueError(mode)
    removed_s2 = remove_initializer_if_unused(model, "S2")
    return model, {
        "task": 281,
        "group": ["R", "C"],
        "basis_inputs": basis_inputs,
        "basis_size": 5,
        "modes": modes,
        "rewrites": rewrites,
        "removed_unused_original_s2": removed_s2,
        "proof": (
            "VR/VC's old 3-index is shared by T and S2. Direct mode embeds both coefficient axes; bridge mode "
            "keeps that old index and absorbs delta[basis,old] into exactly one of T/S2."
        ),
    }


AT_POSITIONS = (2, 4, 5, 6)
AB_POSITIONS = (7, 8, 9, 10)


def task99_strategy(
    model: onnx.ModelProto,
    final: onnx.NodeProto,
    target_position: int,
    coeff_positions: tuple[int, ...],
    old_label: str,
    mapping: list[int],
    basis: int,
    strategy: str,
    pool: list[str],
) -> dict[str, Any]:
    terms, output = equation(final)
    if strategy == "direct":
        if len(set(mapping)) != 7:
            raise ValueError("direct collapse requires injective mapping")
        changed = []
        for position in coeff_positions:
            name = final.input[position]
            value = array(model, name)
            replace_initializer(model, name, expand_axis(value, 0, basis, mapping))
            changed.append({"name": name, "old_shape": list(value.shape), "new_shape": [basis] + list(value.shape[1:])})
        return {"strategy": strategy, "mapping": mapping, "changed": changed, "exact": True}
    if not strategy.startswith("bridge:"):
        raise ValueError(strategy)
    anchor = strategy.split(":", 1)[1]
    anchor_position = next(position for position in coeff_positions if final.input[position] == anchor)
    if not pool:
        raise RuntimeError("no free label")
    retained_label = pool.pop(0)
    if retained_label in terms[target_position]:
        raise AssertionError(retained_label)
    for position in coeff_positions:
        terms[position] = terms[position].replace(old_label, retained_label)
    terms[anchor_position] = old_label + terms[anchor_position]
    value = array(model, anchor)
    transformed = bridge_axis(value, 0, basis, mapping)
    replace_initializer(model, anchor, transformed)
    set_equation(final, terms, output)
    return {
        "strategy": strategy,
        "mapping": mapping,
        "retained_old_index_label": retained_label,
        "anchor": anchor,
        "old_shape": list(value.shape),
        "new_shape": list(transformed.shape),
        "exact": True,
    }


def build_99(c_one: int, c_lb: int, at_strategy: str, ab_strategy: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task099.onnx")
    basis_inputs = ["one4"] * c_one + ["a", "b", "Lt", "Lt2"] + ["Lb"] * c_lb + ["Lbh", "Lb2", "Lb3"]
    basis = len(basis_inputs)
    final = replace_concat_group(model, ["At", "Ab"], basis_inputs, "shared_basis", 1)
    one_slots = [index for index, value in enumerate(basis_inputs) if value == "one4"]
    lb_slots = [index for index, value in enumerate(basis_inputs) if value == "Lb"]
    mapping_at = [one_slots[index % c_one] for index in range(3)] + [basis_inputs.index(value) for value in ("a", "b", "Lt", "Lt2")]
    mapping_ab = [lb_slots[0], lb_slots[1 % c_lb], basis_inputs.index("Lbh"), lb_slots[2 % c_lb], basis_inputs.index("Lb2"), basis_inputs.index("Lb3"), one_slots[0]]
    pool = free_letters(final)
    at = task99_strategy(model, final, 0, AT_POSITIONS, "i", mapping_at, basis, at_strategy, pool)
    ab = task99_strategy(model, final, 1, AB_POSITIONS, "j", mapping_ab, basis, ab_strategy, pool)
    return model, {
        "task": 99,
        "group": ["At", "Ab"],
        "basis_inputs": basis_inputs,
        "basis_size": basis,
        "retained_one_copies": c_one,
        "retained_lb_copies": c_lb,
        "at": at,
        "ab": ab,
        "proof": (
            "Each original feature position is selected from the common basis. Bridge mode keeps the old 7-index and "
            "multiplies one existing coefficient by delta[basis,old]; direct mode is used only for injective mappings "
            "and embeds every coefficient row on the common axis."
        ),
    }


def attempt(function: Any) -> dict[str, Any]:
    try:
        return {"pass": True, "result": function()}
    except BaseException as exc:
        return {"pass": False, "error": f"{type(exc).__name__}: {exc}"}


def full_check(model: onnx.ModelProto) -> str:
    onnx.checker.check_model(model, full_check=True)
    return "PASS"


def strict_check(model: onnx.ModelProto) -> str:
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return "PASS"


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def options(disabled: bool) -> ort.SessionOptions:
    result = ort.SessionOptions()
    result.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disabled else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    result.intra_op_num_threads = result.inter_op_num_threads = 1
    result.log_severity_level = 4
    return result


def first_input(task: int) -> np.ndarray:
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                return converted["input"]
    raise RuntimeError(f"task{task:03d}: no convertible known input")


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitizer rejected")
    return ort.InferenceSession(sanitized.SerializeToString(), options(disabled), providers=["CPUExecutionProvider"])


def dual_runtime(source: onnx.ModelProto, candidate: onnx.ModelProto, task: int) -> dict[str, Any]:
    data = first_input(task)
    result: dict[str, Any] = {}
    for disabled, label in ((True, "disable_all"), (False, "default")):
        def run() -> dict[str, Any]:
            left = np.asarray(make_session(source, disabled).run(["output"], {"input": data})[0])
            right = np.asarray(make_session(candidate, disabled).run(["output"], {"input": data})[0])
            return {
                "source_shape": list(left.shape),
                "candidate_shape": list(right.shape),
                "raw_equal": bool(np.array_equal(left, right)),
                "threshold_equal": bool(np.array_equal(left > 0, right > 0)),
                "source_nonfinite": int(left.size - np.count_nonzero(np.isfinite(left))),
                "candidate_nonfinite": int(right.size - np.count_nonzero(np.isfinite(right))),
            }
        result[label] = attempt(run)
    return result


def truthful_trace(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    existing = {value.name for value in traced.graph.output}
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    session = ort.InferenceSession(traced.SerializeToString(), options(True), providers=["CPUExecutionProvider"])
    values = session.run(names, {"input": first_input(task)})
    mismatches = []
    nonfinite = 0
    for name, value in zip(names, values):
        value = np.asarray(value)
        declared = dims(typed[name])
        actual = list(value.shape)
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def structure(source: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    before = Counter(node.op_type for node in source.graph.node)
    after = Counter(node.op_type for node in candidate.graph.node)
    new_ops = list((after - before).elements())
    forbidden_new = sorted(op for op in new_ops if op in {"Hardmax", "Gather", "GatherElements", "GatherND", "ScatterND"})
    return {
        "source_ops": dict(before),
        "candidate_ops": dict(after),
        "new_ops": new_ops,
        "forbidden_new_ops": forbidden_new,
        "no_new_lookup_or_hardmax": not forbidden_new,
        "private_zero_or_approximation": False,
    }


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def candidate_specs() -> list[tuple[str, int, Any, dict[str, Any]]]:
    specs: list[tuple[str, int, Any, dict[str, Any]]] = []
    # task013: bridge needs one extra label per occurrence; only three labels are free.
    for group in (("C0", "C1"), ("C0", "T"), ("C1", "T"), ("C0", "C1", "T")):
        for values in itertools.product(("direct", "bridge"), repeat=len(group)):
            modes = dict(zip(group, values))
            bridge_occurrences = sum({"C0": 2, "C1": 2, "T": 4}[name] for name in group if modes[name] == "bridge")
            if bridge_occurrences > 3:
                continue
            label = "task013_" + "".join(group) + "_" + "_".join(f"{name}{m[0]}" for name, m in modes.items())
            specs.append((label, 13, lambda g=group, m=modes: build_13(g, m), {"alias_count": 1}))
    for values in itertools.product(("direct", "bridge"), repeat=2):
        modes55 = dict(zip(("H", "V"), values))
        label55 = "task055_" + "_".join(f"{name}{mode[0]}" for name, mode in modes55.items())
        specs.append((label55, 55, lambda m=modes55: build_55(m), {"alias_count": 1}))
    for values in itertools.product(("direct", "bridge_T", "bridge_S2"), repeat=2):
        modes281 = dict(zip(("R", "C"), values))
        label281 = "task281_" + "_".join(f"{name}{mode}" for name, mode in modes281.items())
        specs.append((label281, 281, lambda m=modes281: build_281(m), {"alias_count": 1}))
    # task099 canonicalizes equal-size anchors. Each canonical row lists all concrete anchor aliases.
    for c_one in range(1, 4):
        for c_lb in range(1, 4):
            at_classes = [("small", "bridge:ST_u", ["ST_u"]), ("large", "bridge:FTc", ["FTc", "RTc", "DTc"])]
            ab_classes = [("small", "bridge:DB__f0a", ["DB__f0a"]), ("large", "bridge:SB", ["SB", "FBc", "RBc"])]
            if c_one == 3:
                at_classes.append(("direct", "direct", ["axis_all"]))
            if c_lb == 3:
                ab_classes.append(("direct", "direct", ["axis_all"]))
            for at_class, at_strategy, at_aliases in at_classes:
                for ab_class, ab_strategy, ab_aliases in ab_classes:
                    aliases = [f"at={left};ab={right}" for left in at_aliases for right in ab_aliases]
                    label = f"task099_o{c_one}_l{c_lb}_at{at_class}_ab{ab_class}"
                    specs.append(
                        (
                            label,
                            99,
                            lambda co=c_one, cl=c_lb, ats=at_strategy, abs_=ab_strategy: build_99(co, cl, ats, abs_),
                            {"alias_count": len(aliases), "concrete_anchor_aliases": aliases},
                        )
                    )
    return specs


def main() -> None:
    before = hashes()
    expected_authority = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
    if before["submission_base_8009.46.zip"] != expected_authority:
        raise RuntimeError("authority drift")
    if before["others/71407/task013.onnx"] != "97d6a181110e43e8a5b20031ac766bc38fa8d5787070a7bc026306a2da1c7173":
        raise RuntimeError("task013 staged drift")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    for stale in CANDIDATES.glob("*.onnx"):
        stale.unlink()
    baseline_profiles = {str(task): profile(BASE / f"task{task:03d}.onnx") for task in TASKS}
    source_models = {task: onnx.load(BASE / f"task{task:03d}.onnx") for task in TASKS}
    baseline_audits = {
        str(task): {
            "full_checker": attempt(lambda m=source_models[task]: full_check(m)),
            "strict_data_prop": attempt(lambda m=source_models[task]: strict_check(m)),
            "truthful_trace": attempt(lambda m=source_models[task], t=task: truthful_trace(m, t)),
        }
        for task in TASKS
    }
    rows = []
    specs = candidate_specs()
    print(f"candidate classes={len(specs)}", flush=True)
    for index, (label, task, builder, aliases) in enumerate(specs, 1):
        model, proof = builder()
        path = CANDIDATES / f"{label}.onnx"
        onnx.save(model, path)
        checks = {
            "full_checker": attempt(lambda m=model: full_check(m)),
            "strict_data_prop": attempt(lambda m=model: strict_check(m)),
            "truthful_trace": attempt(lambda m=model, t=task: truthful_trace(m, t)),
        }
        measured = profile(path)
        runtime = dual_runtime(source_models[task], model, task)
        structural = structure(source_models[task], model)
        baseline = baseline_profiles[str(task)]
        strict_lower = measured["cost"] < baseline["cost"]
        row = {
            "label": label,
            "task": task,
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "aliases": aliases,
            "proof": proof,
            "baseline_profile": baseline,
            "actual_profile": measured,
            "cost_delta": measured["cost"] - baseline["cost"],
            "strict_lower": strict_lower,
            "checks": checks,
            "dual_runtime_one_known": runtime,
            "structure": structural,
        }
        rows.append(row)
        print(f"[{index:02d}/{len(specs)}] {label} cost={measured['cost']} delta={row['cost_delta']:+d}", flush=True)
    lower = [row for row in rows if row["strict_lower"]]
    concrete_costs = []
    for row in rows:
        alias_values = row["aliases"].get("concrete_anchor_aliases", [row["label"]])
        for alias in alias_values:
            concrete_costs.append(
                {
                    "task": row["task"],
                    "class": row["label"],
                    "concrete_alias": alias,
                    "actual_profile": row["actual_profile"],
                    "cost_delta": row["cost_delta"],
                    "strict_lower": row["strict_lower"],
                }
            )
    after = hashes()
    result = {
        "authority": "submission_base_8009.46.zip",
        "authority_sha256": expected_authority,
        "task013_baseline": "others/71407/task013.onnx",
        "task013_sha256": before["others/71407/task013.onnx"],
        "root_hashes_before": before,
        "root_hashes_after": after,
        "root_unchanged": before == after,
        "baseline_profiles": baseline_profiles,
        "baseline_audits": baseline_audits,
        "candidate_class_count": len(rows),
        "concrete_candidate_count": sum(int(row["aliases"]["alias_count"]) for row in rows),
        "rows": rows,
        "strict_lower_count": len(lower),
        "deep_policy": {
            "known4": "NOT_RUN_NO_STRICT_LOWER" if not lower else "REQUIRED",
            "fresh10000": "NOT_RUN_NO_STRICT_LOWER" if not lower else "REQUIRED",
        },
    }
    (HERE / "screen_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "concrete_costs.json").write_text(json.dumps(concrete_costs, indent=2) + "\n")
    if not result["root_unchanged"]:
        raise RuntimeError("protected root files changed")
    if lower:
        (HERE / "strict_lower_requires_deep.json").write_text(json.dumps(lower, indent=2) + "\n")
        raise RuntimeError("strict-lower candidate exists; run known4/fresh10000 before any decision")
    print(json.dumps({key: result[key] for key in ("candidate_class_count", "concrete_candidate_count", "strict_lower_count", "root_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
