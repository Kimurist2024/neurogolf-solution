#!/usr/bin/env python3
"""Exact graph-level precontraction enumeration for tasks 074/200/211."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE / "base"
CANDIDATES = HERE / "candidates"
TASKS = (74, 200, 211)

sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


PROTECTED = (
    REPO / "submission_base_8009.46.zip",
    REPO / "submission.zip",
    REPO / "all_scores.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO)): sha256(path) for path in PROTECTED}


def equation(node: onnx.NodeProto) -> tuple[list[str], str]:
    value = next(item.s.decode() for item in node.attribute if item.name == "equation")
    left, right = value.split("->", 1)
    return left.split(","), right


def set_equation(node: onnx.NodeProto, terms: list[str], output: str) -> None:
    next(item for item in node.attribute if item.name == "equation").s = (
        ",".join(terms) + "->" + output
    ).encode()


def array(model: onnx.ModelProto, name: str) -> np.ndarray:
    return np.asarray(numpy_helper.to_array(next(item for item in model.graph.initializer if item.name == name)))


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


def prune_unused(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input}
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def rewrite_einsum(
    node: onnx.NodeProto,
    replacements: dict[int, tuple[str, str]],
    skipped: set[int],
) -> None:
    terms, output = equation(node)
    inputs = list(node.input)
    new_inputs = []
    new_terms = []
    for position, (name, term) in enumerate(zip(inputs, terms)):
        if position in skipped:
            continue
        if position in replacements:
            name, term = replacements[position]
        new_inputs.append(name)
        new_terms.append(term)
    del node.input[:]
    node.input.extend(new_inputs)
    set_equation(node, new_terms, output)


FAMILIES_74: dict[str, dict[str, Any]] = {
    "T_plain": {"feature": "Tfeat", "selected": False, "occurrences": [(2, 3), (17, 18), (50, 51), (65, 66)]},
    "T_selected": {"feature": "Tfeat", "selected": True, "occurrences": [(8, 9, 10), (23, 24, 25), (56, 57, 58), (71, 72, 73)]},
    "B_plain": {"feature": "Bfeat", "selected": False, "occurrences": [(4, 5), (19, 20), (52, 53), (67, 68)]},
    "B_selected": {"feature": "Bfeat", "selected": True, "occurrences": [(11, 12, 13), (26, 27, 28), (59, 60, 61), (74, 75, 76)]},
    "G_plain": {"feature": "Gfeat", "selected": False, "occurrences": [(6, 7), (21, 22), (54, 55), (69, 70)]},
    "G_selected": {"feature": "Gfeat", "selected": True, "occurrences": [(14, 15, 16), (29, 30, 31), (62, 63, 64), (77, 78, 79)]},
}


def build_74_precontract(selected_families: tuple[str, ...]) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task074.onnx")
    node = model.graph.node[0]
    original_terms, _ = equation(node)
    replacements: dict[int, tuple[str, str]] = {}
    skipped: set[int] = set()
    tensors = []
    for family_name in selected_families:
        family = FAMILIES_74[family_name]
        feature = array(model, family["feature"])
        poly = array(model, "poly3")
        if family["selected"]:
            combined = np.einsum("...d,d,di->...i", feature, array(model, "sel_hi"), poly)
        else:
            combined = np.einsum("...d,di->...i", feature, poly)
        combined_name = f"pc_{family_name}"
        add_initializer(model, combined_name, combined.astype(feature.dtype))
        tensors.append({"family": family_name, "name": combined_name, "shape": list(combined.shape)})
        for occurrence in family["occurrences"]:
            feature_position = occurrence[0]
            poly_position = occurrence[-1]
            feature_term = original_terms[feature_position]
            poly_term = original_terms[poly_position]
            if feature_term[-1] != poly_term[0]:
                raise AssertionError((feature_term, poly_term))
            replacements[feature_position] = (combined_name, feature_term[:-1] + poly_term[1:])
            skipped.update(occurrence[1:])
    rewrite_einsum(node, replacements, skipped)
    removed = prune_unused(model)
    return model, {
        "task": 74,
        "kind": "pair_or_triple_precontract",
        "families": list(selected_families),
        "tensors": tensors,
        "removed_unused_initializers": removed,
        "proof": (
            "Each replacement is the exact finite sum over the private degree index: "
            "C[...,i]=sum_d feature[...,d]*(sel[d] if selected else 1)*poly3[d,i]. "
            "All four occurrences of each chosen family reuse the same C."
        ),
    }


def build_74_selector(features: tuple[str, ...], into_poly: bool = False) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task074.onnx")
    node = model.graph.node[0]
    replacements: dict[int, tuple[str, str]] = {}
    skipped: set[int] = set()
    created = []
    if into_poly:
        name = "poly3_selected"
        value = array(model, "poly3") * array(model, "sel_hi")[:, None]
        add_initializer(model, name, value)
        created.append({"name": name, "shape": list(value.shape)})
        selected_names = ("T_selected", "B_selected", "G_selected")
        for family_name in selected_names:
            for feature_position, selector_position, poly_position in FAMILIES_74[family_name]["occurrences"]:
                terms, _ = equation(node)
                replacements[poly_position] = (name, terms[poly_position])
                skipped.add(selector_position)
    else:
        for short in features:
            family_name = short + "_selected"
            source = FAMILIES_74[family_name]["feature"]
            name = source + "_selected"
            value = array(model, source) * array(model, "sel_hi").reshape((1,) * (array(model, source).ndim - 1) + (3,))
            add_initializer(model, name, value)
            created.append({"name": name, "shape": list(value.shape)})
            for feature_position, selector_position, _poly_position in FAMILIES_74[family_name]["occurrences"]:
                terms, _ = equation(node)
                replacements[feature_position] = (name, terms[feature_position])
                skipped.add(selector_position)
    rewrite_einsum(node, replacements, skipped)
    removed = prune_unused(model)
    return model, {
        "task": 74,
        "kind": "selector_absorption_poly" if into_poly else "selector_absorption_feature",
        "features": list(features),
        "created": created,
        "removed_unused_initializers": removed,
        "proof": "Elementwise multiplication by sel_hi is moved from an Einsum operand into the selected feature/poly coefficient exactly.",
    }


def build_74_permutation(permutation: tuple[int, int, int]) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task074.onnx")
    for name in ("Tfeat", "Bfeat", "Gfeat"):
        replace_initializer(model, name, np.take(array(model, name), permutation, axis=-1))
    replace_initializer(model, "poly3", np.take(array(model, "poly3"), permutation, axis=0))
    replace_initializer(model, "sel_hi", np.take(array(model, "sel_hi"), permutation, axis=0))
    return model, {
        "task": 74,
        "kind": "global_degree_channel_permutation",
        "permutation": list(permutation),
        "proof": "Every occurrence of the shared degree index is reindexed by the same bijection; finite sums are unchanged.",
    }


def build_200(mode: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task200.onnx")
    final = model.graph.node[8]
    br = array(model, "Br")
    bias = array(model, "conv_bias")
    replacements: dict[int, tuple[str, str]] = {}
    skipped: set[int] = set()
    created = []
    if mode == "pair_vector":
        name = "Br_bias_vector"
        value = np.einsum("rw,r->w", br, bias)
        add_initializer(model, name, value)
        replacements[3] = (name, "w")
        skipped.add(4)
        proof = "Br_bias_vector[w]=sum_r Br[r,w]*conv_bias[r]."
    elif mode == "masked_clone":
        name = "Br_bias_masked"
        value = br * bias[:, None]
        add_initializer(model, name, value)
        replacements[3] = (name, "rw")
        skipped.add(4)
        proof = "The selector is absorbed elementwise into a Br clone; r is then summed implicitly."
    elif mode in {"channel_swap", "channel_swap_pair"}:
        permutation = np.array([1, 0])
        replace_initializer(model, "W_kernel", array(model, "W_kernel")[permutation])
        replace_initializer(model, "Br", br[permutation])
        # The permuted Conv bias [0,1] is exactly the already-stored OneHot values.
        model.graph.node[6].input[2] = "oh_values"
        perm_name = "channel_swap"
        perm = np.array([[0, 1], [1, 0]], dtype=np.float16)
        add_initializer(model, perm_name, perm)
        terms, output = equation(final)
        inputs = list(final.input)
        new_inputs = [inputs[0], perm_name, inputs[1], inputs[2]]
        new_terms = [terms[0], "tu", terms[1].replace("t", "u"), terms[2].replace("t", "u")]
        if mode == "channel_swap":
            new_inputs += ["Br", "oh_values"]
            new_terms += [terms[3], terms[4]]
        else:
            name = "swapped_Br_oh_vector"
            value = np.einsum("rw,r->w", br[permutation], array(model, "oh_values"))
            add_initializer(model, name, value)
            new_inputs.append(name)
            new_terms.append("w")
        del final.input[:]
        final.input.extend(new_inputs)
        set_equation(final, new_terms, output)
        removed = prune_unused(model)
        return model, {
            "task": 200,
            "kind": mode,
            "created": [perm_name] + ([name] if mode == "channel_swap_pair" else []),
            "removed_unused_initializers": removed,
            "proof": (
                "W_kernel/Br channels are swapped and conv_bias'=[0,1] reuses oh_values. "
                "channel_swap[t,u] reindexes the dynamic Bfac channel, while the second Br/oh contraction is invariant."
            ),
        }
    else:
        raise ValueError(mode)
    created.append({"name": name, "shape": list(value.shape)})
    rewrite_einsum(final, replacements, skipped)
    removed = prune_unused(model)
    return model, {
        "task": 200,
        "kind": mode,
        "created": created,
        "removed_unused_initializers": removed,
        "proof": proof,
    }


def make_211_plan(model: onnx.ModelProto, d_mode: str, m_mode: str) -> tuple[dict[int, tuple[str, str]], set[int], list[dict[str, Any]], list[str]]:
    p = array(model, "P")
    d = array(model, "D")
    m = array(model, "M")
    replacements: dict[int, tuple[str, str]] = {}
    skipped: set[int] = set()
    created = []
    aliases = []
    if d_mode == "t_vector":
        value = np.einsum("ct,t->c", p, d)
        add_initializer(model, "PD_t_vector", value)
        replacements[16] = ("PD_t_vector", "c")
        skipped.add(17)
        created.append({"name": "PD_t_vector", "shape": list(value.shape)})
    elif d_mode == "PD_all":
        value = p * d[None, :]
        add_initializer(model, "PD", value)
        for left, selector, _right in ((1, 2, 3), (4, 5, 6), (7, 8, 9)):
            replacements[left] = ("PD", {1: "ra", 4: "rx", 7: "ru"}[left])
            skipped.add(selector)
        replacements[16] = ("PD", "ct")
        skipped.add(17)
        created.append({"name": "PD", "shape": list(value.shape)})
        aliases = [f"triple_absorb_sides={bits}" for bits in itertools.product("LR", repeat=3)]
    elif d_mode != "none":
        raise ValueError(d_mode)

    if m_mode in {"pair_vector", "PM_all", "chain_PM", "chain_MP", "full"}:
        vector = np.einsum("cy,yY->c", p, m)
    if m_mode == "pair_vector":
        add_initializer(model, "PM_rowsum", vector)
        replacements[18] = ("PM_rowsum", "c")
        skipped.add(19)
        replacements[20] = ("PM_rowsum", "c")
        skipped.add(21)
        created.append({"name": "PM_rowsum", "shape": list(vector.shape)})
    elif m_mode == "PM_all":
        value = p @ m
        add_initializer(model, "PM", value)
        replacements[12] = ("PM", "cQ")
        skipped.add(13)
        replacements[18] = ("PM", "cY")
        skipped.add(19)
        replacements[20] = ("PM", "cN")
        skipped.add(21)
        created.append({"name": "PM", "shape": list(value.shape)})
    elif m_mode == "chain_PM":
        value = p @ m
        add_initializer(model, "PM_chain", value)
        replacements[12] = ("PM_chain", "cQ")
        skipped.add(13)
        created.append({"name": "PM_chain", "shape": list(value.shape)})
    elif m_mode == "chain_MP":
        value = m @ p.T
        add_initializer(model, "MP_chain", value)
        replacements[13] = ("MP_chain", "qw")
        skipped.add(14)
        created.append({"name": "MP_chain", "shape": list(value.shape)})
    elif m_mode == "full":
        kernel = p @ m @ p.T
        add_initializer(model, "PMP_kernel", kernel)
        add_initializer(model, "PM_rowsum", vector)
        replacements[12] = ("PMP_kernel", "cw")
        skipped.update({13, 14})
        replacements[18] = ("PM_rowsum", "c")
        skipped.add(19)
        replacements[20] = ("PM_rowsum", "c")
        skipped.add(21)
        created.extend(
            [
                {"name": "PMP_kernel", "shape": list(kernel.shape)},
                {"name": "PM_rowsum", "shape": list(vector.shape)},
            ]
        )
    elif m_mode != "none":
        raise ValueError(m_mode)
    return replacements, skipped, created, aliases


def build_211(d_mode: str, m_mode: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task211.onnx")
    replacements, skipped, created, aliases = make_211_plan(model, d_mode, m_mode)
    rewrite_einsum(model.graph.node[0], replacements, skipped)
    removed = prune_unused(model)
    return model, {
        "task": 211,
        "kind": "precontract",
        "d_mode": d_mode,
        "m_mode": m_mode,
        "created": created,
        "equivalent_aliases": aliases,
        "removed_unused_initializers": removed,
        "proof": (
            "PD applies D elementwise on one side of each shared channel; PM/MP/PMP are exact finite matrix "
            "contractions. PM_rowsum also contracts the dangling M output index, which appears nowhere else."
        ),
    }


def build_211_permutation() -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = onnx.load(BASE / "task211.onnx")
    permutation = np.array([1, 0])
    replace_initializer(model, "P", array(model, "P")[:, permutation])
    replace_initializer(model, "D", array(model, "D")[permutation])
    replace_initializer(model, "M", array(model, "M")[np.ix_(permutation, permutation)])
    return model, {
        "task": 211,
        "kind": "global_latent_channel_permutation",
        "permutation": [1, 0],
        "proof": "Every occurrence of each independent latent 2-channel index is reindexed by the same bijection on P/D and both M axes.",
    }


def specs() -> list[tuple[str, int, Callable[[], tuple[onnx.ModelProto, dict[str, Any]]]]]:
    result = []
    names = tuple(FAMILIES_74)
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            label = "task074_pc_" + "-".join(subset)
            result.append((label, 74, lambda s=subset: build_74_precontract(s)))
    for size in range(1, 4):
        for subset in itertools.combinations(("T", "B", "G"), size):
            label = "task074_sel_" + "".join(subset)
            result.append((label, 74, lambda s=subset: build_74_selector(s)))
    result.append(("task074_sel_poly", 74, lambda: build_74_selector((), into_poly=True)))
    for permutation in itertools.permutations((0, 1, 2)):
        if permutation != (0, 1, 2):
            label = "task074_perm_" + "".join(map(str, permutation))
            result.append((label, 74, lambda p=permutation: build_74_permutation(p)))
    for mode in ("pair_vector", "masked_clone", "channel_swap", "channel_swap_pair"):
        result.append((f"task200_{mode}", 200, lambda m=mode: build_200(m)))
    d_modes = ("none", "t_vector", "PD_all")
    m_modes = ("none", "pair_vector", "PM_all", "chain_PM", "chain_MP", "full")
    for d_mode, m_mode in itertools.product(d_modes, m_modes):
        if d_mode == m_mode == "none":
            continue
        result.append(
            (
                f"task211_D-{d_mode}_M-{m_mode}",
                211,
                lambda d=d_mode, m=m_mode: build_211(d, m),
            )
        )
    result.append(("task211_perm_10", 211, build_211_permutation))
    return result


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    if cost < 0:
        raise RuntimeError((memory, params, cost))
    return {"memory": memory, "params": params, "cost": cost}


def main() -> None:
    before = hashes()
    if before["submission_base_8009.46.zip"] != "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927":
        raise RuntimeError("authority drift")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    for path in CANDIDATES.glob("*.onnx"):
        path.unlink()
    baseline = {str(task): profile(BASE / f"task{task:03d}.onnx") for task in TASKS}
    rows = []
    candidates = specs()
    print(f"candidate_count={len(candidates)}", flush=True)
    for index, (label, task, builder) in enumerate(candidates, 1):
        model, proof = builder()
        path = CANDIDATES / f"{label}.onnx"
        onnx.save(model, path)
        actual = profile(path)
        base = baseline[str(task)]
        row = {
            "label": label,
            "task": task,
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "proof": proof,
            "baseline_profile": base,
            "actual_profile": actual,
            "cost_delta": actual["cost"] - base["cost"],
            "strict_lower": actual["cost"] < base["cost"],
            "new_ops": [],
            "private_zero_or_approximation": False,
            "algebraic_equivalence": "EXACT_FINITE_CONTRACTION_OR_BIJECTIVE_REINDEX",
        }
        rows.append(row)
        print(f"[{index:03d}/{len(candidates)}] {label} cost={actual['cost']} delta={row['cost_delta']:+d}", flush=True)
    lower = [row for row in rows if row["strict_lower"]]
    after = hashes()
    result = {
        "authority": "submission_base_8009.46.zip",
        "authority_sha256": before["submission_base_8009.46.zip"],
        "root_hashes_before": before,
        "root_hashes_after": after,
        "root_unchanged": before == after,
        "baseline_profiles": baseline,
        "candidate_count": len(rows),
        "rows": rows,
        "strict_lower_count": len(lower),
        "deep_policy": {
            "full_checker": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
            "strict_data_prop": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
            "truthful_trace": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
            "known4": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
            "fresh_each_seed_10000": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
        },
    }
    (HERE / "profile_results.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["root_unchanged"]:
        raise RuntimeError("protected root drift")
    if lower:
        (HERE / "strict_lower_requires_deep.json").write_text(json.dumps(lower, indent=2) + "\n")
        raise RuntimeError("strict-lower found: lower-only deep audit required")
    print(json.dumps({"candidate_count": len(rows), "strict_lower_count": 0, "root_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
