#!/usr/bin/env python3
"""Build and screen exact selector-elimination candidates for tasks 398/324."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import string
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
AUTHORITY = REPO / "submission_base_8009.46.zip"
BASE = HERE / "base"
CANDIDATES = HERE / "candidates"
WORK = HERE / "work"
TASKS = (398, 324)

sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "result": fn()}
    except BaseException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def array(model: onnx.ModelProto, name: str) -> np.ndarray:
    return np.asarray(numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == name)))


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [item for item in model.graph.initializer if item.name != name]
    if len(kept) == len(model.graph.initializer):
        raise KeyError(name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def equation(node: onnx.NodeProto) -> tuple[list[str], str]:
    value = next(item.s.decode() for item in node.attribute if item.name == "equation")
    left, right = value.split("->", 1)
    return left.split(","), right


def set_equation(node: onnx.NodeProto, operands: list[str], output: str) -> None:
    next(item for item in node.attribute if item.name == "equation").s = (
        ",".join(operands) + "->" + output
    ).encode()


def set_inputs_equation(node: onnx.NodeProto, inputs: list[str], operands: list[str], output: str) -> None:
    if len(inputs) != len(operands):
        raise AssertionError((len(inputs), len(operands)))
    del node.input[:]
    node.input.extend(inputs)
    set_equation(node, operands, output)


def drop_positions(node: onnx.NodeProto, positions: list[int]) -> None:
    operands, output = equation(node)
    inputs = list(node.input)
    for position in sorted(positions, reverse=True):
        del inputs[position]
        del operands[position]
    set_inputs_equation(node, inputs, operands, output)


def build_398(source: onnx.ModelProto, mode: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    node = model.graph.node[4]
    operands, _ = equation(node)
    q4_positions = [i for i, name in enumerate(node.input) if name == "Q4"]
    if q4_positions != [7, 20, 33, 46, 52, 58, 59]:
        raise AssertionError(q4_positions)
    q4 = array(model, "Q4")
    k = array(model, "K")
    selected_k_positions = [9, 22, 35, 48, 54]
    unselected_k_positions = [2, 15, 28, 41, 61]
    if mode == "global":
        # Simultaneously compensate the first Qk/K occurrences.  This still
        # removes one Q4 from every selected second occurrence, which is the
        # unavoidable shared-use contradiction witnessed below.
        for name in ("Q0", "Q1", "Q2", "Q3"):
            replace_initializer(model, name, array(model, name) * q4)
        replace_initializer(model, "K", k * q4[:, None, None, None])
        detail = {
            "algebraic_claim": "INVALID_SHARED_GLOBAL_REWRITE",
            "q4_positions_removed": q4_positions,
            "globally_changed": ["Q0", "Q1", "Q2", "Q3", "K"],
            "shared_contradiction": (
                "each Qk/K pair occurs once without Q4 and once with Q4; one global parity "
                "cannot equal exponent 0 and 1 simultaneously"
            ),
        }
    elif mode == "k_clone":
        clone = "K_Q4"
        add_initializer(model, clone, k * q4[:, None, None, None])
        for position in selected_k_positions:
            node.input[position] = clone
        detail = {
            "algebraic_claim": "EXACT_ALL_INPUTS",
            "identity": "K_Q4[a,d,e,f]=Q4[a]*K[a,d,e,f]",
            "selected_k_positions": selected_k_positions,
            "preserved_original_k_positions": unselected_k_positions,
            "q4_square_positions": [58, 59],
            "q4_square_identity": "Q4[R]*Q4[R]=1 for Q4=[1,-1,-1]",
        }
    else:
        raise ValueError(mode)
    drop_positions(node, q4_positions)
    remove_initializer(model, "Q4")
    return model, detail


def used_letters(node: onnx.NodeProto) -> set[str]:
    operands, output = equation(node)
    return {char for char in "".join(operands) + output if char.isalpha()}


def replace_individual_onehot(node: onnx.NodeProto) -> dict[str, Any]:
    """Replace each e[target] by S[target,u] B[v,w] E[e,u] E[e,w]."""
    operands, output = equation(node)
    pool = [char for char in string.ascii_letters if char not in used_letters(node)]
    new_inputs: list[str] = []
    new_operands: list[str] = []
    replacements: list[dict[str, Any]] = []
    for name, term in zip(node.input, operands):
        if name != "onehot_values":
            new_inputs.append(name)
            new_operands.append(term)
            continue
        if len(term) != 1:
            raise AssertionError(term)
        if len(pool) < 4:
            raise RuntimeError("not enough Einsum labels")
        u, v, w, e = [pool.pop(0) for _ in range(4)]
        target = term
        new_inputs.extend(["seedsel", "bgsel", "Emap", "Emap"])
        new_operands.extend([target + u, v + w, e + u, e + w])
        replacements.append({"target": target, "aux": {"u4": u, "v2": v, "w4": w, "e3": e}})
    set_inputs_equation(node, new_inputs, new_operands, output)
    return {"node_output": node.output[0], "replacements": replacements}


def replace_paired_onehot_node21(node: onnx.NodeProto) -> dict[str, Any]:
    """Replace e[A]e[h] using e[A] * delta[A,h] with five aux labels."""
    operands, output = equation(node)
    targets = [term for name, term in zip(node.input, operands) if name == "onehot_values"]
    if targets != ["A", "h"]:
        raise AssertionError(targets)
    pool = [char for char in string.ascii_letters if char not in used_letters(node)]
    if len(pool) < 5:
        raise RuntimeError(f"not enough labels: {pool}")
    u, v, w, e, d = [pool.pop(0) for _ in range(5)]
    new_inputs = [name for name in node.input if name != "onehot_values"]
    new_operands = [term for name, term in zip(node.input, operands) if name != "onehot_values"]
    new_inputs.extend(["seedsel", "bgsel", "Emap", "Emap", "refdiff", "refdiff"])
    new_operands.extend(["A" + u, v + w, e + u, e + w, d + "A", d + "h"])
    set_inputs_equation(node, new_inputs, new_operands, output)
    return {
        "node_output": node.output[0],
        "targets": targets,
        "aux": {"u4": u, "v2": v, "w4": w, "e3": e, "d2": d},
        "identity": "e[A]*delta[A,h] = e[A]*e[h] for e=[0,1]",
    }


def build_324_synth(source: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    e = array(model, "onehot_values")
    seedsel = array(model, "seedsel")
    bgsel = array(model, "bgsel")
    emap = array(model, "Emap")
    refdiff = array(model, "refdiff")
    derived = np.einsum("au,vw,eu,ew->a", seedsel, bgsel, emap, emap)
    pair = np.einsum("Au,vw,eu,ew,dA,dh->Ah", seedsel, bgsel, emap, emap, refdiff, refdiff)
    target_pair = np.einsum("A,h->Ah", e, e)
    if not np.array_equal(derived, e) or not np.array_equal(pair, target_pair):
        raise AssertionError({"derived": derived.tolist(), "pair": pair.tolist()})
    details = [replace_individual_onehot(model.graph.node[index]) for index in (10, 11)]
    details.append(replace_paired_onehot_node21(model.graph.node[21]))
    remove_initializer(model, "onehot_values")
    return model, {
        "algebraic_claim": "EXACT_ALL_INPUTS",
        "single_identity": "einsum(seedsel[a,u],bgsel[v,w],Emap[e,u],Emap[e,w])->a == [0,1]",
        "derived_single": derived.tolist(),
        "derived_pair": pair.tolist(),
        "nodes": details,
        "new_initializers": 0,
        "new_nodes": 0,
    }


def build_324_global(source: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    e = array(model, "onehot_values")
    base_mask = e[None, :, None, None] * e[None, None, None, :]
    replace_initializer(model, "base0", array(model, "base0") * base_mask)
    replace_initializer(model, "refdiff", array(model, "refdiff") * e[None, :])
    replace_initializer(model, "signpow", array(model, "signpow") * e[:, None])
    for index in (10, 11, 21):
        node = model.graph.node[index]
        operands, output = equation(node)
        kept = [(name, term) for name, term in zip(node.input, operands) if name != "onehot_values"]
        set_inputs_equation(node, [x[0] for x in kept], [x[1] for x in kept], output)
    remove_initializer(model, "onehot_values")
    return model, {
        "algebraic_claim": "INVALID_SHARED_GLOBAL_REWRITE",
        "globally_changed": ["base0", "refdiff", "signpow"],
        "reason": "all three tensors have non-selector uses whose indices must remain unmasked",
    }


def build_324_clone(source: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    e = array(model, "onehot_values")
    base_mask = e[None, :, None, None] * e[None, None, None, :]
    clones = {
        "base0_onehot": array(model, "base0") * base_mask,
        "refdiff_onehot": array(model, "refdiff") * e[None, :],
        "signpow_onehot": array(model, "signpow") * e[:, None],
    }
    for name, value in clones.items():
        add_initializer(model, name, value)
    replacements = {
        10: {9: "base0_onehot", 12: "base0_onehot"},
        11: {7: "base0_onehot", 10: "base0_onehot", 13: "refdiff_onehot"},
        21: {3: "refdiff_onehot", 14: "signpow_onehot"},
    }
    for index in (10, 11, 21):
        node = model.graph.node[index]
        for position, name in replacements[index].items():
            node.input[position] = name
        operands, output = equation(node)
        kept = [(name, term) for name, term in zip(node.input, operands) if name != "onehot_values"]
        set_inputs_equation(node, [x[0] for x in kept], [x[1] for x in kept], output)
    remove_initializer(model, "onehot_values")
    return model, {
        "algebraic_claim": "EXACT_ALL_INPUTS",
        "clones": {name: list(value.shape) for name, value in clones.items()},
        "clone_params": int(sum(value.size for value in clones.values())),
    }


def profile(model: onnx.ModelProto, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"selector176_{label}_") as directory:
        path = Path(directory) / "model.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    if cost < 0:
        raise RuntimeError((memory, params, cost))
    return {"memory": memory, "params": params, "cost": cost}


def competition(model: onnx.ModelProto, task: int, label: str) -> dict[str, Any]:
    result = scoring.score_and_verify(copy.deepcopy(model), task, str(WORK), label=label, require_correct=False)
    if result is None:
        raise RuntimeError("official-compatible scorer rejected")
    return result


def checker(model: onnx.ModelProto) -> str:
    onnx.checker.check_model(model, full_check=True)
    return "PASS"


def strict(model: onnx.ModelProto) -> str:
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return "PASS"


def opts(disable: bool) -> ort.SessionOptions:
    value = ort.SessionOptions()
    value.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    value.intra_op_num_threads = value.inter_op_num_threads = 1
    value.log_severity_level = 4
    return value


def synthetic_inputs(task: int) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split][:2]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted["input"])
    for seed in range(4):
        rng = np.random.default_rng(176000 + task + seed)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        height = int(rng.integers(10, 21))
        width = int(rng.integers(10, 21))
        colors = rng.integers(0, 10, size=(height, width))
        for r in range(height):
            for c in range(width):
                x[0, colors[r, c], r, c] = 1.0
        rows.append(x)
    return rows


def compare(source: onnx.ModelProto, candidate: onnx.ModelProto, task: int, disable: bool) -> dict[str, Any]:
    sa = ort.InferenceSession(source.SerializeToString(), opts(disable), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(candidate.SerializeToString(), opts(disable), providers=["CPUExecutionProvider"])
    first = None
    raw_equal = threshold_equal = finite = 0
    inputs = synthetic_inputs(task)
    for index, x in enumerate(inputs):
        a = np.asarray(sa.run(["output"], {"input": x})[0])
        b = np.asarray(sb.run(["output"], {"input": x})[0])
        same_raw = bool(np.array_equal(a, b))
        same_threshold = bool(np.array_equal(a > 0, b > 0))
        is_finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        raw_equal += int(same_raw)
        threshold_equal += int(same_threshold)
        finite += int(is_finite)
        if first is None and not same_raw:
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            first = {
                "probe": index,
                "raw_differences": int(np.count_nonzero(a != b)),
                "threshold_differences": int(np.count_nonzero((a > 0) != (b > 0))),
                "max_abs_delta": float(np.nanmax(delta)),
            }
    return {
        "total": len(inputs),
        "raw_equal": raw_equal,
        "threshold_equal": threshold_equal,
        "finite": finite,
        "all_raw_equal": raw_equal == len(inputs),
        "all_threshold_equal": threshold_equal == len(inputs),
        "first_difference": first,
    }


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    banned = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
    ops = [node.op_type for node in model.graph.node]
    reasons: list[str] = []
    if model.functions:
        reasons.append("local_functions")
    if model.graph.sparse_initializer:
        reasons.append("sparse_initializer")
    if any(item.domain not in {"", "ai.onnx"} for item in model.opset_import):
        reasons.append("custom_opset")
    if any(node.domain not in {"", "ai.onnx"} for node in model.graph.node):
        reasons.append("custom_node_domain")
    if any(op.upper() in banned or "SEQUENCE" in op.upper() for op in ops):
        reasons.append("banned_op")
    if any(attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS} for node in model.graph.node for attr in node.attribute):
        reasons.append("nested_graph")
    if any(item.external_data or item.data_location == onnx.TensorProto.EXTERNAL for item in model.graph.initializer):
        reasons.append("external_data")
    lookup = sorted(set(ops) & {"TfIdfVectorizer", "CategoryMapper", "GatherND", "ScatterND", "ScatterElements"})
    if lookup:
        reasons.append("lookup_ops")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "lookup_ops": lookup,
        "lookup_free": not lookup,
        "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
    }


def evaluate(source: onnx.ModelProto, task: int, label: str, model: onnx.ModelProto, detail: dict[str, Any], baseline: dict[str, int]) -> dict[str, Any]:
    path = CANDIDATES / f"task{task:03d}_{label}.onnx"
    onnx.save(model, path)
    row: dict[str, Any] = {
        "task": task,
        "label": label,
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "detail": detail,
        "checker_full": attempt(lambda: checker(model)),
        "strict_data_prop": attempt(lambda: strict(model)),
        "structural_ub_no_lookup": structural(model),
        "profile": attempt(lambda: profile(model, f"{task}_{label}")),
        "competition": attempt(lambda: competition(model, task, f"{task}_{label}")),
        "synthetic_default": attempt(lambda: compare(source, model, task, False)),
        "synthetic_disabled": attempt(lambda: compare(source, model, task, True)),
    }
    cost = row["profile"].get("result", {}).get("cost")
    exact_claim = detail["algebraic_claim"] == "EXACT_ALL_INPUTS"
    # Algebraic candidates inherit task324 authority's pre-existing default-
    # optimizer TopK shape failure.  The official runtime is DISABLE_ALL, so
    # use its raw equality to decide whether a formally exact lower candidate
    # deserves the independent known4/truthful deep audit.  The default failure
    # is recorded and remains a fail-closed adoption gate in that audit.
    synthetic_exact = bool(
        row["synthetic_disabled"]["ok"]
        and row["synthetic_disabled"]["result"]["all_raw_equal"]
    )
    row["strict_lower"] = isinstance(cost, int) and cost < baseline["cost"]
    row["exact_pre_gate"] = bool(exact_claim and synthetic_exact)
    if row["strict_lower"] and row["exact_pre_gate"]:
        row["decision"] = "DEEP_AUDIT_REQUIRED"
    elif row["strict_lower"]:
        row["decision"] = "REJECT_ALGEBRAIC_OR_RUNTIME_COUNTEREXAMPLE"
    else:
        row["decision"] = "REJECT_NOT_LOWER"
    return row


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    sources: dict[int, onnx.ModelProto] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            path = BASE / f"task{task:03d}.onnx"
            path.write_bytes(data)
            sources[task] = onnx.load_model_from_string(data)
    baselines = {task: profile(model, f"{task}_base") for task, model in sources.items()}
    specs = []
    for mode in ("global", "k_clone"):
        model, detail = build_398(sources[398], mode)
        specs.append((398, f"q4_{mode}", model, detail))
    for label, builder in (
        ("onehot_global", build_324_global),
        ("onehot_clone", build_324_clone),
        ("onehot_synth", build_324_synth),
    ):
        model, detail = builder(sources[324])
        specs.append((324, label, model, detail))
    rows = [evaluate(sources[task], task, label, model, detail, baselines[task]) for task, label, model, detail in specs]
    payload = {
        "authority": str(AUTHORITY.relative_to(REPO)),
        "authority_sha256": sha256(AUTHORITY),
        "baselines": baselines,
        "source_sha256": {str(task): sha256(BASE / f"task{task:03d}.onnx") for task in TASKS},
        "rows": rows,
        "deep_audit": [row["label"] for row in rows if row["decision"] == "DEEP_AUDIT_REQUIRED"],
    }
    out = HERE / "screen_results.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "baselines": baselines,
        "rows": [{
            "label": row["label"],
            "cost": row["profile"].get("result", {}).get("cost"),
            "correct": row["competition"].get("result", {}).get("correct"),
            "synthetic_exact": row["exact_pre_gate"],
            "decision": row["decision"],
        } for row in rows],
        "deep_audit": payload["deep_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
