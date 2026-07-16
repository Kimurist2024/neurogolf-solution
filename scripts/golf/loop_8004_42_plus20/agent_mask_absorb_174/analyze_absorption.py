#!/usr/bin/env python3
"""Exact mask/cap absorption audit for authority task257 and task310.

The cheap global rewrites deliberately serve as counterexamples: the factors
are shared by source and output indices, so masking the initializer globally
changes source-index coefficients.  Selective clones preserve all source uses
and are algebraically exact, but their actual scorer cost is measured rather
than assumed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
AUTHORITY = REPO / "submission_base_8009.46.zip"
BASE = HERE / "base"
CANDIDATES = HERE / "candidates"
WORK = HERE / "work"
TASKS = (257, 310)

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


def get_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    tensor = next(item for item in model.graph.initializer if item.name == name)
    return np.asarray(numpy_helper.to_array(tensor))


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.asarray(array), name=name)
            )
            return
    raise KeyError(name)


def add_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    if any(item.name == name for item in model.graph.initializer):
        raise ValueError(f"initializer already exists: {name}")
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(array), name=name)
    )


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [item for item in model.graph.initializer if item.name != name]
    if len(kept) == len(model.graph.initializer):
        raise KeyError(name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def einsum_equation(node: onnx.NodeProto) -> list[str]:
    attr = next(item for item in node.attribute if item.name == "equation")
    equation = attr.s.decode()
    left, _ = equation.split("->", 1)
    return left.split(",")


def set_einsum_equation(node: onnx.NodeProto, operands: list[str]) -> None:
    attr = next(item for item in node.attribute if item.name == "equation")
    output = attr.s.decode().split("->", 1)[1]
    attr.s = (",".join(operands) + "->" + output).encode()


def drop_operands(node: onnx.NodeProto, positions: list[int]) -> None:
    operands = einsum_equation(node)
    inputs = list(node.input)
    for position in sorted(positions, reverse=True):
        del operands[position]
        del inputs[position]
    del node.input[:]
    node.input.extend(inputs)
    set_einsum_equation(node, operands)


def build_257(source: onnx.ModelProto, mode: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    node = model.graph.node[0]
    operands = einsum_equation(node)
    if operands != [
        "nihw", "ph", "pr", "qh", "sr", "qs", "th", "ur", "ut",
        "vw", "vc", "jw", "kc", "jk", "lw", "mc", "ml", "r", "c",
        "ao", "ai", "bo", "xi", "bx", "do", "yi", "yd",
    ]:
        raise AssertionError("unexpected task257 equation")
    feat = get_array(model, "feat")
    mask = get_array(model, "mask")
    masked = feat * mask[None, :]
    detail: dict[str, Any] = {
        "absorbed_initializer": "mask",
        "mask_nonzero": int(np.count_nonzero(mask)),
        "masked_factor": "feat",
    }
    if mode == "global":
        replace_initializer(model, "feat", masked)
        detail.update({
            "algebraic_claim": "INVALID_SHARED_GLOBAL_REWRITE",
            "changed_occurrences": [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15],
            "reason": "feat is also used at source h/w and at other output factors",
        })
    elif mode == "clone":
        clone = "feat_cap"
        add_initializer(model, clone, masked)
        # Absorb r into feat[p,r] and c into feat[v,c].  All h/w and the other
        # r/c factors retain the original unmasked feat.
        node.input[2] = clone
        node.input[10] = clone
        detail.update({
            "algebraic_claim": "EXACT_ALL_INPUTS",
            "changed_occurrences": [2, 10],
            "preserved_unmasked_source_occurrences": [1, 3, 6, 9, 11, 14],
            "preserved_unmasked_other_output_occurrences": [4, 7, 12, 15],
            "identity": "feat_cap[p,r]=feat[p,r]*mask[r]; feat_cap[v,c]=feat[v,c]*mask[c]",
        })
    else:
        raise ValueError(mode)
    drop_operands(node, [17, 18])
    remove_initializer(model, "mask")
    return model, detail


P_POSITIONS = {
    "P0": {"source": [0, 18], "output": [3, 21]},
    "P1": {"source": [1, 19], "output": [4, 22]},
    "P2": {"source": [2, 20], "output": [5, 23]},
}


def build_310(
    source: onnx.ModelProto, factor: str, mode: str
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    node = model.graph.node[20]
    operands = einsum_equation(node)
    if len(operands) != 31 or operands[14] != "R" or operands[30] != "S":
        raise AssertionError("unexpected task310 output equation")
    base = get_array(model, factor)
    cap = get_array(model, "cap")
    masked = base * cap[:, None]
    roles = P_POSITIONS[factor]
    detail: dict[str, Any] = {
        "absorbed_initializer": "cap",
        "cap_nonzero": int(np.count_nonzero(cap)),
        "masked_factor": factor,
        "source_occurrences": roles["source"],
        "output_occurrences": roles["output"],
    }
    if mode == "global":
        replace_initializer(model, factor, masked)
        detail.update({
            "algebraic_claim": "INVALID_SHARED_GLOBAL_REWRITE",
            "changed_occurrences": roles["source"] + roles["output"],
            "reason": f"{factor} is shared by source h/w and output R/S indices",
        })
    elif mode == "clone":
        clone = f"{factor}_cap"
        add_initializer(model, clone, masked)
        for position in roles["output"]:
            node.input[position] = clone
        detail.update({
            "algebraic_claim": "EXACT_ALL_INPUTS",
            "changed_occurrences": roles["output"],
            "preserved_unmasked_source_occurrences": roles["source"],
            "identity": f"{clone}[R,k]={factor}[R,k]*cap[R] and likewise for S",
        })
    else:
        raise ValueError(mode)
    drop_operands(node, [14, 30])
    remove_initializer(model, "cap")
    return model, detail


def full_checker(model: onnx.ModelProto) -> str:
    onnx.checker.check_model(model, full_check=True)
    return "PASS"


def strict_data_prop(model: onnx.ModelProto) -> str:
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return "PASS"


def profile(model: onnx.ModelProto, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"absorb174_{label}_") as directory:
        path = Path(directory) / "model.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    if cost < 0:
        raise RuntimeError(f"cost_of rejected {(memory, params, cost)}")
    return {"memory": memory, "params": params, "cost": cost}


def competition_profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, Any]:
    result = scoring.score_and_verify(
        copy.deepcopy(model), task, str(WORK), label=label, require_correct=False
    )
    if result is None:
        raise RuntimeError("official-compatible scorer rejected model")
    return result


def options(level: str) -> ort.SessionOptions:
    result = ort.SessionOptions()
    result.intra_op_num_threads = 1
    result.inter_op_num_threads = 1
    if level == "disabled":
        result.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif level == "default":
        result.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        raise ValueError(level)
    return result


def session(model: onnx.ModelProto, level: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        model.SerializeToString(), options(level), providers=["CPUExecutionProvider"]
    )


def synthetic_inputs(task: int) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for seed in range(8):
        rng = np.random.default_rng(174000 + task * 10 + seed)
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        # Sparse, valid one-hot cells.  Coordinates deliberately include the
        # source region beyond the output cap, where global rewrites differ.
        count = 12 if task == 310 else 8
        coords = [(8 + (3 * k + seed) % 12, 8 + (5 * k + 2 * seed) % 12) for k in range(count)]
        for k, (row, col) in enumerate(coords):
            x[0, int(rng.integers(0, 10)), row, col] = 1.0
        # Ensure some signal inside the cap too.
        x[0, (seed + 1) % 10, seed % 4, (seed * 3) % 8] = 1.0
        result.append(x)
    return result


def compare_synthetic(
    source: onnx.ModelProto, candidate: onnx.ModelProto, task: int, level: str
) -> dict[str, Any]:
    source_session = session(source, level)
    candidate_session = session(candidate, level)
    rows: list[dict[str, Any]] = []
    for index, x in enumerate(synthetic_inputs(task)):
        a = source_session.run(["output"], {"input": x})[0]
        b = candidate_session.run(["output"], {"input": x})[0]
        finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        raw_equal = bool(np.array_equal(a, b))
        mask_equal = bool(np.array_equal(a > 0.0, b > 0.0))
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        rows.append({
            "probe": index,
            "finite": finite,
            "raw_equal": raw_equal,
            "threshold_equal": mask_equal,
            "differing_raw": int(np.count_nonzero(a != b)),
            "differing_threshold": int(np.count_nonzero((a > 0.0) != (b > 0.0))),
            "max_abs_delta": float(np.nanmax(delta)) if delta.size else 0.0,
        })
    return {
        "probes": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "all_raw_equal": all(row["raw_equal"] for row in rows),
        "all_threshold_equal": all(row["threshold_equal"] for row in rows),
        "first_difference": next((row for row in rows if not row["raw_equal"]), None),
        "rows": rows,
    }


def use_map(model: onnx.ModelProto, node_index: int) -> list[dict[str, Any]]:
    node = model.graph.node[node_index]
    operands = einsum_equation(node)
    return [
        {"position": index, "initializer": name, "indices": operands[index]}
        for index, name in enumerate(node.input)
        if any(item.name == name for item in model.graph.initializer)
    ]


def evaluate(
    source: onnx.ModelProto,
    task: int,
    label: str,
    candidate: onnx.ModelProto,
    detail: dict[str, Any],
    baseline: dict[str, int],
) -> dict[str, Any]:
    path = CANDIDATES / f"task{task:03d}_{label}.onnx"
    onnx.save(candidate, path)
    row: dict[str, Any] = {
        "task": task,
        "label": label,
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "detail": detail,
        "checker_full": attempt(lambda: full_checker(candidate)),
        "strict_data_prop": attempt(lambda: strict_data_prop(candidate)),
    }
    row["profile"] = attempt(lambda: profile(candidate, f"{task}_{label}"))
    row["competition"] = attempt(
        lambda: competition_profile(candidate, task, f"{task}_{label}")
    )
    for level in ("default", "disabled"):
        row[f"synthetic_{level}"] = attempt(
            lambda level=level: compare_synthetic(source, candidate, task, level)
        )
    exact_claim = detail["algebraic_claim"] == "EXACT_ALL_INPUTS"
    if exact_claim:
        row["known_raw_bit_identical"] = attempt(
            lambda: scoring.outputs_bit_identical(source, candidate, task)
        )
    else:
        row["known_raw_bit_identical"] = {
            "ok": True,
            "result": False,
            "reason": "invalid global rewrite; synthetic algebraic counterexample is decisive",
        }
    cost = row["profile"].get("result", {}).get("cost")
    lower = isinstance(cost, int) and cost < baseline["cost"]
    synthetic_exact = all(
        row[f"synthetic_{level}"]["ok"]
        and row[f"synthetic_{level}"]["result"]["all_raw_equal"]
        for level in ("default", "disabled")
    )
    algebraically_eligible = exact_claim and synthetic_exact
    row["strictly_lower"] = lower
    row["algebraically_eligible"] = algebraically_eligible
    row["deep_gate_required"] = bool(lower and algebraically_eligible)
    if lower and not algebraically_eligible:
        row["decision"] = "REJECT_ALGEBRAIC_COUNTEREXAMPLE"
    elif algebraically_eligible and not lower:
        row["decision"] = "REJECT_COST_REGRESSION"
    elif lower and algebraically_eligible:
        row["decision"] = "REQUIRES_KNOWN4_FRESH10000"
    else:
        row["decision"] = "REJECT"
    row["known4"] = "NOT_RUN_NO_EXACT_LOWER_CANDIDATE"
    row["fresh10000"] = "NOT_RUN_NO_EXACT_LOWER_CANDIDATE"
    row["truthful_shape"] = "NOT_RUN_NO_EXACT_LOWER_CANDIDATE"
    row["ub_no_lookup"] = "NOT_RUN_NO_EXACT_LOWER_CANDIDATE"
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

    baselines = {task: profile(model, f"{task}_source") for task, model in sources.items()}
    rows: list[dict[str, Any]] = []
    for mode in ("global", "clone"):
        candidate, detail = build_257(sources[257], mode)
        rows.append(evaluate(
            sources[257], 257, f"feat_{mode}", candidate, detail, baselines[257]
        ))
    for factor in ("P0", "P1", "P2"):
        for mode in ("global", "clone"):
            candidate, detail = build_310(sources[310], factor, mode)
            rows.append(evaluate(
                sources[310], 310, f"{factor.lower()}_{mode}", candidate, detail, baselines[310]
            ))

    payload = {
        "authority": str(AUTHORITY.relative_to(REPO)),
        "authority_sha256": sha256(AUTHORITY),
        "tasks": {
            str(task): {
                "source_path": str((BASE / f"task{task:03d}.onnx").relative_to(REPO)),
                "source_sha256": sha256(BASE / f"task{task:03d}.onnx"),
                "baseline": baselines[task],
                "initializer_params": {
                    item.name: int(math.prod(item.dims))
                    for item in sources[task].graph.initializer
                },
                "einsum_initializer_uses": use_map(sources[task], 0 if task == 257 else 20),
            }
            for task in TASKS
        },
        "candidates": rows,
        "exact_lower_candidates": [
            row["label"] for row in rows
            if row["strictly_lower"] and row["algebraically_eligible"]
        ],
        "decision": "NO_APPLY" if not any(row["deep_gate_required"] for row in rows) else "DEEP_GATE_REQUIRED",
        "root_modified": False,
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "baselines": baselines,
        "rows": [
            {
                "label": row["label"],
                "cost": row["profile"].get("result", {}).get("cost"),
                "competition_correct": row["competition"].get("result", {}).get("correct"),
                "synthetic_default_exact": row["synthetic_default"].get("result", {}).get("all_raw_equal"),
                "decision": row["decision"],
            }
            for row in rows
        ],
        "exact_lower": payload["exact_lower_candidates"],
        "result": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
