#!/usr/bin/env python3
"""Current-authority cost<=10 pattern transplant scan for tasks at cost 11..100.

The scan is deliberately fail closed.  It never writes the authority archive or
root ledgers.  Candidate graphs must be finite, statically shaped, free of
forbidden operators/graphs/functions/sparse tensors, and exact on the complete
known corpus before they are profiled.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
EXPECTED = (1, 10, 30, 30)
OUT = HERE / "pattern_scan.json"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT))
from scripts.lib import scoring  # noqa: E402


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TEMPLATES = import_path(
    "lowcost401_templates", ROOT / "scripts/golf/extra15_cost25_scan_294/scan.py"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(model: onnx.ModelProto, threads: int = 1) -> ort.InferenceSession | None:
    try:
        onnx.checker.check_model(model, full_check=True)
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            return None
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        return ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return None


def known_cases(task: int) -> list[dict[str, np.ndarray]]:
    raw = scoring.load_examples(task)
    result = []
    for subset in ("train", "test", "arc-gen"):
        for example in raw[subset]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                result.append(converted)
    return result


def evaluate(
    runtime: ort.InferenceSession,
    items: list[dict[str, np.ndarray]],
    limit: int | None = None,
) -> dict[str, Any]:
    chosen = items if limit is None else items[:limit]
    right = wrong = errors = nonfinite = small = shape = 0
    minimum_positive = None
    first_wrong = None
    for index, item in enumerate(chosen):
        try:
            raw = runtime.run(["output"], {"input": item["input"]})[0]
            if raw.shape != item["output"].shape:
                shape += 1
                wrong += 1
                if first_wrong is None:
                    first_wrong = index
                continue
            if not np.isfinite(raw).all():
                nonfinite += 1
            positive = raw[raw > 0]
            if positive.size:
                value = float(positive.min())
                minimum_positive = value if minimum_positive is None else min(minimum_positive, value)
                if bool(np.any(positive < 0.25)):
                    small += 1
            if np.isfinite(raw).all() and np.array_equal(raw > 0, item["output"] > 0):
                right += 1
            else:
                wrong += 1
                if first_wrong is None:
                    first_wrong = index
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_wrong is None:
                first_wrong = f"{index}:{type(exc).__name__}"
    return {
        "total": len(chosen), "right": right, "wrong": wrong, "errors": errors,
        "nonfinite_cases": nonfinite, "shape_mismatches": shape,
        "small_positive_cases": small, "minimum_positive": minimum_positive,
        "first_wrong": first_wrong,
    }


def clean(row: dict[str, Any]) -> bool:
    return bool(
        row["right"] == row["total"]
        and row["wrong"] == row["errors"] == row["nonfinite_cases"]
        == row["shape_mismatches"] == row["small_positive_cases"] == 0
    )


def canonical_io(model: onnx.ModelProto) -> bool:
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return False
    for value, name in ((model.graph.input[0], "input"), (model.graph.output[0], "output")):
        if value.name != name or not value.type.HasField("tensor_type"):
            return False
        tensor = value.type.tensor_type
        if tensor.elem_type not in (TensorProto.FLOAT, TensorProto.BOOL):
            return False
        dims = tensor.shape.dim
        if len(dims) != 4 or any(d.HasField("dim_param") for d in dims):
            return False
        if tuple(int(d.dim_value) for d in dims) != EXPECTED:
            return False
    return True


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "reasons": [f"checker_or_shape:{type(exc).__name__}:{exc}"]}
    if not canonical_io(inferred):
        reasons.append("noncanonical_io")
    if model.functions:
        reasons.append("local_functions")
    if model.graph.sparse_initializer:
        reasons.append("sparse_initializers")
    for opset in model.opset_import:
        if opset.domain not in ("", "ai.onnx"):
            reasons.append(f"nonstandard_domain:{opset.domain}")
    banned = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
    init = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    nonfinite = [name for name, array in init.items() if not np.isfinite(array).all()]
    if nonfinite:
        reasons.append("nonfinite_initializers:" + ",".join(nonfinite))
    max_einsum_inputs = 0
    conv_bias_findings = []
    nested = 0
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in banned or "SEQUENCE" in upper:
            reasons.append(f"banned_op:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested += 1
        if node.op_type == "Einsum":
            max_einsum_inputs = max(max_einsum_inputs, len(node.input))
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3:
            weight = init.get(node.input[1])
            bias = init.get(node.input[2])
            if weight is not None and bias is not None:
                expected = int(weight.shape[1] if node.op_type == "ConvTranspose" else weight.shape[0])
                # group is one throughout this low-cost lane.  A mismatch is rejected
                # rather than relying on ORT's unchecked short-bias reads.
                if bias.ndim != 1 or int(bias.size) != expected:
                    conv_bias_findings.append({
                        "node": node.output[0], "bias_size": int(bias.size),
                        "expected_output_channels": expected,
                    })
    if nested:
        reasons.append(f"nested_graphs:{nested}")
    if conv_bias_findings:
        reasons.append("conv_bias_shape_mismatch")
    static_names = {v.name for v in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info)}
    graph_outputs = {v.name for v in inferred.graph.output}
    missing = [name for node in inferred.graph.node for name in node.output if name and name not in graph_outputs and name not in static_names]
    if missing:
        reasons.append("missing_intermediate_shape")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "ops": [node.op_type for node in model.graph.node],
        "nodes": len(model.graph.node),
        "initializer_elements": int(sum(array.size for array in init.values())),
        "max_einsum_inputs": max_einsum_inputs,
        "conv_bias_findings": conv_bias_findings,
        "nonfinite_initializers": nonfinite,
    }


def source(model: onnx.ModelProto, name: str, family: str, detail: str) -> dict[str, Any]:
    data = model.SerializeToString()
    return {"name": name, "family": family, "detail": detail, "sha256": digest(data), "_data": data, "_model": model}


def unique_sources(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = row["sha256"]
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def make_model(node: onnx.NodeProto, initializers: list[onnx.TensorProto] | None = None) -> onnx.ModelProto:
    graph = helper.make_graph(
        [node], "lowcost401",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(EXPECTED))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(EXPECTED))],
        initializers or [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def color_gather(task: int, items: list[dict[str, np.ndarray]]) -> dict[str, Any] | None:
    indices = []
    for out_channel in range(10):
        matches = []
        for in_channel in range(10):
            if all(np.array_equal(item["output"][0, out_channel], item["input"][0, in_channel]) for item in items):
                matches.append(in_channel)
        if not matches:
            return None
        indices.append(matches[0])
    item = numpy_helper.from_array(np.asarray(indices, dtype=np.int64), "idx")
    model = make_model(helper.make_node("Gather", ["input", "idx"], ["output"], axis=1), [item])
    return source(model, f"task{task:03d}_synth_color_gather", "cost10_gather_synthesis", f"known-exact channel map {indices}")


def remove_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]


def einsum_initializer_subsets(task: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    if len(base.graph.node) != 1 or base.graph.node[0].op_type != "Einsum":
        return []
    node = base.graph.node[0]
    equation = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "equation"), None)
    if not isinstance(equation, bytes):
        return []
    lhs, rhs = equation.decode().split("->")
    terms = lhs.split(",")
    if len(terms) != len(node.input):
        return []
    init_names = {item.name for item in base.graph.initializer}
    distinct = sorted({name for name in node.input if name in init_names})
    # Exhaustive for compact factorizations; cap pathological historical graphs.
    if not distinct or len(distinct) > 9:
        return []
    result = []
    for count in range(1, len(distinct) + 1):
        for removed_tuple in itertools.combinations(distinct, count):
            removed = set(removed_tuple)
            keep = [index for index, name in enumerate(node.input) if name not in removed]
            if not keep:
                continue
            kept_terms = [terms[index] for index in keep]
            labels = set("".join(kept_terms))
            if any(char.isalpha() and char not in labels for char in rhs):
                continue
            model = copy.deepcopy(base)
            target = model.graph.node[0]
            del target.input[:]
            target.input.extend([node.input[index] for index in keep])
            attr = next(a for a in target.attribute if a.name == "equation")
            attr.s = (",".join(kept_terms) + "->" + rhs).encode()
            remove_unused_initializers(model)
            result.append(source(
                model,
                f"task{task:03d}_drop_init_{'_'.join(removed_tuple)}",
                "cost10_einsum_factor_ablation",
                "remove whole initializer operands and their now-unused parameter tensors",
            ))
    return unique_sources(result)


def conv_crop_variants(task: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    if len(base.graph.node) != 1 or base.graph.node[0].op_type != "ConvTranspose":
        return []
    node = base.graph.node[0]
    if len(node.input) < 2:
        return []
    init = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    kernel = init.get(node.input[0]) if node.input[0] in init else init.get(node.input[1])
    kernel_name = node.input[0] if node.input[0] in init else node.input[1]
    if kernel is None or kernel.ndim != 4:
        return []
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    dilation = list(attrs.get("dilations", [1, 1]))
    result = []
    for r0 in range(kernel.shape[2]):
        for r1 in range(r0 + 1, kernel.shape[2] + 1):
            for c0 in range(kernel.shape[3]):
                for c1 in range(c0 + 1, kernel.shape[3] + 1):
                    if (r0, r1, c0, c1) == (0, kernel.shape[2], 0, kernel.shape[3]):
                        continue
                    updated = [
                        pads[0] - r0 * dilation[0],
                        pads[1] - c0 * dilation[1],
                        pads[2] - (kernel.shape[2] - r1) * dilation[0],
                        pads[3] - (kernel.shape[3] - c1) * dilation[1],
                    ]
                    if min(updated) < 0:
                        continue
                    cropped = kernel[:, :, r0:r1, c0:c1]
                    if not np.isfinite(cropped).all():
                        continue
                    model = copy.deepcopy(base)
                    item = next(value for value in model.graph.initializer if value.name == kernel_name)
                    item.CopyFrom(numpy_helper.from_array(cropped.astype(kernel.dtype), kernel_name))
                    target = model.graph.node[0]
                    old = next((a for a in target.attribute if a.name == "pads"), None)
                    if old is None:
                        target.attribute.extend([helper.make_attribute("pads", updated)])
                    else:
                        del old.ints[:]
                        old.ints.extend(updated)
                    result.append(source(
                        model, f"task{task:03d}_finite_crop_r{r0}{r1}_c{c0}{c1}",
                        "cost_le10_finite_convtranspose_crop",
                        "crop a finite contiguous kernel support and compensate pads exactly",
                    ))
    # Optional-bias ablation is safe to test only when it also removes a real initializer.
    if len(node.input) >= 3 and node.input[2] in init:
        model = copy.deepcopy(base)
        del model.graph.node[0].input[2]
        remove_unused_initializers(model)
        result.append(source(model, f"task{task:03d}_drop_bias", "cost_le10_convtranspose_bias_ablation", "remove optional bias"))
    return unique_sources(result)


def main() -> int:
    started = time.monotonic()
    HERE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    costs: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            if 11 <= cost <= 100:
                costs[task] = cost
    items = {task: known_cases(task) for task in costs}
    previous_tasks: dict[int, dict[str, Any]] = {}
    if OUT.exists():
        try:
            previous = json.loads(OUT.read_text(encoding="utf-8"))
            if previous.get("authority_sha256") == AUTHORITY_SHA256:
                previous_tasks = {int(row["task"]): row for row in previous.get("tasks", [])}
        except Exception:
            previous_tasks = {}
    report: dict[str, Any] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "scope": "all authority tasks with cost 11..100",
        "task_count": len(costs),
        "acceptance": {
            "known_exact": True, "finite_only": True, "no_runtime_errors": True,
            "no_small_positives": True, "no_ub_or_shape_cloak": True,
            "half_cost_goal": True,
        },
        "lowcost_templates": [], "rejected_lowcost_templates": [],
        "tasks": [], "finalists": [], "counters": {},
    }
    counters: Counter[str] = Counter()
    with zipfile.ZipFile(AUTHORITY) as archive:
        # Every distinct current cost<=10 model is a literal cross-task template.
        low = []
        low_task_costs = []
        with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cost = int(row["cost"])
                if cost <= 10:
                    low_task_costs.append((int(row["task"][4:]), cost))
        for task, cost in low_task_costs:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_from_string(data)
            row = source(model, f"authority_task{task:03d}_cost{cost}", "literal_current_cost_le10", "exact immutable authority member")
            row["source_task"] = task
            row["source_cost"] = cost
            low.append(row)
        low = unique_sources(low)
        admitted_low = []
        for row in low:
            audit = structure(row["_model"])
            compact = {key: value for key, value in row.items() if not key.startswith("_")}
            compact["structure"] = audit
            if audit["pass"]:
                report["lowcost_templates"].append(compact)
                admitted_low.append(row)
            else:
                report["rejected_lowcost_templates"].append(compact)
        generic = []
        for row in TEMPLATES.generic_variants():
            candidate = source(row["_model"], row["name"], row["family"], row["proof"])
            if structure(candidate["_model"])["pass"]:
                generic.append(candidate)
        generic = unique_sources([*admitted_low, *generic])
        generic_runtime = [(row, make_session(row["_model"])) for row in generic]

        for task in sorted(costs, key=lambda value: (costs[value], value)):
            if task in previous_tasks:
                report["tasks"].append(previous_tasks[task])
                report["finalists"].extend(previous_tasks[task].get("survivors", []))
                continue
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_from_string(base_data)
            specific = []
            synth = color_gather(task, items[task])
            if synth is not None:
                specific.append(synth)
            specific.extend(einsum_initializer_subsets(task, base))
            specific.extend(conv_crop_variants(task, base))
            # Reuse the established input-only sub-equation generator as another
            # direct transfer from the cost-0 task067/task179 family.
            # The legacy input-only helper is exponential in repeated-input arity.
            # Its entire generic output-only family is already in ``generic`` and
            # the prior exhaustive evidence is referenced in REPORT.md; do not
            # regenerate millions of duplicate task202-style sub-equations here.
            specific = unique_sources(specific)
            variants = [*generic_runtime, *((row, make_session(row["_model"])) for row in specific)]
            task_row = {
                "task": task, "authority_cost": costs[task],
                "half_target": costs[task] // 2,
                "known_count": len(items[task]), "variant_count": len(variants),
                "quick_exact": 0, "known_exact": 0, "strict_lower": 0,
                "half_cost": 0, "best_quick": {"right": -1}, "survivors": [],
            }
            for row, runtime in variants:
                counters["candidate_task_evaluations"] += 1
                if runtime is None:
                    counters["session_reject"] += 1
                    continue
                quick = evaluate(runtime, items[task], min(12, len(items[task])))
                if quick["right"] > task_row["best_quick"]["right"]:
                    task_row["best_quick"] = {"name": row["name"], "family": row["family"], **quick}
                if not clean(quick):
                    continue
                task_row["quick_exact"] += 1
                full = evaluate(runtime, items[task])
                if not clean(full):
                    continue
                task_row["known_exact"] += 1
                audit = structure(row["_model"])
                if not audit["pass"]:
                    counters["structural_reject"] += 1
                    continue
                with tempfile.TemporaryDirectory(prefix=f"low401_{task:03d}_", dir="/tmp") as work:
                    try:
                        profile = scoring.score_and_verify(row["_model"], task, work, label="candidate", require_correct=True)
                    except Exception:
                        profile = None
                if profile is None or int(profile["cost"]) >= costs[task]:
                    continue
                task_row["strict_lower"] += 1
                stable, margin = scoring.model_margin_stable(row["_model"], task)
                if not stable:
                    counters["margin_reject"] += 1
                    continue
                half = int(profile["cost"]) * 2 <= costs[task]
                if half:
                    task_row["half_cost"] += 1
                path = CANDIDATES / f"task{task:03d}_{row['name']}_cost{int(profile['cost'])}.onnx"
                path.write_bytes(row["_data"])
                finalist = {
                    "task": task, "name": row["name"], "family": row["family"],
                    "detail": row["detail"], "sha256": row["sha256"],
                    "authority_cost": costs[task], "candidate_cost": int(profile["cost"]),
                    "half_target_met": half, "known": full, "structure": audit,
                    "margin_stable": bool(stable), "margin_min": margin,
                    "path": str(path.relative_to(ROOT)),
                }
                task_row["survivors"].append(finalist)
                report["finalists"].append(finalist)
            report["tasks"].append(task_row)
            # Crash-safe progress evidence; the final write below adds counters and
            # elapsed time, but completed task rows are never lost during a long run.
            OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "task": task, "cost": costs[task], "variants": len(variants),
                "best": task_row["best_quick"], "strict_lower": task_row["strict_lower"],
                "half": task_row["half_cost"],
            }), flush=True)
    counters["candidate_task_evaluations"] = sum(int(row["variant_count"]) for row in report["tasks"])
    counters["lowcost_unique_models"] = len(report["lowcost_templates"]) + len(report["rejected_lowcost_templates"])
    counters["lowcost_structurally_admitted"] = len(report["lowcost_templates"])
    counters["lowcost_structurally_rejected"] = len(report["rejected_lowcost_templates"])
    report["counters"] = dict(counters)
    report["elapsed_seconds"] = time.monotonic() - started
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finalists": report["finalists"], "counters": report["counters"], "elapsed": report["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
