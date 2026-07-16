#!/usr/bin/env python3
"""Runtime-audit every history graph that can beat the 8006.61 authority."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INVENTORY = HERE / "inventory.json"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP_CLOAK = {
    "TFIDFVECTORIZER",
    "HARDMAX",
    "GATHERND",
    "SCATTERELEMENTS",
    "SCATTERND",
    "CENTERCROPPAD",
    "RESIZE",
    "SHRINK",
    "TOPK",
}
CONFIGS = (
    ("disable_all_t1", "disable_all", 1),
    ("disable_all_t4", "disable_all", 4),
    ("default_t1", "default", 1),
    ("default_t4", "default", 4),
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def cases(task: int) -> list[dict[str, np.ndarray]]:
    examples = scoring.load_examples(task)
    result: list[dict[str, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for raw in examples.get(split, []):
            converted = scoring.convert_to_numpy(raw)
            if converted is not None:
                result.append(converted)
    return result


def options(optimization: str, threads: int, profile_prefix: str | None = None) -> ort.SessionOptions:
    value = ort.SessionOptions()
    value.log_severity_level = 4
    value.intra_op_num_threads = threads
    value.inter_op_num_threads = 1
    value.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if profile_prefix is not None:
        value.enable_profiling = True
        value.profile_file_prefix = profile_prefix
    return value


def summarize_run(
    runtime: ort.InferenceSession,
    known: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "total": len(known),
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "nan_values": 0,
        "positive_infinity_values": 0,
        "negative_infinity_values": 0,
        "near_positive_values_0_to_0_25": 0,
        "near_positive_examples": 0,
        "min_positive": None,
        "observed_output_shapes": [],
        "first_failure": None,
    }
    positives: list[float] = []
    shapes: set[tuple[int, ...]] = set()
    for index, case in enumerate(known, start=1):
        try:
            raw = runtime.run(
                [runtime.get_outputs()[0].name],
                {runtime.get_inputs()[0].name: case["input"]},
            )[0]
        except Exception as exc:  # noqa: BLE001
            row["runtime_errors"] += 1
            if row["first_failure"] is None:
                row["first_failure"] = {
                    "case": index,
                    "runtime": f"{type(exc).__name__}: {exc}",
                }
            continue
        shapes.add(tuple(int(dim) for dim in raw.shape))
        correct = np.array_equal(raw > 0.0, case["output"].astype(bool))
        row["right" if correct else "wrong"] += 1
        if not correct and row["first_failure"] is None:
            row["first_failure"] = {"case": index, "kind": "gold_mismatch"}
        finite = np.isfinite(raw)
        row["nonfinite_values"] += int((~finite).sum())
        row["nan_values"] += int(np.isnan(raw).sum())
        row["positive_infinity_values"] += int(np.isposinf(raw).sum())
        row["negative_infinity_values"] += int(np.isneginf(raw).sum())
        near = finite & (raw > 0.0) & (raw < 0.25)
        near_count = int(near.sum())
        row["near_positive_values_0_to_0_25"] += near_count
        row["near_positive_examples"] += int(near_count > 0)
        positive = raw[finite & (raw > 0.0)]
        if positive.size:
            positives.append(float(positive.min()))
    row["min_positive"] = min(positives) if positives else None
    row["observed_output_shapes"] = [list(item) for item in sorted(shapes)]
    row["known_perfect"] = bool(
        row["right"] == row["total"]
        and row["wrong"] == 0
        and row["runtime_errors"] == 0
    )
    return row


def profile_disabled_all(
    task: int,
    model: onnx.ModelProto,
    known: list[dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, int] | None]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return {"session_error": "sanitize_model rejected graph", "known_perfect": False}, None
    trace_path: str | None = None
    with tempfile.TemporaryDirectory(prefix=f"expand20g_{task:03d}_", dir=HERE) as tmp:
        prefix = os.path.join(tmp, f"profile_{uuid.uuid4().hex[:8]}")
        try:
            runtime = ort.InferenceSession(
                sanitized.SerializeToString(),
                options("disable_all", 1, prefix),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001
            return {"session_error": f"{type(exc).__name__}: {exc}", "known_perfect": False}, None
        row = summarize_run(runtime, known)
        try:
            trace_path = runtime.end_profiling()
            memory, params = scoring.score_network(sanitized, trace_path)
            if memory is None or params is None or memory < 0 or params < 0:
                cost = None
            else:
                cost = {
                    "memory": int(memory),
                    "params": int(params),
                    "cost": int(memory) + int(params),
                }
        except Exception as exc:  # noqa: BLE001
            row["profile_error"] = f"{type(exc).__name__}: {exc}"
            cost = None
        finally:
            if trace_path:
                try:
                    Path(trace_path).unlink()
                except FileNotFoundError:
                    pass
        return row, cost


def audit_config(
    model: onnx.ModelProto,
    known: list[dict[str, np.ndarray]],
    optimization: str,
    threads: int,
) -> dict[str, Any]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return {"session_error": "sanitize_model rejected graph", "known_perfect": False}
    try:
        runtime = ort.InferenceSession(
            sanitized.SerializeToString(),
            options(optimization, threads),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # noqa: BLE001
        return {"session_error": f"{type(exc).__name__}: {exc}", "known_perfect": False}
    return summarize_run(runtime, known)


def runtime_shapes(
    task: int,
    model: onnx.ModelProto,
    optimization: str,
    known: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected graph")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(sanitized), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(sanitized)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    runtime = ort.InferenceSession(
        traced.SerializeToString(),
        options(optimization, 1),
        providers=["CPUExecutionProvider"],
    )
    values = runtime.run(names, {runtime.get_inputs()[0].name: known[0]["input"]})
    mismatches = [
        {
            "tensor": name,
            "declared": value_shape(typed[name]),
            "runtime": list(np.asarray(value).shape),
        }
        for name, value in zip(names, values)
        if value_shape(typed[name]) != list(np.asarray(value).shape)
    ]
    return {
        "optimization": optimization,
        "traced_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def structure(
    task: int,
    model: onnx.ModelProto,
    known: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"shape:{type(exc).__name__}:{exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    shape_rows: list[dict[str, Any]] = []
    if checker and strict:
        for optimization in ("disable_all", "default"):
            try:
                shape_rows.append(runtime_shapes(task, model, optimization, known))
            except Exception as exc:  # noqa: BLE001
                shape_rows.append(
                    {
                        "optimization": optimization,
                        "truthful": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        shape_rows = [{"truthful": False, "error": "checker_or_strict_failed"}]
    domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
        | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    nested = any(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    conv_findings = check_conv_bias(model)
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive_shapes": all(
            value_shape(value) is not None
            for value in values
            if value.type.HasField("tensor_type")
        ),
        "truthful_runtime_shapes": all(row.get("truthful") for row in shape_rows),
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and value_shape(model.graph.input[0]) == [1, 10, 30, 30]
            and value_shape(model.graph.output[0]) == [1, 10, 30, 30]
        ),
        "standard_domain": not domains,
        "no_banned_ops": not banned,
        "no_nested_functions_sparse": (
            not nested and not model.functions and not model.graph.sparse_initializer
        ),
        "conv_family_ub0": not conv_findings,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "errors": errors,
        "runtime_shape_evidence": shape_rows,
        "custom_domains": domains,
        "banned_ops": banned,
        "conv_bias_findings": conv_findings,
        "lookup_or_cloak_ops": sorted(
            {node.op_type for node in model.graph.node if node.op_type.upper() in LOOKUP_CLOAK}
        ),
        "max_einsum_inputs": max_einsum,
        "giant_einsum": max_einsum > 16,
        "op_histogram": dict(sorted(ops.items())),
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    inventory = json.loads(INVENTORY.read_text())
    tasks = [int(task) for task in inventory["targets"]]
    known_cache = {task: cases(task) for task in tasks}
    authority_costs = {
        int(row["task"]): int(row["authority_cost"])
        for row in inventory["summary"]
    }
    leads = [
        row
        for task in tasks
        for row in inventory["rows_by_task"][str(task)]
        if row.get("could_be_actual_strict_lower") and row.get("candidate_path")
    ]
    result_rows: list[dict[str, Any]] = []
    for index, lead in enumerate(leads, start=1):
        task = int(lead["task"])
        path = ROOT / str(lead["candidate_path"])
        model = onnx.load(path)
        disabled, cost = profile_disabled_all(task, model, known_cache[task])
        actual_lower = bool(
            cost is not None and 0 <= cost["cost"] < authority_costs[task]
        )
        known_four: dict[str, Any] = {"disable_all_t1": disabled}
        # Only an actual-lower, complete disabled run can become an LB probe.
        if actual_lower and disabled.get("known_perfect"):
            for label, optimization, threads in CONFIGS[1:]:
                known_four[label] = audit_config(
                    model, known_cache[task], optimization, threads
                )
        known4_complete = len(known_four) == 4 and all(
            row.get("known_perfect") for row in known_four.values()
        )
        if actual_lower and known4_complete:
            static = structure(task, model, known_cache[task])
        else:
            static = {
                "pass": False,
                "not_run": "requires actual strict lower and complete known×4",
                "lookup_or_cloak_ops": lead.get("structure", {}).get("lookup_or_cloak_ops", []),
                "giant_einsum": lead.get("structure", {}).get("giant_einsum", False),
                "custom_domains": lead.get("structure", {}).get("custom_domains", []),
            }
        exact = bool(lead.get("exact_computational_graph_equivalent"))
        any_nonfinite = any(
            int(row.get("nonfinite_values", 0)) > 0 for row in known_four.values()
        )
        any_near = any(
            int(row.get("near_positive_values_0_to_0_25", 0)) > 0
            for row in known_four.values()
        )
        if actual_lower and known4_complete and static.get("pass"):
            if exact and not any_nonfinite and not any_near:
                decision = "EXACT_FIXED_CANDIDATE"
            else:
                decision = "LB_PROBE_REQUIRED"
        elif not actual_lower:
            decision = "REJECT_ACTUAL_NOT_STRICT_LOWER_OR_UNSCORABLE"
        elif not known4_complete:
            decision = "REJECT_KNOWN_OR_RUNTIME"
        else:
            decision = "REJECT_SCHEMA_SHAPE_OR_UB"
        risks: list[str] = []
        if lead.get("structure", {}).get("giant_einsum"):
            risks.append("giant_einsum")
        if lead.get("structure", {}).get("lookup_or_cloak_ops"):
            risks.append("lookup_or_cloak")
        if lead.get("structure", {}).get("custom_domains"):
            risks.append("custom_domain")
        if any_nonfinite:
            risks.append("nonfinite_output")
        if any_near:
            risks.append("near_positive_output")
        result_rows.append(
            {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "sources": lead["sources"],
                "source_count": lead["source_count"],
                "authority_cost": authority_costs[task],
                "static_cost_floor": lead.get("static_cost_floor"),
                "actual_cost": cost,
                "actual_strict_lower": actual_lower,
                "gain_if_valid": (
                    math.log(authority_costs[task] / cost["cost"])
                    if actual_lower and cost and cost["cost"] > 0
                    else 0.0
                ),
                "exact_computational_graph_equivalent": exact,
                "known_complete_four_configs": known_four,
                "known_four_complete_pass": known4_complete,
                "structure": static,
                "risk_classification": sorted(set(risks)),
                "decision": decision,
            }
        )
        print(
            f"[{index}/{len(leads)}] task{task:03d} "
            f"cost={cost['cost'] if cost else None}/{authority_costs[task]} "
            f"known={disabled.get('right', 0)}/{disabled.get('total', '?')} "
            f"decision={decision}",
            flush=True,
        )
        if index % 5 == 0:
            (HERE / "audit_partial.json").write_text(
                json.dumps({"rows": result_rows}, indent=2) + "\n"
            )
    probes = [row for row in result_rows if row["decision"] == "LB_PROBE_REQUIRED"]
    exact = [row for row in result_rows if row["decision"] == "EXACT_FIXED_CANDIDATE"]
    result = {
        "authority": inventory["authority"],
        "authority_sha256": inventory["authority_sha256"],
        "targets": tasks,
        "history_inventory": inventory["inventory"],
        "leads_profiled": len(result_rows),
        "rows": result_rows,
        "exact_fixed_candidates": exact,
        "lb_probe_required": probes,
        "decision_counts": dict(sorted(Counter(row["decision"] for row in result_rows).items())),
        "fixed_black_net_set": [18, 48, 112, 134, 168, 198, 233, 251, 277, 286, 365, 366],
        "fixed_adoption_set": [13, 70, 158, 254, 267, 323, 379],
        "policy_notes": {
            "black_set_is_net_specific_not_task_permanent": True,
            "giant_lookup_private_nonfinite_fresh_below_90_not_probe_rejects": True,
            "non_exact_requires_lb_probe": True,
            "local_fresh_is_ranking_only": True,
        },
        "protected_files_modified": [],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(
        json.dumps(
            {
                "authority": inventory["authority"],
                "authority_sha256": inventory["authority_sha256"],
                "status": "LB_PROBE_REQUIRED",
                "candidates": [
                    {
                        "task": row["task"],
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "authority_cost": row["authority_cost"],
                        "candidate_cost": row["actual_cost"]["cost"],
                        "projected_gain": row["gain_if_valid"],
                        "risk_classification": row["risk_classification"],
                        "sources": row["sources"],
                    }
                    for row in probes
                ],
                "merge_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "authority": inventory["authority"],
                "authority_sha256": inventory["authority_sha256"],
                "winners": [
                    {
                        "task": row["task"],
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "authority_cost": row["authority_cost"],
                        "candidate_cost": row["actual_cost"]["cost"],
                        "gain": row["gain_if_valid"],
                        "proof": "strong computational graph equivalence after metadata/unused-init normalization",
                    }
                    for row in exact
                ],
                "merge_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result["decision_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
