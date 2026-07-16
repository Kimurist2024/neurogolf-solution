#!/usr/bin/env python3
"""One of three disjoint workers for the 8012.15 cost-501..1000 lane.

Every worker owns a stable round-robin subset of the eligible tasks and runs
the same three searches on that subset: history/archive rebasing, exact graph
simplification, and transfer of current low-cost graph patterns.  This is an
evidence-only lane; only files below ``restart8012_cost501_1000_3w_408`` are
written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
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
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
EXPECTED = (1, 10, 30, 30)
THRESHOLD = 0.90
FRESH_PER_SEED = 2_000

# Current 8012.15 census, sorted by (cost, task).  The nine excluded members
# are retained here so the evidence proves that the whole requested band was
# enumerated before policy exclusions were applied.
BAND: tuple[tuple[int, int], ...] = (
    (34, 511), (363, 512), (284, 517), (378, 520), (368, 521), (69, 523),
    (237, 529), (19, 535), (35, 544), (66, 551), (328, 553), (238, 562),
    (165, 570), (117, 604), (46, 622), (277, 631), (107, 656), (198, 661),
    (364, 685), (131, 688), (12, 710), (361, 745), (251, 755), (201, 803),
    (382, 820), (280, 828), (157, 847), (330, 885), (182, 929), (370, 931),
    (319, 975),
)

# docs/golf/private_zero_tasks.md plus the explicitly named latest-LB black
# set.  task251 is also retained as a known LB-black catalogue entry.
PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    118, 133, 134, 138, 145, 157, 158, 168, 169, 170, 173, 174, 178, 182,
    185, 187, 191, 192, 196, 198, 202, 204, 205, 208, 209, 216, 219, 222,
    233, 246, 251, 255, 273, 277, 285, 286, 302, 319, 325, 333, 343, 346,
    361, 365, 366, 372, 377, 379, 391, 393, 396,
}
EXPLICIT_LATEST_LB_BLACK = {70, 134, 202, 343}
ELIGIBLE = tuple(task for task, _cost in BAND if task not in PRIVATE_ZERO_OR_UNSOUND)
COSTS = {task: cost for task, cost in BAND}
CHANGED_FROM_8011_05 = {378, 69, 165, 117, 46, 107, 131, 361, 201, 157, 330, 182, 370}


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY = import_path(
    "cost408_policy_support",
    ROOT / "scripts/golf/cost101_250_half_307/scan_policy95_history.py",
)
SUPPORT = POLICY.load_support()
SUPPORT.POLICY_THRESHOLD = THRESHOLD
SUPPORT.FRESH_PER_SEED = FRESH_PER_SEED
PATTERN = import_path(
    "cost408_pattern_support",
    ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/scan_patterns.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def task_from_name(name: str) -> int | None:
    matches = re.findall(r"task[_-]?(\d{3})(?!\d)", name, re.IGNORECASE)
    return int(matches[-1]) if matches else None


def compact_runtime(row: dict[str, Any]) -> dict[str, Any]:
    return POLICY.compact(row)


def runtime_pass(row: dict[str, Any]) -> bool:
    return POLICY.row_pass(row, threshold=THRESHOLD)


def parameter_count(model: onnx.ModelProto) -> int:
    value = SUPPORT.scoring.calculate_params(model)
    return int(value) if value is not None else 10**18


def quick_preflight(model: onnx.ModelProto) -> list[str]:
    reasons: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        reasons.append("full_checker")
    inferred = None
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
    except Exception:
        reasons.append("strict_shape")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        reasons.append("io_count")
    else:
        def shape(value: onnx.ValueInfoProto) -> tuple[int | None, ...]:
            return tuple(
                int(dim.dim_value) if dim.HasField("dim_value") else None
                for dim in value.type.tensor_type.shape.dim
            )
        if model.graph.input[0].name != "input" or shape(model.graph.input[0]) != EXPECTED:
            reasons.append("input_io")
        if model.graph.output[0].name != "output" or shape(model.graph.output[0]) != EXPECTED:
            reasons.append("output_io")
    if model.functions or model.graph.sparse_initializer:
        reasons.append("function_or_sparse")
    for item in model.graph.initializer:
        try:
            array = onnx.numpy_helper.to_array(item)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                reasons.append("nonfinite_initializer")
                break
        except Exception:
            reasons.append("unreadable_initializer")
            break
    for node in model.graph.node:
        if (
            node.op_type in {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
            or "Sequence" in node.op_type
        ):
            reasons.append(f"banned:{node.op_type}")
        if node.domain not in ("", "ai.onnx"):
            reasons.append(f"domain:{node.domain}")
        if any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for attr in node.attribute
        ):
            reasons.append("nested_graph")
    try:
        if list(SUPPORT.check_conv_bias(copy.deepcopy(model))):
            reasons.append("conv_bias_ub")
    except Exception:
        reasons.append("conv_bias_audit_error")
    if inferred is not None:
        typed = {
            value.name: value
            for value in (
                list(inferred.graph.input)
                + list(inferred.graph.output)
                + list(inferred.graph.value_info)
            )
        }
        graph_outputs = {value.name for value in inferred.graph.output}
        for node in inferred.graph.node:
            for name in node.output:
                if not name or name in graph_outputs:
                    continue
                value = typed.get(name)
                if value is None or not value.type.HasField("tensor_type"):
                    reasons.append("missing_intermediate_shape")
                    continue
                dims = value.type.tensor_type.shape.dim
                if not dims or any(
                    dim.HasField("dim_param")
                    or not dim.HasField("dim_value")
                    or int(dim.dim_value) <= 0
                    for dim in dims
                ):
                    reasons.append("nonstatic_intermediate_shape")
    return sorted(set(reasons))


def failfast_known(data: bytes, cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    maximum_failures = total - math.ceil(THRESHOLD * total)
    right = wrong = evaluated = 0
    errors = nonfinite = shape = small = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    reason = None
    try:
        runtime = SUPPORT.make_session(data, True, 1)
    except Exception as exc:
        return {
            "total": total, "evaluated": 0, "right": 0, "wrong": 0,
            "accuracy": 0.0, "errors": 1, "nonfinite_cases": 0,
            "nonfinite_elements": 0, "runtime_shape_mismatches": 0,
            "small_positive_elements_0_to_0_25": 0,
            "session_error": f"{type(exc).__name__}: {exc}",
            "early_reject_reason": "session_error",
        }
    for index, example in enumerate(cases):
        evaluated += 1
        benchmark = SUPPORT.scoring.convert_to_numpy(example)
        if benchmark is None:
            errors += 1
            reason = "conversion_error"
            break
        try:
            raw = np.asarray(runtime.run(["output"], {"input": benchmark["input"]})[0])
        except Exception:
            errors += 1
            reason = "runtime_error"
            break
        if tuple(raw.shape) != EXPECTED:
            shape += 1
            reason = "shape_mismatch"
            break
        bad = int(np.count_nonzero(~np.isfinite(raw)))
        if bad:
            nonfinite += bad
            reason = "nonfinite"
            break
        positives = raw > 0
        small_here = int(np.count_nonzero(positives & (raw < 0.25)))
        if small_here:
            small += small_here
            reason = "small_positive"
            break
        correct = bool(np.array_equal(positives, benchmark["output"] > 0))
        right += int(correct)
        wrong += int(not correct)
        if np.any(positives):
            minimum_positive = min(minimum_positive, float(raw[positives].min()))
        if np.any(~positives):
            maximum_nonpositive = max(maximum_nonpositive, float(raw[~positives].max()))
        if wrong > maximum_failures:
            reason = "accuracy_upper_bound_below_90"
            break
    accuracy = right / total if reason is None else (right + total - evaluated) / total
    return {
        "total": total, "evaluated": evaluated, "right": right, "wrong": wrong,
        "accuracy": accuracy, "accuracy_is_upper_bound": reason is not None,
        "errors": errors, "nonfinite_cases": int(nonfinite > 0),
        "nonfinite_elements": nonfinite, "runtime_shape_mismatches": shape,
        "small_positive_elements_0_to_0_25": small,
        "minimum_positive": None if minimum_positive == math.inf else minimum_positive,
        "maximum_nonpositive": None if maximum_nonpositive == -math.inf else maximum_nonpositive,
        "sign_mismatch_cases_vs_disable_threads1": 0,
        "sign_mismatch_cells_vs_disable_threads1": 0,
        "early_reject_reason": reason,
    }


def prune(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]


def serialized_variant(
    model: onnx.ModelProto, name: str, family: str, detail: str
) -> tuple[bytes, dict[str, Any]]:
    data = model.SerializeToString()
    return data, {"name": name, "family": family, "detail": detail, "sha256": digest(data)}


def exact_simplifier_variants(task: int, base: onnx.ModelProto) -> Iterable[tuple[bytes, dict[str, Any]]]:
    init_names = {item.name for item in base.graph.initializer}
    for index, node in enumerate(base.graph.node):
        # General branch bypasses can create enormous-but-statically-typed
        # broadcasts that are legal to the checker yet unsafe to execute.
        # Restrict this in-process lane to unary no-op-like operators; broader
        # bypass families require per-candidate crash isolation.
        if node.op_type not in {"Identity", "Cast", "Dropout"}:
            continue
        if len(node.output) != 1 or not node.output[0]:
            continue
        old = node.output[0]
        for slot, source in enumerate(node.input):
            if not source or source in init_names or source == old:
                continue
            model = copy.deepcopy(base)
            del model.graph.node[index]
            is_output = any(value.name == old for value in model.graph.output)
            for consumer in model.graph.node:
                for position, name in enumerate(consumer.input):
                    if name == old:
                        consumer.input[position] = source
            if is_output:
                model.graph.node.extend([
                    helper.make_node("Identity", [source], [old], name=f"bypass_{index}_{slot}")
                ])
            prune(model)
            yield serialized_variant(
                model, f"task{task:03d}_bypass_n{index}_i{slot}", "exact_simplifier",
                f"remove {node.op_type} and route dynamic input {slot}",
            )

    optional_last = {"Conv": 2, "ConvTranspose": 2, "Gemm": 2, "Clip": 1}
    for index, node in enumerate(base.graph.node):
        minimum = optional_last.get(node.op_type)
        if minimum is None:
            continue
        for slot in range(len(node.input) - 1, minimum - 1, -1):
            if slot != len(node.input) - 1 or node.input[slot] not in init_names:
                continue
            model = copy.deepcopy(base)
            del model.graph.node[index].input[slot]
            prune(model)
            yield serialized_variant(
                model, f"task{task:03d}_drop_{node.op_type}_{index}_{slot}",
                "exact_simplifier", f"drop trailing optional initializer {slot}",
            )

    for row in PATTERN.einsum_initializer_subsets(task, base):
        yield row["_data"], {
            key: value for key, value in row.items() if not key.startswith("_")
        }
    # Bound the crop generator: finite compact kernels only.
    if len(base.graph.node) == 1 and base.graph.node[0].op_type == "ConvTranspose":
        init = {item.name: onnx.numpy_helper.to_array(item) for item in base.graph.initializer}
        arrays = [init.get(name) for name in base.graph.node[0].input if name in init]
        kernels = [array for array in arrays if array is not None and array.ndim == 4]
        if kernels and kernels[0].shape[2] <= 10 and kernels[0].shape[3] <= 10:
            for row in PATTERN.conv_crop_variants(task, base):
                yield row["_data"], {
                    key: value for key, value in row.items() if not key.startswith("_")
                }


def lowcost_variants(
    task: int,
    base: onnx.ModelProto,
    cases: list[dict[str, Any]],
    generic: list[tuple[bytes, dict[str, Any]]],
) -> Iterable[tuple[bytes, dict[str, Any]]]:
    yield from generic
    converted = [SUPPORT.scoring.convert_to_numpy(case) for case in cases]
    converted = [item for item in converted if item is not None]
    synth = PATTERN.color_gather(task, converted)
    if synth is not None:
        yield synth["_data"], {
            key: value for key, value in synth.items() if not key.startswith("_")
        }
    for row in PATTERN.einsum_initializer_subsets(task, base):
        yield row["_data"], {
            key: value for key, value in row.items() if not key.startswith("_")
        }


def build_generic_lowcost() -> list[tuple[bytes, dict[str, Any]]]:
    result: list[tuple[bytes, dict[str, Any]]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(AUTHORITY) as archive:
        # Derive the source list from the 8012.15 score ledger, but use only
        # immutable authority bytes as templates.
        import csv
        with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
            low = [
                (int(row["task"][4:]), int(row["cost"]))
                for row in csv.DictReader(handle) if int(row["cost"]) <= 10
            ]
        for source_task, cost in low:
            data = archive.read(f"task{source_task:03d}.onnx")
            key = digest(data)
            if key in seen:
                continue
            seen.add(key)
            result.append((data, {
                "name": f"authority_task{source_task:03d}_cost{cost}",
                "family": "lowcost_literal_transfer",
                "detail": "current authority cost<=10 graph",
                "sha256": key,
            }))
    for row in PATTERN.TEMPLATES.generic_variants():
        data = row["_model"].SerializeToString()
        key = digest(data)
        if key in seen:
            continue
        seen.add(key)
        result.append((data, {
            "name": row["name"], "family": "lowcost_generic_transfer",
            "detail": row.get("proof", "generic low-cost pattern"), "sha256": key,
        }))
    return result


class Worker:
    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self.tasks = ELIGIBLE[worker_id::3]
        self.task_set = set(self.tasks)
        self.cases = {task: SUPPORT.known_cases(task) for task in self.tasks}
        self.seen: dict[int, set[str]] = {task: set() for task in self.tasks}
        self.survivors: dict[int, list[tuple[dict[str, Any], bytes]]] = {
            task: [] for task in self.tasks
        }
        self.counters: Counter[str] = Counter()
        self.task_rows: dict[int, dict[str, Any]] = {
            task: {
                "task": task, "authority_cost": COSTS[task],
                "changed_since_8011_05": task in CHANGED_FROM_8011_05,
                "known_counts": counts, "families": {}, "screen_survivors": [],
            }
            for task, (_cases, counts) in self.cases.items()
        }

    def consider(self, task: int, data: bytes, meta: dict[str, Any]) -> None:
        self.counters["candidate_encounters"] += 1
        key = digest(data)
        if key in self.seen[task]:
            self.counters["duplicate_sha"] += 1
            return
        self.seen[task].add(key)
        family = str(meta.get("family", "history_archive"))
        family_row = self.task_rows[task]["families"].setdefault(
            family, {"encounters": 0, "unique": 0, "known_policy90": 0, "strict_lower": 0}
        )
        family_row["encounters"] += 1
        family_row["unique"] += 1
        try:
            model = onnx.load_model_from_string(data)
        except Exception:
            self.counters["parse_reject"] += 1
            return
        if parameter_count(model) >= COSTS[task]:
            self.counters["parameter_lower_bound_reject"] += 1
            return
        reasons = quick_preflight(model)
        if reasons:
            self.counters["preflight_reject"] += 1
            return
        cases, _counts = self.cases[task]
        known = failfast_known(data, cases)
        if not runtime_pass(known) or known.get("early_reject_reason") is not None:
            self.counters["known_policy90_reject"] += 1
            return
        family_row["known_policy90"] += 1
        profile = POLICY.fast_profile(SUPPORT, task, model, cases[0])
        if profile is None or int(profile["cost"]) >= COSTS[task]:
            self.counters["actual_cost_reject"] += 1
            return
        structure = POLICY.structure_audit(SUPPORT, task, model, data)
        if not structure["pass"]:
            self.counters["structure_reject"] += 1
            return
        family_row["strict_lower"] += 1
        item = {
            **{key_: value for key_, value in meta.items() if not key_.startswith("_")},
            "sha256": key, "task": task, "authority_cost": COSTS[task],
            "candidate_cost": int(profile["cost"]), "profile": profile,
            "known_disable_threads1": known, "structure": structure,
        }
        self.survivors[task].append((item, data))
        self.task_rows[task]["screen_survivors"].append({
            key_: value for key_, value in item.items() if key_ != "structure"
        })
        print(json.dumps({
            "worker": self.worker_id, "task": task, "family": family,
            "cost": profile["cost"], "known": known["accuracy"], "sha": key[:12],
        }), flush=True)

    def reprofile_authority(self) -> None:
        with zipfile.ZipFile(AUTHORITY) as archive:
            for task in self.tasks:
                data = archive.read(f"task{task:03d}.onnx")
                model = onnx.load_model_from_string(data)
                cases, _ = self.cases[task]
                profile = POLICY.fast_profile(SUPPORT, task, model, cases[0])
                if profile is None or int(profile["cost"]) != COSTS[task]:
                    raise RuntimeError(
                        f"task{task:03d} authority profile {profile} != {COSTS[task]}"
                    )
                self.task_rows[task]["authority_reprofile"] = {
                    "sha256": digest(data), "profile": profile,
                    "cost_matches_census": True,
                }
                self.seen[task].add(digest(data))

    def scan_history(self) -> None:
        paths = subprocess.check_output(
            ["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True
        ).splitlines()
        for relpath in paths:
            task = task_from_name(relpath)
            if task not in self.task_set:
                continue
            self.counters["history_loose_matching"] += 1
            try:
                data = (ROOT / relpath).read_bytes()
            except Exception:
                self.counters["history_read_error"] += 1
                continue
            self.consider(task, data, {
                "name": Path(relpath).name, "family": "history_loose_strict_lower",
                "detail": "workspace loose-model history rebase", "source": relpath,
            })
        zips = subprocess.check_output(
            ["rg", "--files", "-g", "*.zip"], cwd=ROOT, text=True
        ).splitlines()
        for index, relzip in enumerate(zips, 1):
            try:
                with zipfile.ZipFile(ROOT / relzip) as archive:
                    for member in archive.namelist():
                        if not member.lower().endswith(".onnx"):
                            continue
                        task = task_from_name(member)
                        if task not in self.task_set:
                            continue
                        self.counters["history_zip_matching"] += 1
                        try:
                            data = archive.read(member)
                        except Exception:
                            self.counters["history_read_error"] += 1
                            continue
                        self.consider(task, data, {
                            "name": Path(member).name, "family": "history_zip_strict_lower",
                            "detail": "ZIP-member history rebase",
                            "source": f"{relzip}!{member}",
                        })
            except Exception:
                self.counters["history_zip_open_error"] += 1
            if index % 100 == 0:
                print(json.dumps({
                    "worker": self.worker_id, "zip_progress": index,
                    "survivors": sum(len(rows) for rows in self.survivors.values()),
                }), flush=True)

    def scan_current_simplifiers(self) -> None:
        with zipfile.ZipFile(AUTHORITY) as archive:
            for task in self.tasks:
                print(json.dumps({
                    "worker": self.worker_id, "phase": "exact_simplifier", "task": task
                }), flush=True)
                base = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
                for data, meta in exact_simplifier_variants(task, base):
                    self.consider(task, data, meta)

    def scan_lowcost(self) -> None:
        generic = build_generic_lowcost()
        self.counters["lowcost_generic_models"] = len(generic)
        with zipfile.ZipFile(AUTHORITY) as archive:
            for task in self.tasks:
                print(json.dumps({
                    "worker": self.worker_id, "phase": "lowcost_transfer", "task": task
                }), flush=True)
                base = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
                cases, _ = self.cases[task]
                for data, meta in lowcost_variants(task, base, cases, generic):
                    self.consider(task, data, meta)

    def full_audit(self) -> list[dict[str, Any]]:
        task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
        finalists: list[dict[str, Any]] = []
        candidates_dir = HERE / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        for task in self.tasks:
            rows = self.survivors[task]
            # Prefer lowest actual cost, then higher complete-known accuracy.
            rows.sort(key=lambda pair: (
                int(pair[0]["candidate_cost"]),
                -float(pair[0]["known_disable_threads1"]["accuracy"]),
                pair[0]["sha256"],
            ))
            attempts = []
            for rank, (item, data) in enumerate(rows[:8], 1):
                known_cases, _ = self.cases[task]
                known_four_raw = SUPPORT.evaluate_four(data, known_cases)
                known_four = {
                    name: compact_runtime(row) for name, row in known_four_raw.items()
                }
                known_pass = all(runtime_pass(row) for row in known_four_raw.values())
                fresh = []
                if known_pass:
                    for seed in (
                        408_100_000 + self.worker_id * 10_000 + task,
                        408_200_000 + self.worker_id * 10_000 + task,
                    ):
                        fresh_cases, generation = SUPPORT.fresh_cases(task, seed, task_map)
                        runtime_raw = SUPPORT.evaluate_four(data, fresh_cases)
                        fresh.append({
                            "seed": seed, "generation": generation,
                            "runtime": {
                                name: compact_runtime(row) for name, row in runtime_raw.items()
                            },
                            "pass": all(runtime_pass(row) for row in runtime_raw.values()),
                        })
                passed = bool(known_pass and len(fresh) == 2 and all(row["pass"] for row in fresh))
                audited = {
                    **item, "rank": rank, "known_four": known_four,
                    "known_four_pass": known_pass, "fresh": fresh,
                    "policy90_pass": passed,
                }
                attempts.append(audited)
                print(json.dumps({
                    "worker": self.worker_id, "full_task": task, "rank": rank,
                    "cost": item["candidate_cost"], "known_pass": known_pass,
                    "fresh": [run["runtime"]["disable_threads1"]["accuracy"] for run in fresh],
                    "pass": passed,
                }), flush=True)
                if passed:
                    path = candidates_dir / (
                        f"task{task:03d}_POLICY90_cost{item['candidate_cost']}_{item['sha256'][:12]}.onnx"
                    )
                    path.write_bytes(data)
                    audited["saved_path"] = rel(path)
                    audited["score_gain"] = math.log(COSTS[task] / int(item["candidate_cost"]))
                    finalists.append(audited)
                    break
            self.task_rows[task]["full_audit_attempts"] = attempts
            self.task_rows[task]["admission"] = next(
                (row for row in finalists if int(row["task"]) == task), None
            )
        return finalists

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        self.reprofile_authority()
        self.scan_history()
        self.scan_current_simplifiers()
        self.scan_lowcost()
        finalists = self.full_audit()
        return {
            "lane": rel(HERE), "worker": self.worker_id, "pid": os.getpid(),
            "authority": rel(AUTHORITY), "authority_sha256": AUTHORITY_SHA256,
            "threshold": THRESHOLD, "fresh_per_seed": FRESH_PER_SEED,
            "assigned_tasks": list(self.tasks), "assigned_count": len(self.tasks),
            "band": [{"task": task, "cost": cost} for task, cost in BAND],
            "excluded_band_tasks": [task for task, _ in BAND if task in PRIVATE_ZERO_OR_UNSOUND],
            "private_zero_or_unsound_catalogue": sorted(PRIVATE_ZERO_OR_UNSOUND),
            "explicit_latest_lb_black": sorted(EXPLICIT_LATEST_LB_BLACK),
            "task_rows": [self.task_rows[task] for task in self.tasks],
            "finalists": finalists,
            "counters": dict(self.counters),
            "elapsed_seconds": time.monotonic() - started,
            "protected_writes": "lane only; root submission/all_scores/others untouched",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("8012.15 authority SHA mismatch")
    payload = Worker(args.worker).run()
    output = HERE / f"worker_{args.worker}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker, "tasks": payload["assigned_tasks"],
        "finalists": [
            {"task": row["task"], "cost": row["candidate_cost"], "gain": row["score_gain"]}
            for row in payload["finalists"]
        ], "elapsed": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
