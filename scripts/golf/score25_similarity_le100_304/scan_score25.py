#!/usr/bin/env python3
"""Score-25-style exact pre-screen for every 8011.05 authority task at cost 51..100.

This lane is intentionally narrower than the parallel half-cost rebuild lane: it
tests output-only zero/one-parameter operator families and input-only Einsum
subexpressions related to the four currently LB-white score-25 members.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUTHORITY = ROOT / "submission_base_8011.05.zip"
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
    "score25_304_templates", ROOT / "scripts/golf/extra15_cost25_scan_294/scan.py"
)


def session(model: onnx.ModelProto) -> ort.InferenceSession | None:
    try:
        onnx.checker.check_model(model, full_check=True)
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            return None
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        return ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return None


def cases(task: int) -> list[dict[str, np.ndarray]]:
    data = scoring.load_examples(task)
    result = []
    for example in data["train"] + data["test"] + data["arc-gen"]:
        converted = scoring.convert_to_numpy(example)
        if converted:
            result.append(converted)
    return result


def evaluate(
    runtime: ort.InferenceSession,
    items: list[dict[str, np.ndarray]],
    limit: int | None,
) -> dict[str, Any]:
    right = wrong = errors = nonfinite = small = 0
    first_wrong = None
    selected = items if limit is None else items[:limit]
    for index, item in enumerate(selected):
        try:
            raw = runtime.run(["output"], {"input": item["input"]})[0]
            if not np.isfinite(raw).all():
                nonfinite += 1
            positives = np.abs(raw[np.abs(raw) > 0])
            if positives.size and bool(np.any(positives < 0.25)):
                small += 1
            ok = raw.shape == item["output"].shape and np.array_equal(
                raw > 0, item["output"] > 0
            )
            if ok:
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
        "total": len(selected), "right": right, "wrong": wrong, "errors": errors,
        "nonfinite_cases": nonfinite, "small_margin_cases": small,
        "first_wrong": first_wrong,
    }


def clean(row: dict[str, Any]) -> bool:
    return bool(
        row["right"] == row["total"]
        and row["wrong"] == row["errors"] == row["nonfinite_cases"]
        == row["small_margin_cases"] == 0
    )


def dedupe(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for source in sources:
        digest = hashlib.sha256(source["_data"]).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        # Sequence-family operators are forbidden by the competition policy.
        if any("Sequence" in node.op_type for node in source["_model"].graph.node):
            continue
        result.append(source)
    return result


def safe_specific_variants(task: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    """Use shared task variants, retaining input-only Einsum cuts if a legacy helper trips."""
    try:
        return TEMPLATES.task_specific_variants(task, base)
    except (IndexError, KeyError, ValueError):
        pass
    if not base.graph.node or base.graph.node[0].op_type != "Einsum":
        return []
    node = base.graph.node[0]
    equation = next(
        helper.get_attribute_value(attr)
        for attr in node.attribute if attr.name == "equation"
    ).decode()
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    input_terms = [term for term, name in zip(terms, node.input) if name == "input"]
    rows = []
    for mask in range(1, 1 << len(input_terms)):
        selected = [term for index, term in enumerate(input_terms) if mask & (1 << index)]
        if any(label not in set("".join(selected)) for label in rhs if label.isalpha()):
            continue
        candidate = TEMPLATES.make_model(
            helper.make_node(
                "Einsum", ["input"] * len(selected), ["output"],
                equation=",".join(selected) + "->" + rhs,
            )
        )
        data = candidate.SerializeToString()
        rows.append({
            "name": f"task{task:03d}_einsum_input_terms_{mask:x}",
            "family": "initializer_removal",
            "proof": "remove all learned operands and retain an input-only sub-equation",
            "_model": candidate, "_data": data,
        })
    return rows


def main() -> None:
    started = time.monotonic()
    HERE.mkdir(parents=True, exist_ok=True)
    costs = {}
    with (ROOT / "all_scores.csv").open() as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            if 51 <= cost <= 100 and float(row["score"]) < 24.9999:
                costs[task] = cost
    known = {task: cases(task) for task in costs}
    with zipfile.ZipFile(AUTHORITY) as archive:
        score25 = []
        for task in (67, 129, 179, 241):
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_from_string(data)
            score25.append({
                "name": f"authority_score25_task{task:03d}",
                "family": "authority_score25_template",
                "proof": "exact byte model from immutable 8011.05 authority",
                "_model": model, "_data": data,
            })
        generic = dedupe([*TEMPLATES.generic_variants(), *score25])
        generic_sessions = [(source, session(source["_model"])) for source in generic]
        report = {
            "authority": str(AUTHORITY.relative_to(ROOT)),
            "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
            "scope": sorted(costs),
            "generic_variant_count": len(generic),
            "results": [], "finalists": [],
        }
        for task in sorted(costs, key=lambda value: (costs[value], value)):
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_from_string(base_data)
            specific = dedupe(safe_specific_variants(task, base))
            variants = generic_sessions + [
                (source, session(source["_model"])) for source in specific
            ]
            row = {
                "task": task, "authority_cost": costs[task],
                "known_count": len(known[task]), "variant_count": len(variants),
                "best_quick": {"right": -1}, "quick_survivors": [],
                "full_survivors": [],
            }
            for source, runtime in variants:
                if runtime is None:
                    continue
                quick = evaluate(runtime, known[task], 12)
                if quick["right"] > row["best_quick"]["right"]:
                    row["best_quick"] = {
                        "name": source["name"], "family": source["family"], **quick,
                    }
                if not clean(quick):
                    continue
                summary = {
                    "name": source["name"], "family": source["family"],
                    "sha256": hashlib.sha256(source["_data"]).hexdigest(),
                    "quick": quick,
                }
                row["quick_survivors"].append(summary)
                full = evaluate(runtime, known[task], None)
                summary["full"] = full
                if not clean(full):
                    continue
                candidate = source["_model"]
                profile = scoring.score_and_verify(
                    candidate, task, str(HERE / "profiles"), source["name"],
                    require_correct=True,
                )
                summary["official_profile"] = profile
                if profile is None or profile["cost"] >= costs[task]:
                    continue
                stable, margin_min = scoring.model_margin_stable(candidate, task)
                summary["margin_stable"] = stable
                summary["margin_min"] = margin_min
                if not stable:
                    continue
                path = HERE / "candidates" / (
                    f"task{task:03d}_{source['name']}_cost{profile['cost']}.onnx"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source["_data"])
                summary["path"] = str(path.relative_to(ROOT))
                row["full_survivors"].append(summary)
                report["finalists"].append({"task": task, **summary})
            report["results"].append(row)
            print(json.dumps({
                "task": task, "cost": costs[task], "variants": len(variants),
                "best": row["best_quick"], "full": len(row["full_survivors"]),
            }), flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    (HERE / "score25_prescreen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "finalists": report["finalists"], "elapsed": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
