#!/usr/bin/env python3
"""Cross-apply finite cost<=10 and generic low-cost structures to cost167..500."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
OUT = HERE / "transfer_candidates"
EVIDENCE = HERE / "transfer_evidence.json"
SAFE_LOW_TASKS = (16, 17, 61, 67, 87, 129, 140, 179, 197, 223, 241, 276, 305, 307, 309, 312, 337)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

import sys
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


BASE = import_path(
    "restart8012_transfer_base",
    ROOT / "scripts/golf/score25_similarity_le100_304/scan_score25.py",
)


def finite_structure(model: onnx.ModelProto) -> tuple[bool, str | None]:
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        if model.functions or model.graph.sparse_initializer:
            raise ValueError("functions or sparse initializers")
        for initializer in model.graph.initializer:
            array = onnx.numpy_helper.to_array(initializer)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"nonfinite initializer {initializer.name}")
        for node in model.graph.node:
            if node.op_type in BANNED or "Sequence" in node.op_type:
                raise ValueError(f"banned op {node.op_type}")
            for attr in node.attribute:
                if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                    raise ValueError("nested graph")
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def make_session(model: onnx.ModelProto, optimization: ort.GraphOptimizationLevel, threads: int) -> ort.InferenceSession | None:
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            return None
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        return ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return None


def evaluate(runtime: ort.InferenceSession | None, cases: list[dict[str, np.ndarray]], limit: int | None) -> dict[str, Any]:
    selected = cases if limit is None else cases[:limit]
    right = wrong = errors = nonfinite = smallpositive = shape_errors = 0
    for item in selected:
        try:
            if runtime is None:
                raise RuntimeError("no runtime")
            raw = runtime.run(["output"], {"input": item["input"]})[0]
            if raw.shape != item["output"].shape:
                shape_errors += 1
                wrong += 1
                continue
            if not np.isfinite(raw).all():
                nonfinite += 1
            if np.any((raw > 0.0) & (raw < 0.25)):
                smallpositive += 1
            if np.array_equal(raw > 0.0, item["output"] > 0.0):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    return {
        "total": len(selected), "right": right, "wrong": wrong, "errors": errors,
        "nonfinite": nonfinite, "smallpositive": smallpositive,
        "shape_errors": shape_errors,
        "accuracy": right / len(selected) if selected else 0.0,
    }


def clean(row: dict[str, Any]) -> bool:
    return not any(row[key] for key in ("errors", "nonfinite", "smallpositive", "shape_errors"))


def dedupe(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for source in sources:
        digest = hashlib.sha256(source["_data"]).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ok, error = finite_structure(source["_model"])
        if not ok:
            continue
        source = dict(source)
        source["sha256"] = digest
        source["structural_error"] = error
        result.append(source)
    return result


def main() -> int:
    started = time.monotonic()
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    authority = json.loads((HERE / "authority.json").read_text())
    scope = {int(row["task"]): int(row["cost"]) for row in authority["scope"]}
    known = {task: BASE.cases(task) for task in scope}
    OUT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(AUTHORITY) as archive:
        low_sources = []
        for task in SAFE_LOW_TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model_from_string(data)
            low_sources.append({
                "name": f"authority_safe_low_task{task:03d}",
                "family": "finite_authority_cost_le10",
                "source_task": task, "_model": model, "_data": data,
            })
        generic = dedupe([*BASE.TEMPLATES.generic_variants(), *low_sources])
        generic_sessions = [
            (source, make_session(source["_model"], ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1))
            for source in generic
        ]

        report: dict[str, Any] = {
            "authority": authority["authority"], "authority_sha256": AUTHORITY_SHA256,
            "scope_count": len(scope), "safe_low_source_tasks": list(SAFE_LOW_TASKS),
            "generic_unique_count": len(generic), "results": [], "finalists": [],
        }
        for task in sorted(scope, key=lambda value: (-scope[value], value)):
            incumbent = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            specific = dedupe(BASE.safe_specific_variants(task, incumbent))
            variants = generic_sessions + [
                (source, make_session(source["_model"], ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1))
                for source in specific
            ]
            row: dict[str, Any] = {
                "task": task, "authority_cost": scope[task],
                "known_count": len(known[task]), "variant_count": len(variants),
                "best_quick": {"accuracy": -1.0}, "survivors": [],
            }
            for source, runtime in variants:
                quick = evaluate(runtime, known[task], 20)
                if quick["accuracy"] > row["best_quick"]["accuracy"]:
                    row["best_quick"] = {
                        "name": source["name"], "family": source["family"], **quick,
                    }
                if not clean(quick) or quick["accuracy"] < 0.80:
                    continue
                full = evaluate(runtime, known[task], None)
                if not clean(full) or full["accuracy"] < 0.95:
                    continue
                model = source["_model"]
                try:
                    with tempfile.TemporaryDirectory(prefix=f"transfer406_{task:03d}_", dir="/tmp") as work:
                        profile = scoring.score_and_verify(
                            model, task, work, source["name"], require_correct=False
                        )
                except Exception as exc:  # noqa: BLE001
                    row["survivors"].append({
                        "name": source["name"], "known": full,
                        "profile_error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                if profile is None or int(profile["cost"]) >= scope[task]:
                    continue
                configs = []
                for name, optimization, threads in (
                    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
                    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
                    ("enable_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
                    ("enable_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
                ):
                    tested = evaluate(make_session(model, optimization, threads), known[task], None)
                    configs.append({"name": name, **tested})
                if not all(clean(value) and value["accuracy"] >= 0.95 for value in configs):
                    continue
                cost = int(profile["cost"])
                digest = hashlib.sha256(source["_data"]).hexdigest()
                path = OUT / f"task{task:03d}_cost{cost}_{digest[:12]}.onnx"
                path.write_bytes(source["_data"])
                item = {
                    "task": task, "authority_cost": scope[task], "candidate_cost": cost,
                    "half": 2 * cost <= scope[task], "name": source["name"],
                    "family": source["family"], "source_task": source.get("source_task"),
                    "sha256": digest, "known": full,
                    "known_exact": full["right"] == full["total"] and full["wrong"] == 0,
                    "known_four_configs": configs, "profile": profile,
                    "candidate_path": str(path.relative_to(ROOT)),
                }
                row["survivors"].append(item)
                report["finalists"].append(item)
            report["results"].append(row)
            print(json.dumps({
                "task": task, "cost": scope[task], "variants": len(variants),
                "best": row["best_quick"], "survivors": len(row["survivors"]),
            }), flush=True)

    report["finalists"].sort(key=lambda value: (
        int(value["task"]), int(value["candidate_cost"]),
        -float(value["known"]["accuracy"]), str(value["sha256"]),
    ))
    best = {}
    for value in report["finalists"]:
        best.setdefault(int(value["task"]), value)
    report["best_by_task"] = list(best.values())
    report["elapsed_seconds"] = time.monotonic() - started
    EVIDENCE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "finalist_count": len(report["finalists"]), "winner_tasks": sorted(best),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
