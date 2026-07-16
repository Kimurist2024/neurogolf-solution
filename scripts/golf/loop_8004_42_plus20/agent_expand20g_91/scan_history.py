#!/usr/bin/env python3
"""SHA-deduplicate every repository ZIP member and loose ONNX for 20 tasks.

This is discovery only.  It uses a conservative static cost lower bound to
retain every graph that could be cheaper than the 8006.61 authority.  Runtime,
truthfulness, known×4 and actual cost are handled by audit_leads.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TARGETS = (66, 46, 117, 270, 165, 310, 238, 156, 35, 69, 354, 19, 237, 378, 368, 284, 363, 34, 89, 125)
ZIP_MEMBER = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)
LOOSE_TASK = re.compile(r"task(\d{3})", re.IGNORECASE)
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

sys.path.insert(0, str(ROOT))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def static_cost_floor(model: onnx.ModelProto) -> dict[str, int] | None:
    """A safe under-estimate: false declarations/constants can only raise actual cost."""
    try:
        inferred = shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=False, data_prop=True
        )
    except Exception:
        inferred = model
    infos = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    free = {value.name for value in inferred.graph.input}
    free.update(value.name for value in inferred.graph.output)
    free.update(item.name for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in free or name in seen:
                continue
            seen.add(name)
            value = infos.get(name)
            dims = shape(value) if value is not None else None
            if dims is None:
                return None
            try:
                dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            except Exception:
                return None
            memory += math.prod(dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) if item.dims else 1 for item in inferred.graph.initializer)
    params += sum(
        math.prod(item.values.dims) if item.values.dims else 1
        for item in inferred.graph.sparse_initializer
    )
    # Constant attributes are deliberately included when statically visible.
    for node in inferred.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "sparse_value":
                params += math.prod(attr.sparse_tensor.values.dims) if attr.sparse_tensor.values.dims else 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    ops = Counter(node.op_type for node in model.graph.node)
    checker_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    strict_error = None
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        strict = True
    except Exception as exc:  # noqa: BLE001
        strict = False
        strict_error = f"{type(exc).__name__}: {exc}"
    domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
        | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    return {
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_data_prop": strict,
        "strict_error": strict_error,
        "standard_domain": not domains,
        "custom_domains": domains,
        "banned_ops": sorted(
            {
                node.op_type
                for node in model.graph.node
                if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
            }
        ),
        "lookup_or_cloak_ops": sorted(
            {node.op_type for node in model.graph.node if node.op_type.upper() in LOOKUP_CLOAK}
        ),
        "max_einsum_inputs": max_einsum,
        "giant_einsum": max_einsum > 16,
        "nested_graphs": any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attr in node.attribute
        ),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "op_histogram": dict(sorted(ops.items())),
    }


def canonical_semantic_bytes(model: onnx.ModelProto) -> bytes:
    """Strong exact class: computation/used constants unchanged; metadata ignored."""
    value = copy.deepcopy(model)
    value.producer_name = ""
    value.producer_version = ""
    value.domain = ""
    value.model_version = 0
    value.doc_string = ""
    del value.metadata_props[:]
    value.graph.name = ""
    value.graph.doc_string = ""
    del value.graph.value_info[:]
    # Unused dense initializers have no mathematical effect.
    used = {name for node in value.graph.node for name in node.input if name}
    normalized_initializers = []
    for item in value.graph.initializer:
        if item.name not in used:
            continue
        array = np.asarray(numpy_helper.to_array(item))
        normalized_initializers.append(numpy_helper.from_array(array, item.name))
    normalized_initializers.sort(key=lambda item: item.name)
    del value.graph.initializer[:]
    value.graph.initializer.extend(normalized_initializers)
    for node in value.graph.node:
        node.name = ""
        node.doc_string = ""
    imports = sorted(value.opset_import, key=lambda item: (item.domain, item.version))
    del value.opset_import[:]
    value.opset_import.extend(imports)
    return value.SerializeToString(deterministic=True)


def add_observation(
    unique: dict[int, dict[str, dict[str, Any]]],
    task: int,
    data: bytes,
    source: str,
    authority_hashes: dict[int, str],
    counts: Counter[str],
) -> None:
    sha = digest(data)
    if sha == authority_hashes[task]:
        counts["authority_duplicates"] += 1
        return
    if sha in unique[task]:
        entry = unique[task][sha]
        entry["source_count"] += 1
        if len(entry["sources"]) < 30:
            entry["sources"].append(source)
        counts["sha_duplicates"] += 1
        return
    unique[task][sha] = {
        "task": task,
        "sha256": sha,
        "data": data,
        "bytes": len(data),
        "sources": [source],
        "source_count": 1,
    }


def zip_paths() -> list[Path]:
    excluded = {".git", ".venv", "node_modules", HERE.name}
    return sorted(
        path
        for path in ROOT.rglob("*.zip")
        if not any(part in excluded for part in path.parts)
        and "others/71403" not in str(path.relative_to(ROOT))
    )


def loose_paths() -> list[Path]:
    excluded = {".git", ".venv", "node_modules", HERE.name}
    return sorted(
        path
        for path in ROOT.rglob("*.onnx")
        if not any(part in excluded for part in path.parts)
        and "others/71403" not in str(path.relative_to(ROOT))
    )


def main() -> int:
    actual_authority_sha256 = digest(AUTHORITY.read_bytes())
    if actual_authority_sha256 != AUTHORITY_SHA256:
        raise RuntimeError(
            f"authority drift: {AUTHORITY} is {actual_authority_sha256}, "
            f"expected {AUTHORITY_SHA256}"
        )
    authority_cost_payload = json.loads((HERE / "evidence/authority_costs.json").read_text())
    authority_costs = {
        int(task): int(cost)
        for task, cost in authority_cost_payload["costs"].items()
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = {task: archive.read(f"task{task:03d}.onnx") for task in TARGETS}
    authority_hashes = {task: digest(data) for task, data in authority_data.items()}
    authority_semantic = {
        task: digest(canonical_semantic_bytes(onnx.load_model_from_string(data)))
        for task, data in authority_data.items()
    }
    unique: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    zips = zip_paths()
    for index, path in enumerate(zips, start=1):
        counts["zip_files_seen"] += 1
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    match = ZIP_MEMBER.search(member)
                    if not match:
                        continue
                    task = int(match.group(1))
                    if task not in TARGETS:
                        continue
                    counts["zip_target_members_seen"] += 1
                    add_observation(
                        unique,
                        task,
                        archive.read(member),
                        f"{path.relative_to(ROOT)}::{member}",
                        authority_hashes,
                        counts,
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append({"zip": str(path.relative_to(ROOT)), "error": repr(exc)})
        if index % 100 == 0:
            print(
                f"zip {index}/{len(zips)} unique={sum(len(rows) for rows in unique.values())}",
                flush=True,
            )
    loose = loose_paths()
    for index, path in enumerate(loose, start=1):
        matches = [int(match.group(1)) for match in LOOSE_TASK.finditer(path.name)]
        tasks = sorted(set(matches) & set(TARGETS))
        if not tasks:
            continue
        try:
            data = path.read_bytes()
            for task in tasks:
                counts["loose_target_files_seen"] += 1
                add_observation(
                    unique,
                    task,
                    data,
                    str(path.relative_to(ROOT)),
                    authority_hashes,
                    counts,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append({"loose": str(path.relative_to(ROOT)), "error": repr(exc)})
        if index % 20000 == 0:
            print(
                f"loose {index}/{len(loose)} unique={sum(len(rows) for rows in unique.values())}",
                flush=True,
            )
    counts["unique_non_authority"] = sum(len(rows) for rows in unique.values())
    candidate_dir = HERE / "candidates/history_prefilter"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    retained_total = 0
    for task in TARGETS:
        rows: list[dict[str, Any]] = []
        for entry in unique[task].values():
            row = {key: value for key, value in entry.items() if key != "data"}
            try:
                model = onnx.load_model_from_string(entry["data"])
                floor = static_cost_floor(model)
                row["static_cost_floor"] = floor
                row["structure"] = structure(model)
                row["semantic_sha256"] = digest(canonical_semantic_bytes(model))
                row["exact_computational_graph_equivalent"] = (
                    row["semantic_sha256"] == authority_semantic[task]
                )
                could_be_lower = floor is None or int(floor["cost"]) < authority_costs[task]
                row["could_be_actual_strict_lower"] = could_be_lower
                if could_be_lower:
                    retained_total += 1
                    path = candidate_dir / (
                        f"task{task:03d}_h{retained_total:04d}_{entry['sha256'][:12]}.onnx"
                    )
                    path.write_bytes(entry["data"])
                    row["candidate_path"] = str(path.relative_to(ROOT))
            except Exception as exc:  # noqa: BLE001
                row["parse_or_scan_error"] = f"{type(exc).__name__}: {exc}"
                row["could_be_actual_strict_lower"] = False
            rows.append(row)
        rows.sort(
            key=lambda row: (
                0 if row.get("could_be_actual_strict_lower") else 1,
                int((row.get("static_cost_floor") or {}).get("cost", 10**18)),
                str(row["sha256"]),
            )
        )
        rows_by_task[str(task)] = rows
        summary.append(
            {
                "task": task,
                "authority_cost": authority_costs[task],
                "authority_sha256": authority_hashes[task],
                "unique_non_authority": len(rows),
                "retained_for_actual_audit": sum(
                    bool(row.get("could_be_actual_strict_lower")) for row in rows
                ),
                "strong_exact_class": sum(
                    bool(row.get("exact_computational_graph_equivalent")) for row in rows
                ),
            }
        )
    result = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": actual_authority_sha256,
        "targets": list(TARGETS),
        "excluded_paths": ["others/71403", ".git", ".venv", "node_modules", str(HERE.relative_to(ROOT))],
        "inventory": {
            "filesystem_zip_count": len(zips),
            "filesystem_loose_onnx_count": len(loose),
            "counts": dict(counts),
            "errors": errors,
        },
        "summary": summary,
        "rows_by_task": rows_by_task,
        "retained_for_actual_audit": retained_total,
    }
    (HERE / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"retained_for_actual_audit={retained_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
