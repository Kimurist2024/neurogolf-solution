#!/usr/bin/env python3
"""Competition-profile every unique historical SHA in the lane inventory."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINES = {187: 1798, 191: 3436, 319: 1003}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
CANDIDATES = HERE / "history_candidates"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HARVEST = load_module(
    "lane157_history_harvest",
    ROOT / "scripts/golf/loop_7999_13/lane_harvest/harvest.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bytes(source: str) -> bytes:
    if "::" in source:
        archive_name, member = source.split("::", 1)
        with zipfile.ZipFile(ROOT / archive_name) as archive:
            return archive.read(member)
    return (ROOT / source).read_bytes()


def canonical_semantic_bytes(model: onnx.ModelProto) -> bytes:
    """Normalize metadata/VI and unused initializers, never computation."""
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
    used = {name for node in value.graph.node for name in node.input if name}
    normalized = []
    for item in value.graph.initializer:
        if item.name not in used:
            continue
        normalized.append(numpy_helper.from_array(np.asarray(numpy_helper.to_array(item)), item.name))
    normalized.sort(key=lambda item: item.name)
    del value.graph.initializer[:]
    value.graph.initializer.extend(normalized)
    for node in value.graph.node:
        node.name = ""
        node.doc_string = ""
    imports = sorted(value.opset_import, key=lambda item: (item.domain, item.version))
    del value.opset_import[:]
    value.opset_import.extend(imports)
    return value.SerializeToString(deterministic=True)


def conv_ub0(model: onnx.ModelProto) -> dict[str, object]:
    initializers = {item.name: item for item in model.graph.initializer}
    findings: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            weight_index, bias_index = 1, 2
            weight = initializers.get(node.input[weight_index]) if len(node.input) > weight_index else None
            expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        elif node.op_type == "ConvTranspose":
            weight_index, bias_index = 1, 2
            weight = initializers.get(node.input[weight_index]) if len(node.input) > weight_index else None
            group = next((int(attr.i) for attr in node.attribute if attr.name == "group"), 1)
            expected = int(weight.dims[1] * group) if weight is not None and len(weight.dims) > 1 else None
        elif node.op_type == "QLinearConv":
            weight_index, bias_index = 3, 8
            weight = initializers.get(node.input[weight_index]) if len(node.input) > weight_index else None
            expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        else:
            continue
        if len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = initializers.get(node.input[bias_index])
        count = int(math.prod(bias.dims)) if bias is not None and bias.dims else None
        safe = bias is not None and len(bias.dims) == 1 and count == expected
        findings.append(
            {
                "node": node.name or (node.output[0] if node.output else ""),
                "op": node.op_type,
                "bias": node.input[bias_index],
                "expected": expected,
                "count": count,
                "safe": safe,
            }
        )
    return {"pass": all(bool(item["safe"]) for item in findings), "findings": findings}


def structure(model: onnx.ModelProto) -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_checker=False, full_checker_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}")
    custom_domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}}
        | {node.domain for node in model.graph.node if node.domain not in {"", "ai.onnx"}}
    )
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    nested = sum(
        attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        for node in model.graph.node
        for attr in node.attribute
    )
    ub = conv_ub0(model)
    result.update(
        custom_domains=custom_domains,
        banned_ops=banned,
        nested_graph_attributes=nested,
        function_count=len(model.functions),
        sparse_initializer_count=len(model.graph.sparse_initializer),
        conv_ub0=ub,
    )
    result["pass"] = bool(
        result.get("full_checker")
        and result.get("strict_data_prop")
        and not custom_domains
        and not banned
        and nested == 0
        and not model.functions
        and not model.graph.sparse_initializer
        and ub["pass"]
    )
    return result


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((HERE / "history_inventory.json").read_text())
    authority_canonical: dict[int, str] = {}
    authority_data: dict[int, bytes] = {}
    for task in BASELINES:
        authority_sha = inventory["tasks"][str(task)]["authority_member_sha256"]
        record = next(
            row
            for row in inventory["tasks"][str(task)]["records"]
            if row["sha256"] == authority_sha
        )
        data = source_bytes(record["sources"][0])
        authority_data[task] = data
        authority_canonical[task] = digest(
            canonical_semantic_bytes(onnx.load_model_from_string(data))
        )

    output: dict[str, object] = {
        "authority": inventory["authority"],
        "authority_sha256": inventory["authority_sha256"],
        "inventory_counts": inventory["counts"],
        "tasks": {},
    }
    for task in sorted(BASELINES):
        rows: list[dict[str, object]] = []
        for record in inventory["tasks"][str(task)]["records"]:
            row: dict[str, object] = {
                "sha256": record["sha256"],
                "serialized_bytes": record["serialized_bytes"],
                "sources": record["sources"],
                "source_count": record["source_count"],
                "is_authority": record["is_authority"],
            }
            try:
                data = source_bytes(record["sources"][0])
                if digest(data) != record["sha256"]:
                    raise RuntimeError("source SHA drift")
                model = onnx.load_model_from_string(data)
                row["node_count"] = len(model.graph.node)
                row["initializer_count"] = len(model.graph.initializer)
                row["op_histogram"] = dict(Counter(node.op_type for node in model.graph.node))
                row["canonical_semantic_sha256"] = digest(canonical_semantic_bytes(model))
                row["same_canonical_graph_as_authority"] = (
                    row["canonical_semantic_sha256"] == authority_canonical[task]
                )
                row["structural"] = structure(model)
                if row["structural"]["pass"]:
                    actual = HARVEST.actual_screen(data, task)
                    row["actual_cost"] = actual
                    row["strict_lower_actual"] = actual is not None and actual < BASELINES[task]
                    if row["strict_lower_actual"]:
                        path = CANDIDATES / f"task{task:03d}_actual{actual}_{record['sha256'][:12]}.onnx"
                        path.write_bytes(data)
                        row["path"] = str(path.relative_to(ROOT))
                else:
                    row["actual_cost"] = None
                    row["strict_lower_actual"] = False
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["actual_cost"] = None
                row["strict_lower_actual"] = False
            rows.append(row)
        output["tasks"][str(task)] = {
            "baseline_actual_cost": BASELINES[task],
            "authority_member_sha256": inventory["tasks"][str(task)]["authority_member_sha256"],
            "authority_canonical_semantic_sha256": authority_canonical[task],
            "unique_sha_count": len(rows),
            "structural_pass_count": sum(bool(row.get("structural", {}).get("pass")) for row in rows),
            "actual_profiled_count": sum(row.get("actual_cost") is not None for row in rows),
            "strict_lower_count": sum(bool(row.get("strict_lower_actual")) for row in rows),
            "strict_lower_exact_authority_count": sum(
                bool(row.get("strict_lower_actual") and row.get("same_canonical_graph_as_authority"))
                for row in rows
            ),
            "rows": rows,
        }
        (HERE / "history_screen.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "task": task,
                    "unique": len(rows),
                    "structural": output["tasks"][str(task)]["structural_pass_count"],
                    "actual": output["tasks"][str(task)]["actual_profiled_count"],
                    "strict_lower": output["tasks"][str(task)]["strict_lower_count"],
                    "strict_lower_exact": output["tasks"][str(task)]["strict_lower_exact_authority_count"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
