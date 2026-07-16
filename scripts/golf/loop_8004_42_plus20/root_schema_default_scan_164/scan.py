#!/usr/bin/env python3
"""Remove attributes that exactly equal the active ONNX schema default."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"schemadefault164_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def normalized(value: Any) -> Any:
    if isinstance(value, onnx.TensorProto):
        array = np.asarray(numpy_helper.to_array(value))
        return (str(array.dtype), tuple(array.shape), array.tobytes())
    if isinstance(value, np.ndarray):
        return (str(value.dtype), tuple(value.shape), value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(normalized(item) for item in value)
    if isinstance(value, float) and math.isnan(value):
        return "__nan__"
    return value


def equal_value(left: onnx.AttributeProto, right: onnx.AttributeProto) -> bool:
    if left.type != right.type:
        return False
    try:
        return normalized(helper.get_attribute_value(left)) == normalized(
            helper.get_attribute_value(right)
        )
    except Exception:
        return False


def schema_for(model: onnx.ModelProto, node: onnx.NodeProto) -> onnx.defs.OpSchema:
    domain = node.domain or ""
    versions = {item.domain or "": int(item.version) for item in model.opset_import}
    return onnx.defs.get_schema(node.op_type, versions.get(domain, 1), domain)


def removable_sites(model: onnx.ModelProto) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.graph.node):
        try:
            schema = schema_for(model, node)
        except Exception:
            continue
        for attribute_index, attribute in enumerate(node.attribute):
            definition = schema.attributes.get(attribute.name)
            if definition is None or not definition.default_value.name:
                continue
            if equal_value(attribute, definition.default_value):
                rows.append({
                    "node_index": node_index,
                    "attribute_index": attribute_index,
                    "op_type": node.op_type,
                    "attribute": attribute.name,
                    "schema_since_version": int(schema.since_version),
                })
    return rows


def build(model: onnx.ModelProto, sites: list[dict[str, Any]]) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    by_node: dict[int, set[int]] = {}
    for site in sites:
        by_node.setdefault(int(site["node_index"]), set()).add(int(site["attribute_index"]))
    for node_index, drop in by_node.items():
        node = candidate.graph.node[node_index]
        keep = [attribute for index, attribute in enumerate(node.attribute) if index not in drop]
        del node.attribute[:]
        node.attribute.extend(keep)
    return candidate


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            sites = removable_sites(model)
            if not sites:
                continue
            variants = [("combined", sites)]
            if len(sites) > 1:
                variants.extend((f"single_{index}", [site]) for index, site in enumerate(sites))
            baseline = profile(model, task)
            for label, chosen in variants:
                candidate = build(model, chosen)
                row: dict[str, Any] = {
                    "task": task,
                    "variant": label,
                    "removed_count": len(chosen),
                    "sites": chosen,
                    "baseline": baseline,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    current = profile(candidate, task)
                    row["candidate"] = current
                    row["strict_lower"] = current["cost"] < baseline["cost"]
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{label}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "profiles": len(rows),
        "strict_lower_count": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": len([row for row in rows if "error" in row]),
    }, indent=2))


if __name__ == "__main__":
    main()
