#!/usr/bin/env python3
"""Scan all 400 authority models for exact inference/default simplifications.

Families:
1. Inference-mode Dropout with unused mask -> Identity.
2. Clip scalar -inf/+inf optional bounds -> omitted optional inputs.
3. Explicit Reshape allowzero=0 -> active schema default.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATES = HERE / "candidates"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HARVEST = load_module(
    "inference_defaults_165_harvest",
    ROOT / "scripts/golf/loop_7999_13/lane_harvest/harvest.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def constant_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            try:
                if attribute.name == "value":
                    values[node.output[0]] = np.asarray(numpy_helper.to_array(attribute.t))
                elif attribute.name == "value_float":
                    values[node.output[0]] = np.asarray(attribute.f, dtype=np.float32)
                elif attribute.name == "value_int":
                    values[node.output[0]] = np.asarray(attribute.i, dtype=np.int64)
                elif attribute.name == "value_floats":
                    values[node.output[0]] = np.asarray(attribute.floats, dtype=np.float32)
                elif attribute.name == "value_ints":
                    values[node.output[0]] = np.asarray(attribute.ints, dtype=np.int64)
            except Exception:  # noqa: BLE001
                continue
    return values


def active_schema(model: onnx.ModelProto, node: onnx.NodeProto) -> onnx.defs.OpSchema:
    domain = node.domain or ""
    versions = {item.domain or "": int(item.version) for item in model.opset_import}
    return onnx.defs.get_schema(node.op_type, versions.get(domain, 1), domain)


def discover(model: onnx.ModelProto) -> tuple[list[dict[str, Any]], dict[str, int]]:
    values = constant_values(model)
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_outputs = {value.name for value in model.graph.output}
    sites: list[dict[str, Any]] = []
    counts = Counter(node.op_type for node in model.graph.node)
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Dropout":
            training_name = node.input[2] if len(node.input) > 2 else ""
            training = values.get(training_name) if training_name else None
            inference = not training_name or (
                training is not None
                and training.size == 1
                and not bool(training.reshape(-1)[0])
            )
            mask = node.output[1] if len(node.output) > 1 else ""
            mask_unused = not mask or (uses[mask] == 0 and mask not in graph_outputs)
            if inference and mask_unused:
                sites.append(
                    {
                        "kind": "dropout_inference_identity",
                        "node_index": index,
                        "training_mode": "absent" if not training_name else training_name,
                        "mask_output": mask,
                    }
                )

        if node.op_type == "Clip":
            for position, infinity, kind in (
                (1, -math.inf, "clip_omit_negative_infinity_min"),
                (2, math.inf, "clip_omit_positive_infinity_max"),
            ):
                name = node.input[position] if len(node.input) > position else ""
                value = values.get(name) if name else None
                if value is None or value.size != 1:
                    continue
                scalar = value.reshape(-1)[0]
                if (infinity < 0 and np.isneginf(scalar)) or (
                    infinity > 0 and np.isposinf(scalar)
                ):
                    sites.append(
                        {
                            "kind": kind,
                            "node_index": index,
                            "input_position": position,
                            "bound": name,
                        }
                    )

        if node.op_type == "Reshape":
            try:
                schema = active_schema(model, node)
                definition = schema.attributes.get("allowzero")
                schema_default = (
                    helper.get_attribute_value(definition.default_value)
                    if definition is not None and definition.default_value.name
                    else None
                )
            except Exception:  # noqa: BLE001
                schema_default = None
            for attribute_index, attribute in enumerate(node.attribute):
                if (
                    attribute.name == "allowzero"
                    and helper.get_attribute_value(attribute) == 0
                    and schema_default == 0
                ):
                    sites.append(
                        {
                            "kind": "reshape_drop_default_allowzero_0",
                            "node_index": index,
                            "attribute_index": attribute_index,
                            "schema_since_version": int(schema.since_version),
                        }
                    )
    return sites, dict(counts)


def drop_unused_initializers(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    removed = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return removed


def apply_sites(
    original: onnx.ModelProto, sites: list[dict[str, Any]]
) -> tuple[onnx.ModelProto, list[str]]:
    model = copy.deepcopy(original)
    by_node: dict[int, list[dict[str, Any]]] = {}
    for site in sites:
        by_node.setdefault(int(site["node_index"]), []).append(site)
    for node_index, chosen in by_node.items():
        node = model.graph.node[node_index]
        for site in chosen:
            kind = str(site["kind"])
            if kind == "dropout_inference_identity":
                replacement = helper.make_node(
                    "Identity",
                    [node.input[0]],
                    [node.output[0]],
                    name=node.name,
                )
                node.CopyFrom(replacement)
            elif kind.startswith("clip_omit_"):
                position = int(site["input_position"])
                while len(node.input) <= position:
                    node.input.append("")
                node.input[position] = ""
                while len(node.input) > 1 and not node.input[-1]:
                    del node.input[-1]
            elif kind == "reshape_drop_default_allowzero_0":
                keep = [attribute for attribute in node.attribute if attribute.name != "allowzero"]
                del node.attribute[:]
                node.attribute.extend(keep)
            else:
                raise RuntimeError(kind)
    return model, drop_unused_initializers(model)


def static_profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"defaults165_static_{task:03d}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def structure(model: onnx.ModelProto) -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_checker=False, full_checker_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}")
    return result


def variants(sites: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    if not sites:
        return []
    rows = [(f"single_{index}", [site]) for index, site in enumerate(sites)]
    if len(sites) > 1:
        rows.append(("combined_all", sites))
    return rows


def main() -> None:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA:
        raise RuntimeError("authority SHA drift")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        if len(members) != 400:
            raise RuntimeError(f"expected 400 ONNX members, got {len(members)}")
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            data = archive.read(member)
            model = onnx.load_model_from_string(data)
            sites, op_histogram = discover(model)
            inventory.append(
                {
                    "task": task,
                    "member": member,
                    "sha256": digest(data),
                    "node_count": len(model.graph.node),
                    "initializer_count": len(model.graph.initializer),
                    "dropout_count": op_histogram.get("Dropout", 0),
                    "clip_count": op_histogram.get("Clip", 0),
                    "reshape_count": op_histogram.get("Reshape", 0),
                    "eligible_sites": sites,
                }
            )
            if not sites:
                continue
            base_actual = HARVEST.actual_screen(data, task)
            base_static = static_profile(model, task)
            for label, chosen in variants(sites):
                candidate, removed_initializers = apply_sites(model, chosen)
                candidate_data = candidate.SerializeToString()
                row: dict[str, object] = {
                    "task": task,
                    "variant": label,
                    "sites": chosen,
                    "removed_initializers": removed_initializers,
                    "baseline_actual_cost": base_actual,
                    "baseline_static_profile": base_static,
                    "sha256": digest(candidate_data),
                    "node_count": len(candidate.graph.node),
                    "initializer_count": len(candidate.graph.initializer),
                }
                row.update(structure(candidate))
                if row.get("full_checker") and row.get("strict_data_prop"):
                    row["candidate_static_profile"] = static_profile(candidate, task)
                    actual = HARVEST.actual_screen(candidate_data, task)
                    row["candidate_actual_cost"] = actual
                    row["strict_lower_actual"] = (
                        actual is not None
                        and base_actual is not None
                        and actual < base_actual
                    )
                    if row["strict_lower_actual"]:
                        path = CANDIDATES / f"task{task:03d}_{label}_actual{actual}.onnx"
                        path.write_bytes(candidate_data)
                        row["path"] = str(path.relative_to(ROOT))
                else:
                    row["candidate_actual_cost"] = None
                    row["strict_lower_actual"] = False
                candidate_rows.append(row)

    root_scan = ROOT / "scripts/golf/loop_8004_42_plus20/root_schema_default_scan_164/scan.json"
    output = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA,
        "member_count": len(inventory),
        "proofs": {
            "dropout_inference_identity": "With training_mode absent or scalar false, ONNX Dropout is inference mode and its first output equals data. If the optional mask is unused, Identity preserves every observable output for every input.",
            "clip_infinite_optional_bounds": "Clip's omitted min/max are unbounded. Replacing scalar -infinity min and +infinity max with omitted optional bounds preserves finite, infinite, and NaN inputs.",
            "reshape_default_allowzero_0": "The active Reshape schema default is allowzero=0; deleting an explicit value 0 does not change semantics.",
            "unused_initializer_removal": "An initializer with no node input use and no graph output exposure cannot affect an observable output.",
        },
        "root_schema_default_scan_164": {
            "path": str(root_scan.relative_to(ROOT)),
            "sha256": digest(root_scan.read_bytes()),
            "reported_profiles": 1171,
            "reported_strict_lower": 0,
        },
        "inventory": inventory,
        "candidates": candidate_rows,
        "summary": {
            "tasks_scanned": len(inventory),
            "dropout_nodes": sum(int(row["dropout_count"]) for row in inventory),
            "eligible_dropout_sites": sum(
                site["kind"] == "dropout_inference_identity"
                for row in inventory
                for site in row["eligible_sites"]
            ),
            "clip_nodes": sum(int(row["clip_count"]) for row in inventory),
            "eligible_clip_sites": sum(
                str(site["kind"]).startswith("clip_omit_")
                for row in inventory
                for site in row["eligible_sites"]
            ),
            "reshape_nodes": sum(int(row["reshape_count"]) for row in inventory),
            "eligible_reshape_default_sites": sum(
                site["kind"] == "reshape_drop_default_allowzero_0"
                for row in inventory
                for site in row["eligible_sites"]
            ),
            "candidate_variants": len(candidate_rows),
            "strict_lower_actual": sum(
                bool(row.get("strict_lower_actual")) for row in candidate_rows
            ),
        },
    }
    (HERE / "scan.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], indent=2))
    for row in candidate_rows:
        print(
            row["task"],
            row["variant"],
            row["baseline_actual_cost"],
            row["candidate_actual_cost"],
            row["strict_lower_actual"],
        )


if __name__ == "__main__":
    main()
