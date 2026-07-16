#!/usr/bin/env python3
"""Extract immutable 8009.46 targets and audit exact mechanical rewrites."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import onnx
import numpy as np
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8009.46.zip"
BASE_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
MEMBER_SHA256 = {
    101: "ad535c519c18681700f956262c48ca5990f15ff4b58fd94de95ef3beff69a84b",
    133: "6c5dc3a593b0900e16966b9d4c40af509a34c1dd1f0264c31cd30eaf9b4570e5",
}
BASE_COST = {101: 5655, 133: 4393}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task101_broadcast_safe_expand(base: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    """Replace `And(all_true[1,1,3,6], scalar)` by a shape-preserving Expand.

    The generic no-op scanner's scalar substitution is not shape preserving and
    is intentionally not used as a candidate.  Expand broadcasts the same bool
    scalar to the exact original `[1,1,3,6]` result shape.
    """
    model = onnx.ModelProto()
    model.CopyFrom(base)
    nodes = list(model.graph.node)
    target = next(node for node in nodes if "tail3_dyn" in node.output)
    if target.op_type != "And" or set(target.input) != {"tail3x6", "is_wide_bool"}:
        raise RuntimeError("unexpected task101 tail gate")
    tail = next(item for item in model.graph.initializer if item.name == "tail3x6")
    array = numpy_helper.to_array(tail)
    if array.shape != (1, 1, 3, 6) or not np.all(array):
        raise RuntimeError("task101 tail3x6 is not the proved all-true broadcast tensor")
    shape_name = "tail3x6_shape"
    shape = numpy_helper.from_array(np.asarray(array.shape, dtype=np.int64), shape_name)
    replacement = helper.make_node(
        "Expand", ["is_wide_bool", shape_name], ["tail3_dyn"], name="tail3_dyn_exact_expand"
    )
    index = nodes.index(target)
    nodes[index] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    keep = [item for item in model.graph.initializer if item.name != "tail3x6"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    model.graph.initializer.append(shape)
    return model, {
        "proof": "For boolean b, And(ones([1,1,3,6]), b) = Expand(b, [1,1,3,6]) elementwise.",
        "removed_initializer": {"name": "tail3x6", "elements": 18},
        "added_initializer": {"name": shape_name, "elements": 4},
        "output_shape_preserved": [1, 1, 3, 6],
    }


def task101_broadcast_safe_resize(base: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    """A smaller exact form: nearest-resize the scalar bool over axes 2/3."""
    model = onnx.ModelProto()
    model.CopyFrom(base)
    nodes = list(model.graph.node)
    target = next(node for node in nodes if "tail3_dyn" in node.output)
    if target.op_type != "And" or set(target.input) != {"tail3x6", "is_wide_bool"}:
        raise RuntimeError("unexpected task101 tail gate")
    tail = next(item for item in model.graph.initializer if item.name == "tail3x6")
    array = numpy_helper.to_array(tail)
    if array.shape != (1, 1, 3, 6) or not np.all(array):
        raise RuntimeError("task101 tail3x6 is not the proved all-true broadcast tensor")
    sizes_name = "tail_hw"
    sizes = numpy_helper.from_array(np.asarray([3, 6], dtype=np.int64), sizes_name)
    replacement = helper.make_node(
        "Resize",
        ["is_wide_bool", "", "", sizes_name],
        ["tail3_dyn"],
        name="tail3_dyn_exact_resize",
        axes=[2, 3],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    nodes[nodes.index(target)] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    keep = [item for item in model.graph.initializer if item.name != "tail3x6"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    model.graph.initializer.append(sizes)
    return model, {
        "proof": "Nearest Resize of a scalar bool to axes[2,3]=[3,6] replicates that scalar exactly; this equals And(all_true[1,1,3,6], scalar).",
        "removed_initializer": {"name": "tail3x6", "elements": 18},
        "added_initializer": {"name": sizes_name, "elements": 2},
        "output_shape_preserved": [1, 1, 3, 6],
    }


def main() -> None:
    if sha256(BASE_ZIP.read_bytes()) != BASE_ZIP_SHA256:
        raise RuntimeError("immutable authority zip hash mismatch")
    baseline = HERE / "baseline"
    candidates = HERE / "candidates"
    baseline.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task, expected in MEMBER_SHA256.items():
            data = archive.read(f"task{task:03d}.onnx")
            if sha256(data) != expected:
                raise RuntimeError(f"task{task:03d} member hash mismatch")
            (baseline / f"task{task:03d}.onnx").write_bytes(data)

    scanner = load_module(
        "lane118_scanner",
        ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
    )
    auditor = load_module(
        "lane118_auditor",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    harvest = load_module(
        "lane118_harvest",
        ROOT / "scripts/golf/loop_7999_13/lane_harvest/harvest.py",
    )

    output: dict[str, object] = {
        "authority_zip": str(BASE_ZIP.relative_to(ROOT)),
        "authority_zip_sha256": BASE_ZIP_SHA256,
        "base_cost": BASE_COST,
        "tasks": {},
    }
    seen: set[tuple[int, str]] = set()
    for task in (101, 133):
        base_path = baseline / f"task{task:03d}.onnx"
        base = onnx.load(base_path)
        task_row: dict[str, object] = {
            "authority_member_sha256": MEMBER_SHA256[task],
            "base_audit": auditor.audit(f"base_task{task:03d}", task, base_path),
            "rewrites": [],
        }
        for kind in scanner.KINDS:
            model, detail = scanner.transform(base, kind)
            if detail["semantic_action_count"] == detail["metadata_action_count"] == 0:
                continue
            data = model.SerializeToString()
            digest = sha256(data)
            if digest == MEMBER_SHA256[task] or (task, digest) in seen:
                continue
            seen.add((task, digest))
            path = candidates / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
            path.write_bytes(data)
            actual_cost = harvest.actual_screen(data, task)
            row = {
                "kind": kind,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "actions": detail,
                "actual_cost": actual_cost,
                "strictly_lower": actual_cost is not None and actual_cost < BASE_COST[task],
                "audit": auditor.audit(f"task{task:03d}_{kind}", task, path),
            }
            task_row["rewrites"].append(row)
            print(task, kind, actual_cost, digest, flush=True)
        if task == 101:
            for label, builder in (
                ("exact_broadcast_expand", task101_broadcast_safe_expand),
                ("exact_broadcast_resize", task101_broadcast_safe_resize),
            ):
                model, detail = builder(base)
                data = model.SerializeToString()
                digest = sha256(data)
                path = candidates / f"task101_{label}_{digest[:12]}.onnx"
                path.write_bytes(data)
                actual_cost = harvest.actual_screen(data, task)
                row = {
                    "kind": label,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest,
                    "actions": detail,
                    "actual_cost": actual_cost,
                    "strictly_lower": actual_cost is not None and actual_cost < BASE_COST[task],
                    "audit": auditor.audit(f"task101_{label}", task, path),
                }
                task_row["rewrites"].append(row)
                print(task, label, actual_cost, digest, flush=True)
        output["tasks"][str(task)] = task_row
        (HERE / "mechanical_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
