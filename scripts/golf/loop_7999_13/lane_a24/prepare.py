#!/usr/bin/env python3
"""Prepare the isolated A24 task198/task277 evidence lane."""

from __future__ import annotations

import collections
import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_7999.13.zip"
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json"
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
EXPECTED_TASK_SHA256 = {
    198: "4e37cca3fc86cd4781a9b1f55c080f13962273e803c4c45d6dda99f74ba95283",
    277: "a6d659f65b084bdbcd7e2cc287a6fc0901a0351863ac64f7b47a5420e251f71d",
}
RULE_REFERENCES = {
    "task277_behavioral_1256": ROOT / "others/1/70201/task277_cost1256_improved.onnx",
    "task277_component_mass": ROOT / "scripts/golf/scratch_claude/task277/mass_q.onnx",
    "task277_component_width": ROOT / "scripts/golf/scratch_claude/task277/width.onnx",
    "task198_generator_runtime_basis": ROOT / "scripts/golf/scratch_claude/task198/cand_runtime_basis.onnx",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def summary(path: Path, task: int, label: str) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    ops = collections.Counter(node.op_type for node in model.graph.node)
    return {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path.read_bytes()),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "params": sum(int(numpy_helper.to_array(item).size) for item in model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "ops": dict(ops),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "declared_inputs": {item.name: shape(item) for item in model.graph.input},
        "declared_outputs": {item.name: shape(item) for item in model.graph.output},
    }


def main() -> None:
    base_sha = sha(BASE_ZIP.read_bytes())
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"wrong base archive: {base_sha}")
    for name in ("baseline", "history", "rule_references"):
        (HERE / name).mkdir(parents=True, exist_ok=True)

    models: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in (198, 277):
            data = archive.read(f"task{task:03d}.onnx")
            actual = sha(data)
            if actual != EXPECTED_TASK_SHA256[task]:
                raise RuntimeError(f"wrong task{task:03d} lineage: {actual}")
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(data)
            models.append(summary(path, task, f"task{task:03d}_base"))

    # Exact executable rewrite: correct only the two false intermediate shape
    # declarations.  Nodes, attributes, tensors, and graph I/O stay byte-for-
    # byte identical after value_info is normalized away.
    truthful_label = "task277_exact_truthful"
    base277 = onnx.load(HERE / "baseline/task277.onnx", load_external_data=False)
    false_shape_names = {"g", "u"}
    for item in base277.graph.value_info:
        if item.name in false_shape_names:
            for dim, size in zip(item.type.tensor_type.shape.dim, (1, 10, 30, 30)):
                dim.dim_value = size
    truthful_path = HERE / "rule_references" / f"{truthful_label}.onnx"
    onnx.save(base277, truthful_path)
    models.append(summary(truthful_path, 277, truthful_label))

    inventory = json.loads(INVENTORY.read_text())
    history: dict[str, object] = {}
    for task in (198, 277):
        for index, entry in enumerate(inventory["retained"][str(task)], 1):
            label = f"task{task:03d}_r{index:02d}"
            source = ROOT / entry["path"]
            path = HERE / "history" / f"{label}.onnx"
            path.write_bytes(source.read_bytes())
            if sha(path.read_bytes()) != entry["sha256"]:
                raise RuntimeError(f"history hash mismatch: {label}")
            models.append(summary(path, task, label))
            history[label] = entry

    references: dict[str, object] = {
        truthful_label: {
            "task": 277,
            "source": "exact task277 from submission_base_7999.13.zip",
            "sha256": sha(truthful_path.read_bytes()),
            "rewrite": "corrected false value_info declarations g/u only",
            "corrected_value_info": sorted(false_shape_names),
            "nodes_initializers_graph_io_unchanged": True,
        }
    }
    for label, source in RULE_REFERENCES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        task = int(label[4:7])
        path = HERE / "rule_references" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        models.append(summary(path, task, label))
        references[label] = {
            "task": task,
            "source": str(source.relative_to(ROOT)),
            "sha256": sha(path.read_bytes()),
        }

    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "lane": "a24",
                "base_score": 7999.13,
                "base_archive": str(BASE_ZIP.relative_to(ROOT)),
                "base_archive_sha256": base_sha,
                "expected_task_sha256": EXPECTED_TASK_SHA256,
                "models": models,
                "history_entries": history,
                "rule_references": references,
                "admission_policy": (
                    "private-black/ambiguous: only truthful generator-rule SOUND rebuild "
                    "or exact bitwise-equivalent rewrite; archive known/fresh alone is insufficient"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
