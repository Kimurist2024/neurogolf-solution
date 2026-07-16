from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "scripts/golf/loop_7999_13/lane_b14"
BASE_ZIP = ROOT / "submission_base_7999.13.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def model_row(task: int, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    equations = []
    for node in model.graph.node:
        for attr in node.attribute:
            if node.op_type == "Einsum" and attr.name == "equation":
                value = helper.get_attribute_value(attr)
                equations.append(value.decode() if isinstance(value, bytes) else str(value))
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path.read_bytes()),
        "file_bytes": path.stat().st_size,
        "node_count": len(model.graph.node),
        "value_info_count": len(model.graph.value_info),
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "nodes": [
            {
                "index": index,
                "name": node.name,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "input_count": len(node.input),
            }
            for index, node in enumerate(model.graph.node)
        ],
        "equations": equations,
        "initializers": [
            {
                "name": item.name,
                "dtype": str(numpy_helper.to_array(item).dtype),
                "shape": list(numpy_helper.to_array(item).shape),
                "elements": int(numpy_helper.to_array(item).size),
            }
            for item in model.graph.initializer
        ],
        "value_info": [
            {"name": item.name, "dtype": item.type.tensor_type.elem_type, "shape": dims(item)}
            for item in model.graph.value_info
        ],
        "inferred_value_info": [
            {"name": item.name, "dtype": item.type.tensor_type.elem_type, "shape": dims(item)}
            for item in inferred.graph.value_info
        ],
    }


def main() -> int:
    (LANE / "baseline").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in (5, 80):
            name = f"task{task:03d}.onnx"
            payload = archive.read(name)
            (LANE / "baseline" / name).write_bytes(payload)
    rows = {
        str(task): model_row(task, LANE / "baseline" / f"task{task:03d}.onnx")
        for task in (5, 80)
    }
    output = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha256(BASE_ZIP.read_bytes()),
        "models": rows,
    }
    (LANE / "baseline_structure.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({task: {k: row[k] for k in ("sha256", "node_count", "value_info_count")} for task, row in rows.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
