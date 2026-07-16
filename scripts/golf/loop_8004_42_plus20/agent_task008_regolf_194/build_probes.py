#!/usr/bin/env python3
"""Build isolated exact-local task008 probes from immutable 8009.46."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "submission_base_8009.46.zip"
ARCHIVE_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
AUTHORITY_SHA = "30abdd1f30f1aa88549edbf22c6e7a4af4fec3036fd8809812456ccb0df6e292"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [value for value in model.graph.initializer if value.name != name]
    if len(kept) == len(model.graph.initializer):
        raise ValueError(f"initializer not found: {name}")
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept)


def derived_scalar(base: onnx.ModelProto, target: str, left: str, right: str) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    remove_initializer(model, target)
    model.graph.node.insert(
        0,
        helper.make_node("Add", [left, right], [target], name=f"derive_{target}"),
    )
    return model


def output_truth_probe(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    output = model.graph.output[0]
    output.type.tensor_type.shape.ClearField("dim")
    for value in (1, 10, 30, 30):
        output.type.tensor_type.shape.dim.add().dim_value = value
    return model


def save(model: onnx.ModelProto, path: Path, require_valid: bool) -> dict[str, object]:
    checker_error = None
    strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        strict_error = f"{type(exc).__name__}: {exc}"
    if require_valid and (checker_error or strict_error):
        raise RuntimeError(f"invalid exact probe {path.name}: {checker_error}; {strict_error}")
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "checker_full": checker_error is None,
        "checker_error": checker_error,
        "strict_data_prop": strict_error is None,
        "strict_error": strict_error,
    }


def main() -> int:
    if digest(ARCHIVE.read_bytes()) != ARCHIVE_SHA:
        raise RuntimeError("8009.46 archive drift")
    with zipfile.ZipFile(ARCHIVE) as archive:
        authority_bytes = archive.read("task008.onnx")
    if digest(authority_bytes) != AUTHORITY_SHA:
        raise RuntimeError("task008 authority drift")
    authority = onnx.load_model_from_string(authority_bytes)
    rows = [save(authority, HERE / "authority/task008.onnx", True)]
    rows[-1]["label"] = "authority"

    probes = [
        ("derive_two", derived_scalar(authority, "two_i8", "one_i8", "one_i8"), True),
        ("derive_three", derived_scalar(authority, "three_i8", "one_i8", "two_i8"), True),
        ("derive_five", derived_scalar(authority, "five_i8", "two_i8", "three_i8"), True),
        ("truthful_output_only", output_truth_probe(authority), False),
    ]
    for label, model, require_valid in probes:
        row = save(model, HERE / "probes" / f"task008_{label}.onnx", require_valid)
        row["label"] = label
        rows.append(row)
    result = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": ARCHIVE_SHA,
        "authority_sha256": AUTHORITY_SHA,
        "rows": rows,
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
