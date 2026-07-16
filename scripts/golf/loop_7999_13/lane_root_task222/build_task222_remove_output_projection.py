#!/usr/bin/env python3
"""Remove task222's unnecessary signed output-color projection."""

from pathlib import Path
import zipfile

import onnx


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ZIP = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"
OUTPUT = Path(__file__).resolve().parent / "task222_remove_output_projection.onnx"


def main() -> None:
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        model = onnx.load_model_from_string(archive.read("task222.onnx"))
    node = model.graph.node[0]
    if node.op_type != "Einsum" or node.input[-1] != "P":
        raise RuntimeError("unexpected task222 graph")
    attribute = next(item for item in node.attribute if item.name == "equation")
    equation = attribute.s.decode("ascii")
    expected = ",ok->borc"
    if not equation.endswith(expected):
        raise RuntimeError(f"unexpected equation ending: {equation}")
    attribute.s = (equation[: -len(expected)] + "->bkrc").encode("ascii")
    del node.input[-1]
    kept = [item for item in model.graph.initializer if item.name != "P"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
