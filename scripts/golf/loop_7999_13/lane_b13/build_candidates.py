from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "scripts" / "golf" / "loop_7999_13" / "lane_b13"
ARCHIVE_254 = (
    ROOT
    / "scripts"
    / "golf"
    / "loop_7999_13"
    / "lane_archive_all400"
    / "task254_r01_static42.onnx"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def build_task254() -> tuple[Path, dict[str, object]]:
    """Precontract the constant-only subnet of the generator-derived formula.

    The archive witness is one giant tensor contraction.  Its three independent
    logical factors are retained exactly, but all constant-only indices are
    eliminated offline.  The resulting direct expression has 15 operands, no
    counted intermediate, and four ordinary finite float32 initializers.
    """
    source = onnx.load(ARCHIVE_254)
    arrays = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in source.graph.initializer}
    v = arrays["V"]
    ccoef = arrays["Ccoef"]
    s0 = arrays["S0"]
    s1 = arrays["S1"]
    acoef = arrays["A"]
    bcoef = arrays["B"]

    # Original indices:
    #   g,gA,gA,gB,ge -> ABe
    #   h,hC,hD,hE,hf -> CDEf
    #   s,se,sf,sj,sj,PZ,Pb,eX,efY,Xb,Yb -> jb
    tg = np.einsum("g,gA,gA,gB,ge->ABe", ccoef, s0, s0, s1, s1)
    th = np.einsum("h,hC,hD,hE,hf->CDEf", ccoef, s0, s0, s1, s1)
    q = np.einsum(
        "s,se,sf,sj,sj,PZ,Pb,eX,efY,Xb,Yb->jb",
        ccoef,
        s0,
        s0,
        s1,
        s1,
        v,
        v,
        acoef,
        bcoef,
        v,
        v,
    )

    inp = helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])
    equation = (
        "akmn,Ak,almd,Bl,ABe,"
        "aopu,Co,aqrd,Dq,atru,Et,CDEf,"
        "aicd,ji,jb->abcd"
    )
    node = helper.make_node(
        "Einsum",
        [
            "input",
            "V",
            "input",
            "V",
            "TG",
            "input",
            "V",
            "input",
            "V",
            "input",
            "V",
            "TH",
            "input",
            "V",
            "Q",
        ],
        ["output"],
        equation=equation,
    )
    graph = helper.make_graph(
        [node],
        "task254_precontracted_standard",
        [inp],
        [out],
        [_tensor("V", v), _tensor("TG", tg), _tensor("TH", th), _tensor("Q", q)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    path = LANE / "candidates" / "task254_precontract15.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    meta: dict[str, object] = {
        "task": 254,
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "source_witness": str(ARCHIVE_254.relative_to(ROOT)),
        "source_witness_sha256": _sha256(ARCHIVE_254),
        "equation": equation,
        "einsum_operands": len(node.input),
        "initializer_elements": int(sum(np.prod(numpy_helper.to_array(t).shape) for t in model.graph.initializer)),
        "initializer_shapes": {t.name: list(numpy_helper.to_array(t).shape) for t in model.graph.initializer},
        "finite_initializers": bool(
            all(np.isfinite(numpy_helper.to_array(t)).all() for t in model.graph.initializer)
        ),
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
    }
    return path, meta


def build_task267_control() -> tuple[Path, dict[str, object]]:
    """Rebuild the known safe cost-60 direct expression as a floor control."""
    v = np.zeros((30,), dtype=np.float32)
    v[0] = -1.0
    v[6] = 1.0
    w = np.zeros((30,), dtype=np.float32)
    w[0] = -4.0
    w[1:6] = 1.0
    w[6] = -1.0
    inp = helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])
    equation = "borc,r,bdst,s,bdhw->bohw"
    node = helper.make_node(
        "Einsum", ["input", "RU", "input", "RV", "input"], ["output"], equation=equation
    )
    graph = helper.make_graph(
        [node],
        "task267_safe_floor_control",
        [inp],
        [out],
        [_tensor("RU", v), _tensor("RV", w)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path = LANE / "controls" / "task267_safe_cost60.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    meta: dict[str, object] = {
        "task": 267,
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "equation": equation,
        "einsum_operands": len(node.input),
        "initializer_elements": 60,
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "role": "non-winning parity/floor control",
    }
    return path, meta


def main() -> int:
    LANE.mkdir(parents=True, exist_ok=True)
    _, task254 = build_task254()
    _, task267 = build_task267_control()
    task254["role"] = "rejected algebraic attempt; e/f coupling was lost"
    task254["known_gate"] = "failed 0/265 in both ORT modes"
    manifest = {
        "immutable_base": "submission_base_7999.13.zip",
        "rejected_attempts": [task254],
        "controls": [task267],
    }
    (LANE / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
