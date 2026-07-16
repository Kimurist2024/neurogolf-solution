#!/usr/bin/env python3
"""Build exact tensor-network gauge reductions of the LB-white task175 net."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8014.69.zip"
AUTHORITY_SHA256 = "a5a811393b5b378c3bfe1e9aef29680b8af1671440aa21e900fe8c05ad54c328"
AUTHORITY_MEMBER_SHA256 = "b6404486ccc1a74c36bab6031f11c54c7326f787a743f02dff77e63c782af343"
CANDIDATES = HERE / "candidates"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name)


def load_authority() -> tuple[onnx.ModelProto, bytes]:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority archive drift")
    with zipfile.ZipFile(AUTHORITY) as archive:
        blob = archive.read("task175.onnx")
    if sha256(blob) != AUTHORITY_MEMBER_SHA256:
        raise RuntimeError("task175 authority member drift")
    return onnx.load_model_from_string(blob), blob


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if set(result) != {"Q", "S", "R", "Msel", "TA", "TB", "W", "V"}:
        raise RuntimeError(f"unexpected initializer set: {sorted(result)}")
    return result


def make_model(initializers: list[onnx.TensorProto], equation: str, inputs: list[str], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Einsum", inputs, ["output"], equation=equation, name="output")],
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=10,
        producer_name="codex-task175-exact-gauge",
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def gauge_remove_w_v(source: dict[str, np.ndarray]) -> onnx.ModelProto:
    """Absorb W into both shared TB occurrences and turn V into implicit sum.

    For T'=X T Y, choosing YX=W preserves the internal bridge.  The boundary
    tensors become TA'=TA X^-1 and V'=Y^-1 V.  The chosen gauge makes V'=[1,1],
    so its contraction is the implicit sum over the now-unpaired D index.
    """
    weight = source["W"].astype(np.float32)
    a = np.float32(weight[1, 0])
    x = np.asarray([[1.0, 0.0], [1.0 - a, 1.0]], dtype=np.float32)
    x_inv = np.asarray([[1.0, 0.0], [a - 1.0, 1.0]], dtype=np.float32)
    y = np.asarray([[1.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    if not np.array_equal(y @ x, weight):
        raise RuntimeError("float32 gauge does not reconstruct W exactly")
    tb = np.einsum("la,agb,br->lgr", x, source["TB"], y, optimize=False).astype(np.float32)
    ta = (source["TA"] @ x_inv).astype(np.float32)
    v_prime = np.linalg.solve(y.astype(np.float64), source["V"].astype(np.float64)).astype(np.float32)
    if not np.array_equal(v_prime, np.ones(2, dtype=np.float32)):
        raise RuntimeError(f"gauge boundary is not implicit-sum ones: {v_prime}")

    names = [
        "S", "R", "S", "Q", "Q", "input",
        "Msel", "R", "S", "Q", "Q", "input",
        "TA", "TB", "R", "S", "Q", "Q", "input",
        "TB", "R", "S", "S", "Q", "input", "S", "Q", "input",
    ]
    equation = (
        "Ps,soa,ou,ut,ai,bihw,Pv,ved,ex,xt,dj,bjwh,"
        "Pl,lgr,gfc,fy,yt,ck,bkzz,rmD,mAZ,BZ,Ap,pM,bMhn,Bq,qN,bNwn->bthw"
    )
    initializers = [
        tensor("Q", source["Q"].copy()),
        tensor("S", source["S"].copy()),
        tensor("R", source["R"].copy()),
        tensor("Msel", source["Msel"].copy()),
        tensor("TA", ta),
        tensor("TB", tb),
    ]
    return make_model(initializers, equation, names, "task175_gauge_remove_w_v")


def gauge_remove_w_v_cp4(source: dict[str, np.ndarray]) -> onnx.ModelProto:
    """Experimental best rank-4 CP approximation of R (four fewer params).

    This is deliberately kept separate from the exact gauge candidate: rank 4
    cannot reproduce the authority R exactly, so it must clear the same gold
    and fresh gates before it can be considered.
    """
    base = gauge_remove_w_v(source)
    packed = arrays_without_expected_set(base)
    ra = np.asarray([
        [0.18568872, 0.51574415, -2.8073765e-12, -2.8615093],
        [-0.0019232124, 0.027282387, 0.70710677, -0.041857686],
        [0.0019232124, -0.027282387, 0.70710677, 0.041857686],
        [5.9745126, -6.2869334, 3.1814898e-10, 0.652959],
    ], dtype=np.float32)
    rb = np.asarray([
        [-4.674226, -6.229141, -1.0, -2.8438098],
        [3.6760302, 0.99381936, 9.3047875e-11, 0.69967997],
        [-0.6063824, 0.055992737, -3.5840922e-12, -0.203196],
    ], dtype=np.float32)
    rc = np.asarray([
        [-2.4765248, -4.413147, -0.70710677, -1.9756294],
        [4.795194, 0.91433716, 1.6167113e-10, 0.8723336],
        [-0.68487227, -0.072587155, -1.1044059e-10, -0.22562204],
        [2.4765248, 4.413147, -0.70710677, 1.9756294],
    ], dtype=np.float32)
    names = [
        "S", "RA", "RB", "RC", "S", "Q", "Q", "input",
        "Msel", "RA", "RB", "RC", "S", "Q", "Q", "input",
        "TA", "TB", "RA", "RB", "RC", "S", "Q", "Q", "input",
        "TB", "RA", "RB", "RC", "S", "S", "Q", "input", "S", "Q", "input",
    ]
    equation = (
        "Ps,sK,oK,aK,ou,ut,ai,bihw,Pv,vJ,eJ,dJ,ex,xt,dj,bjwh,"
        "Pl,lgr,gH,fH,cH,fy,yt,ck,bkzz,rmD,mI,AI,ZI,BZ,Ap,pM,bMhn,Bq,qN,bNwn->bthw"
    )
    initializers = [
        tensor("Q", packed["Q"].copy()), tensor("S", packed["S"].copy()),
        tensor("Msel", packed["Msel"].copy()), tensor("TA", packed["TA"].copy()),
        tensor("TB", packed["TB"].copy()), tensor("RA", ra), tensor("RB", rb), tensor("RC", rc),
    ]
    return make_model(initializers, equation, names, "task175_gauge_remove_w_v_cp4")


def arrays_without_expected_set(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def main() -> None:
    authority, blob = load_authority()
    source = arrays(authority)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    candidate = CANDIDATES / "task175_gauge_remove_w_v.onnx"
    candidate.write_bytes(gauge_remove_w_v(source).SerializeToString())
    memory, params, cost = cost_of(str(candidate))
    cp4 = CANDIDATES / "task175_gauge_remove_w_v_cp4.onnx"
    cp4.write_bytes(gauge_remove_w_v_cp4(source).SerializeToString())
    cp4_memory, cp4_params, cp4_cost = cost_of(str(cp4))
    result = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "authority_member_sha256": sha256(blob),
        "authority_profile": {"memory": 0, "params": 140, "cost": 140},
        "candidates": {
            "gauge_remove_w_v": {
                "path": str(candidate.relative_to(ROOT)),
                "sha256": sha256(candidate.read_bytes()),
                "profile": {"memory": memory, "params": params, "cost": cost},
            },
            "gauge_remove_w_v_cp4": {
                "path": str(cp4.relative_to(ROOT)),
                "sha256": sha256(cp4.read_bytes()),
                "profile": {"memory": cp4_memory, "params": cp4_params, "cost": cp4_cost},
            }
        },
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
