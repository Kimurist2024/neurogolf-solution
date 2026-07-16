#!/usr/bin/env python3
"""Build exact-ArgMax task192 candidates from the LB-white rank-7 authority."""

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
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name)


def authority_model() -> tuple[onnx.ModelProto, bytes]:
    if sha256_bytes(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("8014.69 authority SHA drift")
    with zipfile.ZipFile(AUTHORITY) as archive:
        data = archive.read("task192.onnx")
    return onnx.load_model_from_string(data), data


def common_arrays(source: onnx.ModelProto) -> dict[str, np.ndarray]:
    arrays = {item.name: numpy_helper.to_array(item) for item in source.graph.initializer}
    expected = {"basis0", "hist_selector", "neighbor_map", "route_out", "eig_factor"}
    if set(arrays) != expected:
        raise RuntimeError(f"unexpected authority initializers: {sorted(arrays)}")
    return arrays


def exact_selector_nodes() -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    nodes = [
        helper.make_node(
            "ArgMax", ["hist"], ["selected_i64"], axis=1, keepdims=0,
            name="selected_i64",
        ),
        helper.make_node(
            "OneHot", ["selected_i64", "depth", "onehot_values"], ["selected"],
            axis=-1, name="selected",
        ),
    ]
    initializers = [
        tensor("depth", np.asarray(10, dtype=np.int64)),
        tensor("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]
    return nodes, initializers


def make_graph(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)],
        producer_name="codex-task192-gold-exact-selector", ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def build_control(source: onnx.ModelProto, arrays: dict[str, np.ndarray]) -> onnx.ModelProto:
    selector_nodes, selector_initializers = exact_selector_nodes()
    nodes = [
        helper.make_node(
            "Einsum", ["input", "basis0", "hist_selector"], ["hist"],
            equation="bchw,rc,xr->xc", name="hist",
        ),
        *selector_nodes,
        helper.make_node("Concat", ["basis0", "selected"], ["basis"], axis=0, name="basis"),
        helper.make_node(
            "Einsum",
            [
                "input", "basis0",
                "input", "basis", "neighbor_map", "eig_factor", "eig_factor",
                "input", "basis", "neighbor_map", "eig_factor", "eig_factor",
                "route_out", "basis",
            ],
            ["output"],
            equation=(
                "bchw,rc,bdhq,ld,rl,qt,wt,bepw,me,rm,ps,hs,rn,no->bohw"
            ),
            name="output",
        ),
    ]
    initializers = [
        tensor(name, arrays[name].copy())
        for name in ("basis0", "hist_selector", "neighbor_map", "route_out", "eig_factor")
    ] + selector_initializers
    return make_graph(nodes, initializers, "task192_rank7_exact_argmax_control")


def build_compact(source: onnx.ModelProto, arrays: dict[str, np.ndarray]) -> onnx.ModelProto:
    selector_nodes, selector_initializers = exact_selector_nodes()
    route = arrays["route_out"]
    # basis0 rows are I (inside) and N (nonzero).  The padded dynamic basis is
    # G=[I,S], where S is the exact selected-color one-hot.  Products
    # basis0[i,o]*G[j,o] span I, N and S, so this 2x2x2 tensor reconstructs
    # every authority route exactly without materializing [I,N,S].
    route_product = np.zeros((2, 2, 2), dtype=np.float32)
    route_product[:, 0, 0] = route[:, 0]
    route_product[:, 1, 0] = route[:, 1]
    route_product[:, 0, 1] = route[:, 2]
    nodes = [
        helper.make_node(
            "Einsum", ["input", "basis0", "hist_selector"], ["hist"],
            equation="bchw,rc,xr->xc", name="hist",
        ),
        *selector_nodes,
        helper.make_node(
            "Pad", ["selected", "neighbor_pads", "one"], ["neighbor_basis"],
            mode="constant", name="neighbor_basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "basis0",
                "input", "neighbor_basis", "eig_factor", "eig_factor",
                "input", "neighbor_basis", "eig_factor", "eig_factor",
                "route_product", "basis0", "neighbor_basis",
            ],
            ["output"],
            equation=(
                "bchw,rc,bdhq,rd,qt,wt,bepw,re,ps,hs,rij,io,jo->bohw"
            ),
            name="output",
        ),
    ]
    initializers = [
        tensor("basis0", arrays["basis0"].copy()),
        tensor("hist_selector", arrays["hist_selector"].copy()),
        tensor("eig_factor", arrays["eig_factor"].copy()),
        tensor("route_product", route_product),
        tensor("neighbor_pads", np.asarray([1, 0, 0, 0], dtype=np.int64)),
        tensor("one", np.asarray(1.0, dtype=np.float32)),
        *selector_initializers,
    ]
    return make_graph(nodes, initializers, "task192_rank7_exact_argmax_compact")


def build_authority_equivalent_compact(
    source: onnx.ModelProto, arrays: dict[str, np.ndarray]
) -> onnx.ModelProto:
    """Algebraic rewrite retaining the authority's exact HardSigmoid selector."""
    route = arrays["route_out"]
    route_product = np.zeros((2, 2, 2), dtype=np.float32)
    route_product[:, 0, 0] = route[:, 0]
    route_product[:, 1, 0] = route[:, 1]
    route_product[:, 0, 1] = route[:, 2]
    nodes = [
        helper.make_node(
            "Einsum", ["input", "basis0", "hist_selector"], ["hist"],
            equation="bchw,rc,xr->xc", name="hist",
        ),
        helper.make_node(
            "HardSigmoid", ["hist"], ["selected"], alpha=1.0, beta=-33.0,
            name="selected",
        ),
        helper.make_node(
            "Pad", ["selected", "neighbor_pads", "one"], ["neighbor_basis"],
            mode="constant", name="neighbor_basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "basis0",
                "input", "neighbor_basis", "eig_factor", "eig_factor",
                "input", "neighbor_basis", "eig_factor", "eig_factor",
                "route_product", "basis0", "neighbor_basis",
            ],
            ["output"],
            equation=(
                "bchw,rc,bdhq,rd,qt,wt,bepw,re,ps,hs,rij,io,jo->bohw"
            ),
            name="output",
        ),
    ]
    initializers = [
        tensor("basis0", arrays["basis0"].copy()),
        tensor("hist_selector", arrays["hist_selector"].copy()),
        tensor("eig_factor", arrays["eig_factor"].copy()),
        tensor("route_product", route_product),
        tensor("neighbor_pads", np.asarray([1, 0, 0, 0], dtype=np.int64)),
        tensor("one", np.asarray(1.0, dtype=np.float32)),
    ]
    return make_graph(nodes, initializers, "task192_authority_equivalent_compact")


def build_authority_equivalent_factored(
    source: onnx.ModelProto, arrays: dict[str, np.ndarray]
) -> onnx.ModelProto:
    """Same compact graph with the exactly proportional output routes shared."""
    route = arrays["route_out"]
    threshold = np.float32(-route[0, 2] / route[0, 0])
    path_coeff = np.asarray([route[0, 0], route[1, 0]], dtype=np.float32)
    route_product = np.asarray(
        [[1.0, -threshold], [-1.0, 0.0]], dtype=np.float32
    )
    reconstructed = np.zeros_like(route)
    reconstructed[:, 0] = path_coeff
    reconstructed[:, 1] = -path_coeff
    reconstructed[:, 2] = -threshold * path_coeff
    if not np.array_equal(reconstructed, route):
        raise RuntimeError("authority routes are not float32-exact proportional")
    nodes = [
        helper.make_node(
            "Einsum", ["input", "basis0", "hist_selector"], ["hist"],
            equation="bchw,rc,xr->xc", name="hist",
        ),
        helper.make_node(
            "HardSigmoid", ["hist"], ["selected"], alpha=1.0, beta=-33.0,
            name="selected",
        ),
        helper.make_node(
            "Pad", ["selected", "neighbor_pads", "one"], ["neighbor_basis"],
            mode="constant", name="neighbor_basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "basis0",
                "input", "neighbor_basis", "eig_factor", "eig_factor",
                "input", "neighbor_basis", "eig_factor", "eig_factor",
                "path_coeff", "route_product", "basis0", "neighbor_basis",
            ],
            ["output"],
            equation=(
                "bchw,rc,bdhq,rd,qt,wt,bepw,re,ps,hs,r,ij,io,jo->bohw"
            ),
            name="output",
        ),
    ]
    initializers = [
        tensor("basis0", arrays["basis0"].copy()),
        tensor("hist_selector", arrays["hist_selector"].copy()),
        tensor("eig_factor", arrays["eig_factor"].copy()),
        tensor("path_coeff", path_coeff),
        tensor("route_product", route_product),
        tensor("neighbor_pads", np.asarray([1, 0, 0, 0], dtype=np.int64)),
        tensor("one", np.asarray(1.0, dtype=np.float32)),
    ]
    return make_graph(nodes, initializers, "task192_authority_equivalent_factored")


def main() -> None:
    source, source_data = authority_model()
    arrays = common_arrays(source)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exact_argmax_control": (CANDIDATES / "task192_rank7_exact_argmax_control.onnx", build_control(source, arrays)),
        "exact_argmax_compact": (CANDIDATES / "task192_rank7_exact_argmax_compact.onnx", build_compact(source, arrays)),
        "authority_equivalent_compact": (
            CANDIDATES / "task192_authority_equivalent_compact.onnx",
            build_authority_equivalent_compact(source, arrays),
        ),
        "authority_equivalent_factored": (
            CANDIDATES / "task192_authority_equivalent_factored.onnx",
            build_authority_equivalent_factored(source, arrays),
        ),
    }
    result = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "authority_task192_sha256": sha256_bytes(source_data),
        "authority_profile": {"memory": 200, "params": 244, "cost": 444},
        "candidates": {},
    }
    for key, (path, model) in outputs.items():
        path.write_bytes(model.SerializeToString())
        result["candidates"][key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(path.read_bytes()),
            "profile": profile(path),
        }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
