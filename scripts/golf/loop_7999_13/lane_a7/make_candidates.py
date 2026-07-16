#!/usr/bin/env python3
"""Build isolated A7 candidates from the pinned 7999.13 members."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def save_checked(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model, full_check=True)
    path = HERE / "candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def task153_omit_zero_points() -> None:
    """Use the exact default zero points of QLinearConv."""
    model = onnx.load(HERE / "baseline" / "task153.onnx")
    graph = model.graph
    node = graph.node[28]
    assert node.op_type == "QLinearConv"
    assert list(node.input) == ["feat", "onef", "zu8", "W", "onef", "zi8", "onef", "zu8", "B"]
    del node.input[:]
    node.input.extend(["feat", "onef", "", "W", "onef", "", "onef", "", "B"])
    kept = [init for init in graph.initializer if init.name not in {"zu8", "zi8"}]
    assert len(kept) + 2 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task153_omit_zero_points.onnx")


def task071_cast_attribute() -> None:
    """Replace the correctly shaped CastLike dtype anchor with a Cast attr."""
    model = onnx.load(HERE / "baseline" / "task071.onnx")
    graph = model.graph
    node = graph.node[27]
    assert node.op_type == "CastLike" and list(node.input) == ["gather_u8", "i32zero"]
    node.op_type = "Cast"
    del node.input[:]
    node.input.extend(["gather_u8"])
    node.attribute.extend([onnx.helper.make_attribute("to", onnx.TensorProto.INT32)])
    kept = [init for init in graph.initializer if init.name != "i32zero"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task071_cast_attribute.onnx")


def task088_instance_norm_opset17() -> None:
    """Use the group=C GroupNorm/InstanceNorm identity and old axes attrs.

    All three GroupNormalization sites have one channel per group: the first
    two operate on C=10 with num_groups=10, while the ROI site operates on
    C=1 with num_groups=1.  InstanceNormalization has the same reduction axes
    in those cases.  Moving the three ReduceL1 axes to attributes then permits
    the model-wide ai.onnx opset to be lowered from 21 to 17.
    """
    model = onnx.load(HERE / "baseline" / "task088.onnx")
    graph = model.graph
    for index in (0, 19, 29):
        node = graph.node[index]
        assert node.op_type == "GroupNormalization"
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        assert attrs["stash_type"] == 1
        node.op_type = "InstanceNormalization"
        del node.attribute[:]
        node.attribute.extend([onnx.helper.make_attribute("epsilon", attrs["epsilon"])])

    for index, init_name, axes in (
        (3, "row_axes", [1, 3]),
        (4, "col_axes", [1, 2]),
        (22, "channel_axis", [1]),
    ):
        node = graph.node[index]
        assert node.op_type == "ReduceL1" and list(node.input) == [node.input[0], init_name]
        del node.input[1:]
        node.attribute.extend([onnx.helper.make_attribute("axes", axes)])

    removed = {"row_axes", "col_axes", "channel_axis"}
    kept = [init for init in graph.initializer if init.name not in removed]
    assert len(kept) + 3 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            assert opset.version == 21
            opset.version = 17
    save_checked(model, "task088_instance_norm_opset17.onnx")


def task055_low_rank_acoef(rank: int) -> None:
    """Factor the small-singular-tail Acoef tensor inside the terminal Einsum."""
    assert 1 <= rank <= 6
    model = onnx.load(HERE / "baseline" / "task055.onnx")
    graph = model.graph
    node = graph.node[9]
    assert node.op_type == "Einsum" and node.input[10] == "Acoef"
    equation_attr = next(attr for attr in node.attribute if attr.name == "equation")
    equation = onnx.helper.get_attribute_value(equation_attr).decode("ascii")
    assert "lgo" in equation and "Z" not in equation

    original = next(init for init in graph.initializer if init.name == "Acoef")
    array = numpy_helper.to_array(original).astype(np.float64)
    left, singular, right = np.linalg.svd(array.reshape(6, 9), full_matrices=False)
    # Split the singular values symmetrically to avoid placing the entire
    # condition number in either factor.  The stored values are ordinary f32.
    root = np.sqrt(singular[:rank])
    afac = (left[:, :rank] * root).astype(np.float32)
    bfac = (root[:, None] * right[:rank]).reshape(rank, 3, 3).astype(np.float32)
    kept = [init for init in graph.initializer if init.name != "Acoef"]
    assert len(kept) + 1 == len(graph.initializer)
    kept.extend([
        numpy_helper.from_array(afac, name="Acoef_lz"),
        numpy_helper.from_array(bfac, name="Acoef_zgo"),
    ])
    del graph.initializer[:]
    graph.initializer.extend(kept)

    inputs = list(node.input)
    inputs[10:11] = ["Acoef_lz", "Acoef_zgo"]
    del node.input[:]
    node.input.extend(inputs)
    equation_attr.s = equation.replace("lgo", "lZ,Zgo").encode("ascii")
    save_checked(model, f"task055_acoef_rank{rank}.onnx")


def task055_low_rank_acoef_side(axis: int, rank: int = 2) -> None:
    """Probe rank-2 compression of Acoef's g or o mode."""
    assert axis in (1, 2) and rank == 2
    model = onnx.load(HERE / "baseline" / "task055.onnx")
    graph = model.graph
    node = graph.node[9]
    equation_attr = next(attr for attr in node.attribute if attr.name == "equation")
    equation = onnx.helper.get_attribute_value(equation_attr).decode("ascii")
    original = next(init for init in graph.initializer if init.name == "Acoef")
    array = numpy_helper.to_array(original).astype(np.float64)
    matrix = np.moveaxis(array, axis, 0).reshape(3, 18)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    root = np.sqrt(singular[:rank])
    side = (left[:, :rank] * root).astype(np.float32)
    rest_moved = (root[:, None] * right[:rank]).reshape(rank, 6, 3)

    if axis == 1:
        # gZ * Zlo
        rest = rest_moved.astype(np.float32)
        suffix = "g"
        replacement = "gZ,Zlo"
    else:
        # The non-leading dimensions of the moved matrix are (l,g).
        rest = rest_moved.astype(np.float32)
        suffix = "o"
        replacement = "oZ,Zlg"
    kept = [init for init in graph.initializer if init.name != "Acoef"]
    kept.extend([
        numpy_helper.from_array(side, name=f"Acoef_{suffix}z"),
        numpy_helper.from_array(rest, name=f"Acoef_zl{'o' if axis == 1 else 'g'}"),
    ])
    del graph.initializer[:]
    graph.initializer.extend(kept)
    inputs = list(node.input)
    inputs[10:11] = [f"Acoef_{suffix}z", f"Acoef_zl{'o' if axis == 1 else 'g'}"]
    del node.input[:]
    node.input.extend(inputs)
    equation_attr.s = equation.replace("lgo", replacement).encode("ascii")
    save_checked(model, f"task055_acoef_{suffix}_rank{rank}.onnx")


def task055_drop_quadrature(index: int) -> None:
    """Drop one l quadrature point while preserving the original contraction."""
    assert 0 <= index < 6
    model = onnx.load(HERE / "baseline" / "task055.onnx")
    graph = model.graph
    arrays = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    replacements = {
        "Lpoly": np.delete(arrays["Lpoly"], index, axis=1),
        "Acoef": np.delete(arrays["Acoef"], index, axis=0),
    }
    kept = []
    for init in graph.initializer:
        if init.name in replacements:
            kept.append(numpy_helper.from_array(replacements[init.name], name=init.name))
        else:
            kept.append(init)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, f"task055_drop_l{index}.onnx")


def task088_einsum_row_col() -> None:
    """Remove two axes initializers when marker_h is already nonnegative."""
    model = onnx.load(HERE / "baseline" / "task088.onnx")
    graph = model.graph
    for index, init_name, equation in (
        (3, "row_axes", "nchw->nh"),
        (4, "col_axes", "nchw->nw"),
    ):
        node = graph.node[index]
        assert node.op_type == "ReduceL1" and list(node.input) == ["marker_h", init_name]
        node.op_type = "Einsum"
        del node.input[1:]
        del node.attribute[:]
        node.attribute.extend([onnx.helper.make_attribute("equation", equation)])
    removed = {"row_axes", "col_axes"}
    kept = [init for init in graph.initializer if init.name not in removed]
    assert len(kept) + 2 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task088_einsum_row_col.onnx")


def task174_share_axes() -> None:
    """Reuse the identical int64 Unsqueeze axes for Slice and delete int32 axes."""
    model = onnx.load(HERE / "baseline" / "task174.onnx")
    graph = model.graph
    node = graph.node[19]
    assert node.op_type == "Slice"
    assert list(node.input) == ["qin_i8", "starts_i32", "ends_i32", "axes_slice_i32"]
    node.input[3] = "unsq_axes_i64"
    kept = [init for init in graph.initializer if init.name != "axes_slice_i32"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task174_share_axes.onnx")


def task055_sparse_reuse_x() -> None:
    """Reuse X as the degree-5 basis and sparse-store six Acoef slices."""
    model = onnx.load(HERE / "baseline" / "task055.onnx")
    graph = model.graph
    arrays = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    old_basis = (arrays["Qpoly"].astype(np.float64) @ arrays["Lpoly"].astype(np.float64)) ** 5
    target = np.einsum("ql,lgo->qgo", old_basis, arrays["Acoef"].astype(np.float64))

    # q/8 - 5/8 + l/8 for l={0,2,4,6,8,10} gives six well-spaced
    # affine forms.  Their fifth powers span exactly the same degree<=5
    # polynomial space as the incumbent's six shifted fifth powers.
    q = np.arange(10, dtype=np.float64)
    qpoly = np.stack((q / 8.0 - 5.0 / 8.0, np.full(10, 1.0 / 16.0)), axis=1)
    selected_l = np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64)
    new_basis = (qpoly @ arrays["X"].astype(np.float64)) ** 5
    coefficients = np.linalg.solve(new_basis[:6, selected_l], target[:6].reshape(6, 9))
    reconstructed = new_basis[:, selected_l] @ coefficients
    assert np.max(np.abs(reconstructed - target.reshape(10, 9))) < 1e-10

    kept = []
    for init in graph.initializer:
        if init.name == "Lpoly" or init.name == "Acoef":
            continue
        if init.name == "Qpoly":
            kept.append(numpy_helper.from_array(qpoly.astype(np.float32), name="Qpoly"))
        else:
            kept.append(init)
    del graph.initializer[:]
    graph.initializer.extend(kept)

    values = coefficients.astype(np.float32).reshape(-1)
    linear_indices = np.asarray(
        [int(l) * 9 + go for l in selected_l for go in range(9)], dtype=np.int64
    )
    sparse = onnx.helper.make_sparse_tensor(
        numpy_helper.from_array(values, name="Acoef"),
        numpy_helper.from_array(linear_indices, name="Acoef_indices"),
        [30, 3, 3],
    )
    graph.sparse_initializer.extend([sparse])
    final = graph.node[9]
    for index in (1, 3, 5, 7, 9):
        assert final.input[index] == "Lpoly"
        final.input[index] = "X"
    save_checked(model, "task055_sparse_reuse_x.onnx")


if __name__ == "__main__":
    # QLinearConv zero points are required by the ONNX schema; the task153
    # optional-input probe is intentionally not emitted.
    task071_cast_attribute()
    # The task088 probe is checker-rejected because replacing GroupNorm exposes
    # the incumbent's hidden C=10 value-info dimensions; do not emit it.
    for rank in range(1, 7):
        task055_low_rank_acoef(rank)
    task055_low_rank_acoef_side(1)
    task055_low_rank_acoef_side(2)
    for index in range(6):
        task055_drop_quadrature(index)
    task088_einsum_row_col()
    # task174 axes sharing is schema-rejected because Slice requires all three
    # integer index inputs to have a common dtype.
    # task055 sparse X reuse is likewise not emitted: ONNX strict shape
    # inference does not propagate the sparse initializer rank into Einsum.
