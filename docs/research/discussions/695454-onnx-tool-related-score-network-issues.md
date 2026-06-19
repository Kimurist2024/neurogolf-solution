# onnx-tool related `score_network` issues

- Topic ID: 695454
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695454
- Author: keymoon (@keymoon)
- Posted: 2026-04-29T14:17:43.082324200Z
- Votes: 7
- Total messages: 

## Body

Thank you for the recent `neurogolf_utils` update, which resolved several previously used hacks. But, I still found a few `onnx-tool`-related issues affecting `score_network()`.

At the moment, these fall into two categories:
1. incorrect shape inference causing `MACs` to disagree with the actual graph behavior
2. missing edge-case handling or implementation gaps causing `score_network()` to fail on graphs that otherwise run correctly in ORT

So far, I have confirmed the following cases.

1. `GatherND(batch_dims)` ignores `batch_dims` and inflates downstream `MACs`
2. `TopK` lacks `value_infer` in dynamic-shape contexts
3. `Compress` with omitted `axis` rejects same-rank condition
4. `Pad` default mode crashes because `str` is decoded as bytes
5. broadcast `Where` followed by dynamic `Reshape` asserts on volume
6. `UINT4 Constant` packed `raw_data` cannot be loaded

Below I summarize each case together with a small PoC. I would appreciate any guidance on the intended handling or prioritization for these issues.

## 1. GatherND(batch_dims) ignores batch_dims and inflates downstream MACs

**Detail**

The repro graph is `GatherND(batch_dims=2) -> Greater(scalar zero)`. `onnx-tool` computes the `GatherND` output as if `batch_dims=0`, and the wrong shape then corrupts downstream `Greater` MAC profiling.

**Expected**

- Gather output shape: `[1, 9]`
- Final output shape: `[1, 9]`
- MACs: `9`

**Actual**

- `onnx-tool` reports gather/output shape `[1, 9, 25, 25]`
- `score_network()` reports `MACs = 5625`

**Recommended fix**

- Read and use the `batch_dims` attribute in `GatherNDNode.shape_infer` and `value_infer`.

**Repro script**

```python
import numpy as np
import onnx
import onnx_tool
import onnxruntime as ort
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper

from common import print_exception, print_versions, run_new_score_network, save_model


def make_const(name: str, value: np.ndarray) -> onnx.NodeProto:
    return H.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.asarray(value)),
    )


def build_model() -> onnx.ModelProto:
    input_info = H.make_tensor_value_info("input", TP.UINT8, [1, 9, 25, 25])
    output_info = H.make_tensor_value_info("output", TP.BOOL, [1, 9])

    nodes = [
        make_const("indices", np.zeros((1, 9, 2), dtype=np.int64)),
        H.make_node(
            "GatherND",
            ["input", "indices"],
            ["gathered"],
            batch_dims=2,
            name="gather",
        ),
        make_const("zero", np.array(0, dtype=np.uint8)),
        # This Greater node makes the wrong GatherND shape show up directly in MAC profiling.
        H.make_node("Greater", ["gathered", "zero"], ["output"], name="compare"),
    ]

    value_info = [H.make_tensor_value_info("gathered", TP.UINT8, [1, 9])]
    graph = H.make_graph(nodes, "g", [input_info], [output_info], value_info=value_info)
    return H.make_model(graph, ir_version=11, opset_imports=[H.make_opsetid("", 24)])


def volume(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


if __name__ == "__main__":
    print_versions()

    model = build_model()
    model_path = save_model(model)
    feed = {"input": np.zeros((1, 9, 25, 25), dtype=np.uint8)}

    inferred = onnx.shape_inference.infer_shapes(model).graph
    expected_gather_shape = [
        dim.dim_value for dim in inferred.value_info[0].type.tensor_type.shape.dim
    ]
    expected_output_shape = list(
        ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        .run(None, feed)[0]
        .shape
    )
    expected_macs = volume(expected_output_shape)

    graph = onnx_tool.loadmodel(
        model_path,
        {"verbose": False, "constant_folding": True},
    ).graph
    graph.shape_infer(None)
    graph.profile()

    gather_node = graph.nodemap["gather"]
    compare_node = graph.nodemap["compare"]
    actual_macs = int(sum(graph.macs))

    print("graph: gathernd(batch_dims=2) -> greater(scalar zero)")
    print("  expected gather shape:", expected_gather_shape)
    print("  tool gather shape    :", gather_node.outshape)
    print("  expected output shape:", expected_output_shape)
    print("  tool output shape    :", compare_node.outshape)
    print("  expected macs        :", expected_macs)
    print("  tool macs            :", actual_macs)
    try:
        print("  score_network        :", run_new_score_network(model_path))
    except Exception as exc:
        print_exception("  score_network", exc)
```

**Observed output**

```text
versions: python: 3.14.4, onnx: 1.21.0, onnx-tool: 1.0.1, onnxruntime: 1.24.4
graph: gathernd(batch_dims=2) -> greater(scalar zero)
  expected gather shape: [1, 9]
  tool gather shape    : [1, 9, 25, 25]
  expected output shape: [1, 9]
  tool output shape    : [1, 9, 25, 25]
  expected macs        : 9
  tool macs            : 5625
  score_network        : (5625, 154, 18)
```

## 2. TopK still lacks value_infer in dynamic-shape contexts

**Detail**

The repro graph slices a cell, runs `TopK`, and then uses the selected branch to slice again. Plain `TopK` shape inference works, but downstream shape logic that needs the values crashes.

**Expected**

- `score_network()` should complete successfully.

**Actual**

- `score_network()` raises `NotImplementedError` through `TopK.value_infer`.

**Recommended fix**

- Implement `TopKNode.value_infer`, or avoid unconditional `value_infer` on `shape_calc` paths.

**Repro script**

```python
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper

from common import print_exception, print_versions, run_new_score_network, save_model


def make_const(name: str, value, dtype) -> onnx.NodeProto:
    return H.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.asarray(value, dtype=dtype)),
        name=f"const_{name}",
    )


def build_model() -> onnx.ModelProto:
    input_info = H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 20, 20])
    output_info = H.make_tensor_value_info("output", TP.FLOAT, [1, 10, 1, 1])

    nodes = [
        make_const("k", [1], np.int64),
        make_const("half", [0.5], np.float32),
        make_const("squeeze_axes", [0, 1, 2, 3], np.int64),
        make_const("zero", 0, np.int64),
        make_const("slice_starts_lut", [[0, 0], [1, 1]], np.int64),
        make_const("slice_ends_lut", [[1, 1], [2, 2]], np.int64),
        make_const("slice_axes", [2, 3], np.int64),
        make_const("slice_steps", [1, 1], np.int64),
        make_const("cell_starts", [0, 7], np.int64),
        make_const("cell_ends", [1, 8], np.int64),
        make_const("cell_axes", [2, 3], np.int64),
        H.make_node("Slice", ["input", "cell_starts", "cell_ends", "cell_axes"], ["cell"]),
        H.make_node(
            "TopK",
            ["cell", "k"],
            ["top_values", "top_indices"],
            axis=1,
            largest=1,
            sorted=0,
            name="pick_branch",
        ),
        H.make_node("Greater", ["top_values", "half"], ["branch_mask_4d"]),
        H.make_node("Squeeze", ["branch_mask_4d", "squeeze_axes"], ["branch_mask"]),
        H.make_node("Cast", ["branch_mask"], ["branch_id"], to=TP.INT64),
        H.make_node("Add", ["zero", "branch_id"], ["branch_index"]),
        H.make_node("Gather", ["slice_starts_lut", "branch_index"], ["slice_starts"], axis=0),
        H.make_node("Gather", ["slice_ends_lut", "branch_index"], ["slice_ends"], axis=0),
        H.make_node(
            "Slice",
            ["input", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
            ["output"],
        ),
    ]

    graph = H.make_graph(nodes, "g", [input_info], [output_info])
    return H.make_model(graph, ir_version=11, opset_imports=[H.make_opsetid("", 24)])


if __name__ == "__main__":
    print_versions()
    model_path = save_model(build_model())
    try:
        print("score_network:", run_new_score_network(model_path))
    except Exception as exc:
        print_exception("score_network", exc)
```

**Observed output**

```text
versions: python: 3.14.4, onnx: 1.21.0, onnx-tool: 1.0.1, onnxruntime: 1.24.4
score_network: NotImplementedError: this Node TopK-pick_branch has no value_infer
```

## 3. Compress with omitted axis rejects same-rank condition

**Detail**

The repro graph feeds a same-rank boolean mask into `Compress` with omitted `axis`. ORT accepts flattening semantics, but `onnx-tool` forwards directly to `numpy.compress` and fails.

**Expected**

- Output shape `[2]`
- `score_network()` should complete successfully.

**Actual**

- `score_network()` raises an error through the `Compress` handling path.

**Recommended fix**

- Implement ONNX `Compress` semantics instead of directly calling `numpy.compress` with the raw tensors.

**Repro script**

```python
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper

from common import print_exception, print_versions, run_new_score_network, save_model


def make_const(name: str, value: np.ndarray) -> onnx.NodeProto:
    return H.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.asarray(value), name=name),
    )


def build_model() -> onnx.ModelProto:
    output_info = H.make_tensor_value_info("out", TP.INT64, [2])

    nodes = [
        make_const("data", np.arange(4, dtype=np.int64).reshape(1, 1, 2, 2)),
        make_const("cond", np.array([[[[1, 0], [1, 0]]]], dtype=bool)),
        H.make_node("Compress", ["data", "cond"], ["out"]),
    ]

    graph = H.make_graph(nodes, "g", [], [output_info])
    return H.make_model(graph, ir_version=11, opset_imports=[H.make_opsetid("", 24)])


if __name__ == "__main__":
    print_versions()
    model_path = save_model(build_model())
    try:
        print("score_network:", run_new_score_network(model_path))
    except Exception as exc:
        print_exception("score_network", exc)
```

**Observed output**

```text
versions: python: 3.14.4, onnx: 1.21.0, onnx-tool: 1.0.1, onnxruntime: 1.24.4
score_network: ValueError: condition must be a 1-d array
```

## 4. Pad default mode crashes because str is decoded as bytes

**Detail**

The repro graph is `Pad -> Slice`, where `Pad` relies on its default mode. `Pad` without an explicit mode can hit a code path where `self.mode` is `str` and `.decode()` is called on it.

**Expected**

- Default `constant` mode should work.
- `score_network()` should complete successfully.

**Actual**

- `score_network()` raises an error through the `Pad` handling path.

**Recommended fix**

- Normalize `mode` to `str` once and stop calling `.decode()` unconditionally.

**Repro script**

```python
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper

from common import print_exception, print_versions, run_new_score_network, save_model


def make_const(name: str, value: np.ndarray) -> onnx.NodeProto:
    return H.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.asarray(value)),
    )


def build_model() -> onnx.ModelProto:
    output_info = H.make_tensor_value_info("y", TP.FLOAT, [2])

    nodes = [
        make_const("data", np.arange(4, dtype=np.float32)),
        make_const("start0", np.array([1], dtype=np.int64)),
        make_const("pads", np.array([0, 0], dtype=np.int64)),
        make_const("zero", np.array(0, dtype=np.int64)),
        make_const("ends", np.array([3], dtype=np.int64)),
        make_const("axes", np.array([0], dtype=np.int64)),
        H.make_node("Pad", ["start0", "pads", "zero"], ["starts"]),
        H.make_node("Slice", ["data", "starts", "ends", "axes"], ["y"]),
    ]

    graph = H.make_graph(nodes, "g", [], [output_info])
    return H.make_model(graph, ir_version=11, opset_imports=[H.make_opsetid("", 24)])


if __name__ == "__main__":
    print_versions()
    model_path = save_model(build_model())
    try:
        print("score_network:", run_new_score_network(model_path))
    except Exception as exc:
        print_exception("score_network", exc)
```

**Observed output**

```text
versions: python: 3.14.4, onnx: 1.21.0, onnx-tool: 1.0.1, onnxruntime: 1.24.4
score_network: AttributeError: 'str' object has no attribute 'decode'
```

## 5. Broadcast Where followed by dynamic Reshape asserts on volume

**Detail**

The repro graph is `Where -> Greater -> Where -> Reshape`. ORT treats the first `Where` output as fully broadcast, but `onnx-tool` later asserts on a thinner profiled volume.

**Expected**

- Shape inference should succeed.
- `score_network()` should complete successfully.

**Actual**

- `score_network()` raises an `AssertionError` through the `Reshape` handling path.

**Recommended fix**

- Coordinate broadcast-aware shape representation with `ReshapeNode.shape_infer`.

**Repro script**

```python
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper

from common import print_exception, print_versions, run_new_score_network, save_model


def make_const(name: str, value, dtype=None) -> onnx.NodeProto:
    return H.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.asarray(value, dtype=dtype)),
    )


def build_model() -> onnx.ModelProto:
    mask_info = H.make_tensor_value_info("mask", TP.BOOL, [1, 1, 1, 30])
    flag_info = H.make_tensor_value_info("flag", TP.FLOAT, [1])
    out_info = H.make_tensor_value_info("y", TP.UINT8, [1, 10, 30, 1])

    nodes = [
        make_const("zero_float", [0.0], np.float32),
        make_const("zero_u8", np.array(0, dtype=np.uint8)),
        make_const("paint_vector", np.eye(10, dtype=np.uint8)[1].reshape(1, 10, 1, 1)),
        make_const("shape_horizontal", [1, 10, 30, 1], np.int64),
        make_const("shape_vertical", [1, 10, 1, 30], np.int64),
        H.make_node("Where", ["mask", "paint_vector", "zero_u8"], ["paint"]),
        H.make_node("Greater", ["flag", "zero_float"], ["cond"]),
        H.make_node("Where", ["cond", "shape_horizontal", "shape_vertical"], ["shape"]),
        H.make_node("Reshape", ["paint", "shape"], ["y"]),
    ]

    graph = H.make_graph(nodes, "g", [mask_info, flag_info], [out_info])
    return H.make_model(graph, ir_version=11, opset_imports=[H.make_opsetid("", 24)])


if __name__ == "__main__":
    print_versions()
    model_path = save_model(build_model())
    try:
        print("score_network:", run_new_score_network(model_path))
    except Exception as exc:
        print_exception("score_network", exc)
```

**Observed output**

```text
versions: python: 3.14.4, onnx: 1.21.0, onnx-tool: 1.0.1, onnxruntime: 1.24.4
score_network: AssertionError:
```

## 6. UINT4 Constant packed raw_data cannot be loaded

**Detail**

The repro graph is `Constant(UINT4) -> Cast(UINT8)`. A `Constant` node holding packed `UINT4` data fails before shape inference.

**Expected**

- Decode correctly or reject with a clear unsupported-dtype error.
- `score_network()` should complete successfully.

**Actual**

- `score_network()` raises an error during `onnx-tool` model loading.

**Recommended fix**

- Add explicit sub-byte dtype/raw_data decoding or an explicit unsupported-dtype path.

**Repro script**

```python
import numpy as np
import ml_dtypes
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper

from common import print_exception, print_versions, run_new_score_network, save_model


def build_model() -> onnx.ModelProto:
    packed_uint4 = np.array([0, 1, 1, 0], dtype=ml_dtypes.uint4).reshape(1, 1, 1, 4)
    output_info = H.make_tensor_value_info("y", TP.UINT8, [1, 1, 1, 4])

    nodes = [
        H.make_node(
            "Constant",
            [],
            ["u4"],
            value=numpy_helper.from_array(packed_uint4, "u4"),
        ),
        H.make_node("Cast", ["u4"], ["y"], to=TP.UINT8),
    ]

    graph = H.make_graph(nodes, "g", [], [output_info])
    return H.make_model(graph, ir_version=11, opset_imports=[H.make_opsetid("", 24)])


if __name__ == "__main__":
    print_versions()
    model_path = save_model(build_model())
    try:
        print("score_network:", run_new_score_network(model_path))
    except Exception as exc:
        print_exception("score_network", exc)
```

**Observed output**

```text
versions: python: 3.14.4, onnx: 1.21.0, onnx-tool: 1.0.1, onnxruntime: 1.24.4
score_network: ValueError: buffer size must be a multiple of element size
```

## Comments (0)

(no comments)
