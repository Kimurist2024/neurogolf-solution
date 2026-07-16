# patch for dynamic crop/slice in solution for compliance with new memory profiler

- Topic ID: 695972
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695972
- Author: hengck23 (@hengck23)
- Posted: 2026-04-30T22:04:31.728712900Z
- Votes: 3
- Total messages: 4

## Body

in taskname = "task111", you are asked to dynamically crop an object.  
if you use dynamic slicing, it will gives "Slice_2_output: 1 x 10 x ? x ?" and  the new memory profiler will complain

```
Error: tensor Slice_2_output has symbolic dim_param.
```

but our **cropped shape is fixed. ** it is just that the location varies. Hence we can patch the model to fill in the shape.
in other problem, we can crop to largest size, or use gather etc

```
## definietly not the best solution. added from illustratino only

def make_onnx():
    SHAPE = (1, 10, 30, 30)
    input = argument(Tensor(np.float32, SHAPE))
    from spox.opset.ai.onnx import v17 as op

    def op_constant(value, dtype=np.int64, shape=None):
        value = np.asarray(value, dtype=dtype)
        if shape is not None:
            value = value.reshape(shape)
        return op.constant(value=value)

    def op_slice(x, starts, ends, axes, steps):
        return op.slice(
            x,
            starts=op_constant(starts),
            ends=op_constant(ends),
            axes=op_constant(axes),
            steps=op_constant(steps),
        )

    # ------------------------------------------------------------------
    # 1) Marker window: input channel 5, rows 0..6, cols 1..8.
    #    Shape: 1 x 1 x 7 x 8.
    # ------------------------------------------------------------------
    marker = op_slice(input, [5], [6], [1], [1])
    marker_win = op_slice(marker, [0, 1], [7, 9], [2, 3], [1, 1])

    # ------------------------------------------------------------------
    # 2) Use the one-hot marker window to select crop starts.
    #
    # For marker at global (row=r, col=c), crop starts are:
    #   row_start = r + 1
    #   col_start = c - 1
    # Since marker_win cols are local c_local=0..7 for global c=1..8,
    #   col_start = c_local.
    # ------------------------------------------------------------------
    row_weight = np.zeros((1, 1, 7, 8), dtype=np.float32)
    col_weight = np.zeros((1, 1, 7, 8), dtype=np.float32)
    for r in range(7):
        for c_local in range(8):
            row_weight[0, 0, r, c_local] = r + 1
            col_weight[0, 0, r, c_local] = c_local

    axes_all = op_constant([0, 1, 2, 3])

    row_start_f = op.reduce_sum(
        op.mul(marker_win, op_constant(row_weight, dtype=np.float32)),
        axes=axes_all,
        keepdims=0,
    )
    col_start_f = op.reduce_sum(
        op.mul(marker_win, op_constant(col_weight, dtype=np.float32)),
        axes=axes_all,
        keepdims=0,
    )

    row_start_i = op.cast(row_start_f, to=np.int64)
    col_start_i = op.cast(col_start_f, to=np.int64)
    row_end_i = op.add(row_start_i, op_constant(3, dtype=np.int64))
    col_end_i = op.add(col_start_i, op_constant(3, dtype=np.int64))

    # Slice starts/ends must be 1D tensors of length 2 for axes [2,3].
    unsq0 = op_constant([0])
    row_start_1d = op.unsqueeze(row_start_i, axes=unsq0)
    col_start_1d = op.unsqueeze(col_start_i, axes=unsq0)
    row_end_1d = op.unsqueeze(row_end_i, axes=unsq0)
    col_end_1d = op.unsqueeze(col_end_i, axes=unsq0)

    crop_starts = op.concat([row_start_1d, col_start_1d], axis=0)
    crop_ends = op.concat([row_end_1d, col_end_1d], axis=0)

    # ------------------------------------------------------------------
    # 3) Crop once dynamically, then pad once to 30x30 top-left.
    # ------------------------------------------------------------------
    crop = op.slice(
        input,
        starts=crop_starts,
        ends=crop_ends,
        axes=op_constant([2, 3]),
        steps=op_constant([1, 1]),
    )

    output = op.pad(
        crop,
        pads=op_constant([0, 0, 0, 0, 0, 0, 27, 27]),
        mode="constant",
    )

    model = spox.build(
        inputs={"input": input},
        outputs={"output": output},
    )

    #-------- start patch ------------------------
    # Error: tensor Slice_2_output has symbolic dim_param.
    if 1:
        target_shapes = {
            "Slice_2_output": [1, 10, 3, 3],
        }

        def patch_vi(vi, shape):
            tt = vi.type.tensor_type
            tt.elem_type = onnx.TensorProto.FLOAT
            tt.shape.dim.clear()
            for d in shape:
                tt.shape.dim.add().dim_value = d

        found = set()

        for vi in list(model.graph.value_info) + list(model.graph.output):
            if vi.name in target_shapes:
                patch_vi(vi, target_shapes[vi.name])
                found.add(vi.name)

        for name, shape in target_shapes.items():
            if name not in found:
                model.graph.value_info.append(
                    onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
                )
    #-------- end patch ------------------------
    onnx.checker.check_model(model)
    #model, ok = simplify(model)
    return model



```

## Comments (4)

- **hengck23** (2026-05-01T05:25:02.660Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  variable input size trick everyone should know. Since output memory is not counted, memory planning is to crop, process and pad back. Padding back dynamically for variable image size can be challenging 
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff7c3f308445faee5791b3389190a9275%2FSelection_3230.png?generation=1777613586820129&alt=media)

  - **hengck23** (2026-05-01T09:22:48.283Z, votes: {'canUpvote': True}):
    ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F8c8fe632369f47900a4f27d035d258a6%2FSelection_3233.png?generation=1777627366669356&alt=media)

- **hengck23** (2026-05-03T05:09:50.747Z, votes: {'canUpvote': True}):
  https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html
  
  making dynamic dim fixed

- **hengck23** (2026-05-03T00:57:29.707Z, votes: {'canUpvote': True}):
  gemini offers a simpler solution and also works for dynamic ops like compress.
  reshape it ** immediately ** after the op
  
  ```
      red_index = i32_const(make_red_digonal())
      starts = op.mul(si, i64_const([24]))
      ends  = op.add(starts, i64_const([24]))
      red_idx =  op.reshape(
          op.slice(red_index, starts, ends), #dynamic slice but size is fixed at 24
          shape=op_constant([24], np.int64)
      )
  
  ```
