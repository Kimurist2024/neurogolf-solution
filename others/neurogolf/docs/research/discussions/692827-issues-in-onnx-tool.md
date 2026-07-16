# Issues in onnx-tool

- Topic ID: 692827
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827
- Author: shinh (@shinh0)
- Posted: 2026-04-18T04:08:28.584829900Z
- Votes: 10
- Total messages: 36

## Body

I think there are several edge cases in onnx-tool that may lead to unintended score reductions. The simplest one I found is https://gist.github.com/shinh/34b3f6af69fa7a2cd115c84b5ad476b8 Its output is:

```
$ python3 bugs/expand.py
Sqrt == Expand: True
Sqrt score: (216000, 36000, 0)
Expand score: (720, 248, 1)
```

This shows `Sqrt(Expand(input, [30]))` consumes much less MACs/memory than `Sqrt(input)`, even though they are equivalent.

I have noticed a few other similar cases, and at least one of them seems exploitable in the context of this competition.

Just to be clear, I have not used this trick in my submissions.

## Comments (36)

- **Yiheng Wang** (2026-04-21T04:58:20.797Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  Thanks for pointing our this, I've switched to use 1.0.1 for local tests to avoid my model exploiting bugs. Hope the host can update the scoring tools to let everyone get aligned scores.

  - **(unknown)** (2026-04-21T07:05:29.813Z, votes: {}):
    (deleted)

- **hengck23** (2026-04-26T15:06:29.713Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Expand should be legal. that is the same as in pytorch. There is no increase in memory becuase it  is repeated values and hence no need to recompute.

- **kq5y** (2026-04-19T06:20:43.597Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  Hi @kevinyuluo , I found two `onnx_tool` profiling issues that cause the scorer to report incorrect costs. A self-contained reproduction script that works in v1.0.0 is https://gist.github.com/kq5y/2e8e102d4dd75ecb61d8f9db6328d9ef
  
  ### Bug 1: Negative-step `Slice` has an off-by-one in shape inference
  
  When `Slice` uses a negative step, `onnx_tool` underestimates the output length by 1 in a full-reversal case. For example, slicing an axis of length 30 with `starts=29`, `ends=-100`, and `steps=-1` should return 30 elements, but `onnx_tool` infers 29.
  
  ```python
  # Reverse axis=2 of shape [1, 10, 30, 30]
  # Runtime output: [1, 10, 30, 30]
  # onnx_tool infers: [1, 10, 29, 30]
  helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["rev"])
  # starts=[29], ends=[-100], axes=[2], steps=[-1]
  ```
  
  This propagates to downstream profiling. In the attached PoC, an `Add` after the `Slice` is reported as 8,700 MACs instead of the correct 9,000.
  
  ### Bug 2: `ConstantOfShape` can collapse to zero-sized outputs when its shape depends on input values
  
  `shape_infer(None)` evaluates value-dependent shape subgraphs using zero-filled dummy inputs. If a `ConstantOfShape` dimension is derived from input data, the inferred shape can collapse to zero even when the runtime value is guaranteed to be nonzero for valid inputs.
  
  ```python
  # Probe input[:, :, 0:1, 0:1] -> ReduceSum -> Cast -> * 30
  # Use the result as two dimensions of ConstantOfShape
  # Runtime with valid input: ConstantOfShape produces [1, 10, 30, 30]
  # onnx_tool shape_infer(None): ConstantOfShape produces [1, 10, 0, 0]
  ```
  
  This also propagates to downstream profiling. In the attached PoC, an `Add` after `ConstantOfShape` is reported as 0 MACs under `shape_infer(None)`, but 9,000 MACs when the same model is profiled with the actual runtime input.
  
  Both models execute correctly at runtime. The issue is in `onnx_tool`'s profiling/inference path, not in ONNX execution.
  
  ***It seems the Bug 1 was solved while I was writing this.***

  - **Yu Luo** (2026-04-19T06:28:21.890Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    Thanks for your reproduction script! I'm working on it.

    - **kq5y** (2026-04-19T06:49:17.730Z, votes: {'canUpvote': True}):
      @kevinyuluo Thank you for handling this. But after 585b8449, `_broadcast_shape` rejects `(0, 1)` as incompatible. This breaks `model_profile` for any model whose value-based shape inference produces zero-sized intermediate tensors.
      
      PoC: https://gist.github.com/kq5y/37b0e4139d562822b656e3ed8eb7a94d
      
      Profiling should not fail on zero-sized extents produced during value-based shape inference. For broadcasts like `(0, 1)`, the inferred output extent should be `0`, not an exception.

    - **Yu Luo** (2026-04-19T07:03:38.133Z, votes: {'canUpvote': True}):
      It's basically the same question as Bug 2. The input tensor of your graph will be used to compute the output tensor shape. So, you can't use an arbitrary tensor ( all zeros).

    - **kq5y** (2026-04-19T07:21:32.510Z, votes: {'canUpvote': True}):
      Thanks. To remove the input-dependent-shape part entirely, here is a smaller repro that uses only constants.
      
      It creates a constant empty tensor of shape [0] and multiplies it by a constant tensor of shape [1]. This is a pure broadcast case with no Range/Slice/input-derived shape involved.
      
      https://gist.github.com/kq5y/37b0e4139d562822b656e3ed8eb7a94d#file-new_onnx_tool_broadcast_bug_poc-py
      
      So I think the remaining issue is specifically that `_broadcast_shape` rejects broadcasting involving zero-sized extents (in this case, [0] with [1]), rather than anything related to zero-filled profiling inputs.

    - **Yu Luo** (2026-04-19T07:47:36.637Z, votes: {'canUpvote': True}):
      Neither ONNX nor Numpy defines the behavior when broadcasting a zero-sized tensor. Did you mean a 0-D array? e.g., a = numpy.array(10), it's a scalar array.

    - **kq5y** (2026-04-19T07:57:54.370Z, votes: {'canUpvote': True}):
      No. I mean a 1-D empty array with shape [0], not a 0-D scalar array.
      
      NumPy arrays can have zero-length dimensions: an ndarray shape is a tuple of non-negative integers. NumPy’s broadcasting rules say dimensions are compatible when they are equal or one of them is 1, and ONNX multidirectional broadcasting is defined as NumPy-style broadcasting.
      https://numpy.org/doc/stable/user/basics.broadcasting.html
      
      Also, this is not just theoretical. In a local test:
      - `np.broadcast_shapes((0,), (1,))` returns `(0,)`
      - `np.array([], dtype=np.float32) * np.array([2.0], dtype=np.float32)` produces an array with shape `(0,)`
      - ONNX Runtime executes the attached model and returns output shape `(0,)`
      
      So even if the docs do not explicitly discuss zero-sized extents, `_broadcast_shape` currently differs from NumPy/ONNX runtime behavior on this case.

    - **Yu Luo** (2026-04-19T08:18:24.653Z, votes: {'canUpvote': True}):
      This case has been supported in 6787341597e1fd93b8cc48d0b1636b52734bc673. As it's undefined, the code may fail in other cases.

  - **Yu Luo** (2026-04-19T07:00:23.137Z, votes: {'canUpvote': True}):
    It's not a bug for your Bug 2 case. The shape of ConstantOfShape is computed by the input tensor's values. When you pass with all ones, it results in one MAC count of a fixed shape(1x10x30x30). The shape equation is: 1x10x(input[0, 0, 0, 0]*30)x(input[0, 0, 0, 0]*30). If intput[0,0,0,0]=2, you will get a new MAC of the shape(1x10x60x60). So, you can't use the default input tensor as all zeros.

    - **kq5y** (2026-04-19T07:32:38.827Z, votes: {'canUpvote': True}):
      Thanks for the clarification. You're right — `ConstantOfShape` is working as designed since the output shape is data-dependent. The issue is on the scorer side: `neurogolf_utils.py` calls `graph.shape_infer(None)`, which evaluates with zero-filled inputs, so any data-derived shape collapses to 0. Passing a representative input (e.g., an actual training example) to `shape_infer()` would give accurate profiling for these cases.

- **Mukundan** (2026-04-20T15:15:56.617Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  @kevinyuluo Constant nodes don't seem to get counted toward memory usage: https://gist.github.com/Mukundan314/6463d1895be31de8156eaa13079a8f07
  
  Looks like a bug given the comment at https://github.com/ThanatosShinji/onnx-tool/blob/427849/onnx_tool/graph.py#L1404 implies the bytes should be accounted for elsewhere.

  - **Yu Luo** (2026-04-21T12:34:28.073Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Constant's parameter and memory should be counted in the node where it is actually used. It's the Add node for your case. But onnx-tool failed to do so. I will look into it.

    - **Yu Luo** (2026-04-21T13:02:59.910Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      For a quick fix, you can apply this change to your code: `g = onnx_tool.Model(model, mcfg={'constant_folding': True}).graph` @mukundan314 
      ```python
      ------  ------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
      Add_1   Add     1,024           100.00%     8,192     100.00%     1,024     100.00%     1024       1024
      Total   _       1,024           100%        8,192     100%        1,024     100%        _          _
      ```

    - **Yu Luo** (2026-04-21T13:23:15.747Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      Highly recommend enabling 'constant_folding' while loading a model @mmoffitt. As: `model = onnx_tool.loadmodel(m, {'verbose': False, 'constant_folding': True})`
      'constant_folding=True' has been verified by onnx-tool's model tests.

  - **Michael D. Moffitt** (2026-04-28T21:33:51.957Z, votes: {'canUpvote': True}):
    This issue should now be fixed (see also: [our metric update from today](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230)).  In particular, our scoring engine will perform *constant folding* on all networks before scoring.

- **Yu Luo** (2026-04-18T16:02:20.277Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Fixed as:
  
  PS C:\tmp> python -m onnx_tool -m profile -i .\expand.onnx
  ```python
  Name      Type    Forward_MACs    FPercent    Memory    MPercent      Params  PPercent    InShape     OutShape
  --------  ------  --------------  ----------  --------  ----------  --------  ----------  ----------  ----------
  Expand_0  Expand  0               0.00%       36,008    50.01%             1  100.00%     1x10x30x30  1x10x30x30
  Sqrt_1    Sqrt    216,000         100.00%     36,000    49.99%             0  0.00%       1x10x30x30  1x10x30x30
  Total     _       216,000         100%        72,008    100%               1  100%        _           _
  ```
  PS C:\tmp> python -m onnx_tool -m profile -i .\sqrt.onnx
  ```python
  Name    Type    Forward_MACs    FPercent    Memory    MPercent      Params  PPercent    InShape     OutShape
  ------  ------  --------------  ----------  --------  ----------  --------  ----------  ----------  ----------
  Sqrt_0  Sqrt    216,000         100.00%     36,000    100.00%            0  0.00%       1x10x30x30  1x10x30x30
  Total   _       216,000         100%        36,000    100%               0  100%        _           _
  ```

  - **shinh** (2026-04-20T00:03:15.893Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    Thanks! I don't have time to prepare minimal repros right now, but the following parts look potentially exploitable to me.
    
    https://github.com/ThanatosShinji/onnx-tool/blob/df70e97b294b080afb3f8d55dc3edada55a77ad4/onnx_tool/node.py#L1076
    ```
    EinSum(tensor(shape=(A,B,C,D)), tensor(shape=(1,1,1,1)), equaltion="ijkl,ijkl->ijkl")
    ```
    
    It seems this would produce `tensor(shape=(1,1,1,1))`, while the actual result should broadcast to `(A,B,C,D)`.
    
    
    https://github.com/ThanatosShinji/onnx-tool/blob/df70e97b294b080afb3f8d55dc3edada55a77ad4/onnx_tool/node.py#L80-L91
    ```
    Pow(tensor(shape=(A,1)), tensor(shape=(1,A)))
    ```
    This might be inferred as `tensor(shape=(A,1))`, while the actual broadcasted shape would be `(A,A)`.

    - **Yu Luo** (2026-04-21T12:42:39.007Z, votes: {'canUpvote': True}):
      Here is ONNX's definition for broadcasting relu of EinSum: The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions. Your EinSum equation 'equaltion="ijkl,ijkl->ijkl"' is not a valid broadcast equation.

    - **Yu Luo** (2026-04-21T12:43:10.130Z, votes: {'canUpvote': True}):
      You are right about Pow, it's a broadcasting operator. It will be fixed soon.

    - **(unknown)** (2026-04-21T13:11:05.657Z, votes: {}):
      (deleted)

    - **Yu Luo** (2026-04-21T13:11:18.130Z, votes: {'canUpvote': True}):
      Done 75fa870.

    - **shinh** (2026-04-22T04:05:50.090Z, votes: {'canUpvote': True}):
      I created a repro for EinSum: https://gist.github.com/shinh/a958fff943b98e1b65464048483a9494
      
      Pow is just an example, I think all callers of `_max_shape` should use `_broadcast_shape` instead.

    - **Yu Luo** (2026-04-26T12:49:49.067Z, votes: {'canUpvote': True}):
      ONNX Runtime's implementation is more permissive than the specification. I added the support of broadcast in Einsum anyway.

- **Yu Luo** (2026-04-18T15:56:13.967Z, votes: {'canUpvote': True}):
  Expand in onnx-tool missed the process code of the condition of 'the shape.ndim < input.shape.ndim.'.

- **(unknown)** (2026-04-18T16:36:32.903Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
  (deleted)

  - **Yu Luo** (2026-04-19T04:40:50.870Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Hi @robga , I'm the author of onnx-tool. Can you provide a reproducer of the profile issue?

    - **(unknown)** (2026-04-19T08:39:55.513Z, votes: {}):
      (deleted)

    - **Yu Luo** (2026-04-19T08:50:47.200Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      fixed as:
      ONNX inferred shape: [1, 10, 0, 0]
      ORT metadata shape: [1, 10, 0, 0]
      ORT runtime shape: (1, 10, 0, 0)
      onnx_tool shape: [1, 10, 0, 0]

    - **(unknown)** (2026-04-21T12:13:00.820Z, votes: {}):
      (deleted)

    - **Yu Luo** (2026-04-21T12:54:27.223Z, votes: {'canUpvote': True}):
      Yes, please provide a repro. @robga

    - **(unknown)** (2026-04-21T17:47:08.097Z, votes: {}):
      (deleted)

    - **Yu Luo** (2026-04-26T09:20:36.643Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
      Bug3 is caused by the change in dynamic shape definition. `onnx-tool` does not support dynamic shape inference, so it should raise an error here. Now this bug has been fixed. You should use `shape_infer` like this: `graph.shape_infer({'input': _build_sample_input()})`.

    - **Yu Luo** (2026-04-26T12:12:00.753Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      It's the same for Bug2. You should pass the input to shape_infer, which is needed to calculate the output shape. `graph.shape_infer({'input': sample_input})`
