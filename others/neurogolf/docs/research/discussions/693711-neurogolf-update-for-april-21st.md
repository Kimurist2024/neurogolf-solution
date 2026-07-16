# NeuroGolf Update for April 21st

- Topic ID: 693711
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711
- Author: Michael D. Moffitt (@mmoffitt)
- Posted: 2026-04-21T23:02:27.763522500Z
- Votes: 29
- Total messages: 25
- Pinned: yes

## Body

Many thanks to the hundreds of teams who have entered our NeuroGolf Championship thus far, and especially to those that have suggested improvements to the competition!  We are pleased to announce the following updates:

- At the [request](https://www.kaggle.com/competitions/neurogolf-2026/discussion/691888) of @jazivxt, we have increased the number of submissions per day to **100**.
- We have also updated our metric to ignore test cases whose grid dimensions eclipse **30x30**, as [reported](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692621) by @kosirowada.
- Our scorer is now configured to report **specific failing networks** ([suggested](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692195) by @anglolodorf and many others).
- We have patched a **negative memory vulnerability** [discovered](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693247) by @calibrator.
- Several bugs in the **onnx-tool profiler** ([identified](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827) by @shinh0) have been graciously addressed by its creator, @kevinyuluo.
- In response to @tonylica's [question](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693088) about **versions**, we have pinned our metric and [official starter notebook](https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-championship) to the following releases (some of which are upgrades): `numpy` (2.4.4), `onnx` (1.21.0), `onnxruntime` (1.24.4), and `onnx-tool` (1.0.1 for now)
- Sometime soon (likely tomorrow) we'll kick off a **rescoring of all submissions**, and will post a comment below once that process is complete — you may see a temporary shuffling of positions on the leaderboard during this time.

We'll continue to monitor messages on the discussion forum for additional suggestions, with the intention of resolving any remaining ambiguities over the next few weeks.  Thank you for your patience!

---

***[Refer to the bottom of our [welcome message](https://www.kaggle.com/competitions/neurogolf-2026/discussion/691461) for a complete list of all contest updates]***

## Comments (25)

- **Jiwei Liu** (2026-04-22T22:53:51.507Z, votes: {'totalVotes': 8, 'canUpvote': True, 'totalUpvotes': 8}):
  Our agent might also find several loop holes. unfortunately all three of us are busy with something else we don't have time manually inspecting. We'll ask the agent to write a report and share here.

- **Michael D. Moffitt** (2026-04-24T23:46:24.617Z, votes: {'totalVotes': 6, 'canUpvote': True, 'totalUpvotes': 6}):
  **Update (April 24th):** Early next week, we plan to issue *two more metric updates* that will **(a)** explicitly enforce our "statically-shaped" constraint, and **(b)** include the contribution of parameters from `Constant` operations. Soon afterwards, we aim to kick-off our batch rescoring process. Thank you!

  - **Michael D. Moffitt** (2026-04-28T21:43:14.293Z, votes: {'canUpvote': True}):
    [Metric update complete](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230)!  Please feel free to respond to that thread with questions and comments.

- **Mukundan** (2026-04-22T06:10:54.583Z, votes: {'totalVotes': 6, 'canUpvote': True, 'totalUpvotes': 6}):
  The rules state:
  
  > All tensors and parameters in each ONNX network file must have statically-defined shapes so that the performance of the network can be properly evaluated.
  
  I couldn't find this being enforced in the scorer script. On top of that, when dynamic shapes are used the current scorer appears to under-report memory cost, which inflates the reported score. Could you clarify how the rule is intended to apply?
  
  1. Is this rule still current, or is it outdated?
  2. Will the scorer be updated to check it automatically?
  3. Or will compliance be reviewed manually?
  
  For example this solution for task179 uses dynamic shapes to inflate the score:
  
  ```python
  from spox import Tensor, argument, build
  from spox.opset.ai.onnx import v24 as op
  
  inp = argument(Tensor('float32', (1, 10, 30, 30)))
  crop = op.cast(op.slice(inp, i64([0, 0, 0]), i64([10, 3, 3]), axes=i64([1, 2, 3])), to=np.bool_)
  flip = op.transpose(crop, perm=[0, 1, 3, 2])
  
  one = op.cast(op.reduce_max(crop, keepdims=False), to=np.int64)
  one = op.reshape(one, i64([1]))
  pad27 = op.mul(one, i64([27]))
  
  pads = op.concat([i64([0, 0, 0, 0, 0, 0]), pad27, pad27], axis=0)
  out = op.pad(flip, pads)
  
  model = build({"input": inp}, {"output": out})
  ```

  - **Tony Li** (2026-04-22T21:07:02.430Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    I built a dynamic-pad probe at task179.onnx based your code.
    
    That probe passes kaggle scorer and local correctness on all 267 in-envelope examples for task 179 and scores 18.139, with reported cost 954 and file size 716 bytes.
    
    Net effect versus the current canonical file: +2.937 score.
    
    This is a real score inflation under the current scorer. Task 179’s official local examples are all 3x3, so the crop-to-3x3 trick is semantically valid here. The current scorer only enforces static declared graph input/output shapes, but the cost still comes from onnx_tool.shape_infer(None) -> profile(). 
    
    In this probe, ORT returns [1,10,30,30], while onnx_tool profiles the Pad output as [1,10,3,3], so it undercounts memory badly (memory=846, macs=91, params=17).

    - **Michael D. Moffitt** (2026-04-22T23:04:16.143Z, votes: {'canUpvote': True}):
      Just to confirm: it sounds like you are both advocating for a scorer change similar to the following:
      
      ```python
      def score_network(m):
          ....
          for key in g.nodemap.keys():  # <--- this stanza is new
              if (0,) in [g.nodemap[key].inshape, g.nodemap[key].outshape]:
                  return None, None, None  # Shape not explicitly defined.
          return int(sum(g.macs)), int(g.memory), int(g.params)
      ```
      
      Let me know if something like that would resolve both the ambiguity and the inflation problem.

    - **Tony Li** (2026-04-23T00:18:38.073Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Thanks for checking. I tested your proposed rule locally:
      
      ```python
      for key in g.nodemap.keys():
          if (0,) in [g.nodemap[key].inshape, g.nodemap[key].outshape]:
              return None, None, None
      ```
      
      I do not think that exact check is safe. In `onnx_tool`, `(0,)` is often used for ordinary scalar / no-input cases, especially `Constant` nodes, so it does not reliably mean “shape not explicitly defined”. On my local passing bundle, many valid models would be rejected by that rule : 326 / 400 valid tasks contain at least one node with inshape == (0,) or outshape == (0,). A simple example is a valid canonical task337 model: its Constant node has inshape=(0,), so this rule would reject a normal graph.
      
      I think the fix should target the actual issue instead: validate the **profiled output tensors** against the declared static ONNX outputs, and keep the negative-memory guard. That directly blocks both:
      
      - `task179`-style output/profile mismatches
      - `[1,10,0,0]`-style profiled outputs
      
      A minimal version would be:
      
      ```python
      import math
      import onnx
      
      def norm_shape(shape):
          if shape is None:
              return None
          dims = tuple(int(d) for d in shape)
          return () if dims == (0,) else dims   # treat (0,) as scalar, not "dynamic"
      
      def declared_output_specs(onnx_model):
          specs = []
          for vi in onnx_model.graph.output:
              tt = vi.type.tensor_type
              dims = [int(d.dim_value) for d in tt.shape.dim]
              itemsize = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type).itemsize
              min_bytes = itemsize * math.prod(dims)
              specs.append((vi.name, tuple(dims), min_bytes))
          return specs
      
      def score_network(m, onnx_model):
          g = m.graph
          g.graph_reorder_nodes()
          g.shape_infer(None)
          g.profile()
      
          if not g.valid_profile:
              return None, None, None
      
          # keep the existing negative-memory guard
          for key in g.nodemap:
              if getattr(g.nodemap[key], "memory", 0) < 0:
                  return None, None, None
      
          # validate profiled outputs only
          for name, want_shape, min_bytes in declared_output_specs(onnx_model):
              t = g.tensormap.get(name)
              if t is None:
                  return None, None, None
              if norm_shape(getattr(t, "shape", None)) != want_shape:
                  return None, None, None
              if int(t.get_memsize()) < min_bytes:
                  return None, None, None
      
          return int(sum(g.macs)), int(g.memory), int(g.params)
      ```
      
      So do not reject every `(0,)` node shape; instead, validate profiled output shape and output memory against the declared static outputs. That seems to address both the ambiguity and the inflation directly.

    - **Tony Li** (2026-04-23T00:32:06.367Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
      I tested a second version on the local pinned stack, and it behaves correctly on the known cases.
      
      Observed results:
      - inflated `inflation_onnx/task179.onnx`: blocked, because the declared output is `[1,10,30,30]` but the profiled output is `[1,10,3,3]`
      - dynamic-I/O `task337` example: blocked, because `_declared_output_specs()` rejects non-static declared output dimensions
      
      This version seems safer than the shorter variant because it also:
      - rejects negative aggregate totals
      - rejects non-static or non-positive declared output dimensions
      
      Code:
      
      ```python
      import math
      import onnx
      
      def _norm_profile_shape(shape):
          if shape is None:
              return None
          dims = tuple(int(d) for d in shape)
          return () if dims == (0,) else dims  # treat (0,) as scalar sentinel
      
      def _declared_output_specs(onnx_model):
          specs = []
          for value_info in onnx_model.graph.output:
              tt = value_info.type.tensor_type
              dims = tuple(int(dim.dim_value) for dim in tt.shape.dim)
      
              # Require fully static positive output dimensions.
              if not dims or any(d <= 0 for d in dims):
                  return None
      
              np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
              min_bytes = int(np_dtype.itemsize * math.prod(dims))
              specs.append((value_info.name, dims, min_bytes))
          return specs
      
      def _tensor_memsize(tensor):
          getter = getattr(tensor, "get_memsize", None)
          if not callable(getter):
              return None
          try:
              return int(getter())
          except Exception:
              return None
      
      def score_network(m, onnx_model):
          g = m.graph
          g.graph_reorder_nodes()
          g.shape_infer(None)
          g.profile()
      
          if not g.valid_profile:
              return None, None, None
      
          # Reject negative node-memory artifacts.
          for key in g.nodemap.keys():
              if getattr(g.nodemap[key], "memory", 0) < 0:
                  return None, None, None
      
          macs = int(sum(g.macs))
          memory = int(g.memory)
          params = int(g.params)
      
          # Reject negative aggregate totals as well.
          if macs < 0 or memory < 0 or params < 0:
              return None, None, None
      
          output_specs = _declared_output_specs(onnx_model)
          if output_specs is None:
              return None, None, None
      
          # Validate profiled outputs against declared static outputs.
          for name, declared_shape, min_bytes in output_specs:
              tensor = g.tensormap.get(name)
              if tensor is None:
                  return None, None, None
      
              profiled_shape = _norm_profile_shape(getattr(tensor, "shape", None))
              if profiled_shape != declared_shape:
                  return None, None, None
      
              memsize = _tensor_memsize(tensor)
              if memsize is None or memsize < min_bytes:
                  return None, None, None
      
          return macs, memory, params
      ```

    - **Mukundan** (2026-04-23T04:06:47.440Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      dynamic shapes don't necessarily mean 0 in shape for example, the following PoC allows intermediate operations to have near 0 cost:
      
      ```python
      import numpy as np
      import onnx
      from spox import Tensor, argument, build
      from spox.opset.ai.onnx import v24 as op
      
      
      def strip_identity(model):
          g = model.graph
          ident = next(n for n in g.node if n.op_type == "Identity" and n.output[0] == "output")
          src = ident.input[0]
          if src == "input":
              return model
          producer = next(n for n in g.node if src in n.output)
          producer.output[:] = ["output" if x == src else x for x in producer.output]
          g.node.remove(ident)
          return model
      
      
      i32 = lambda x: op.constant(value=np.asarray(x, dtype=np.int32))
      f32 = lambda x: op.constant(value=np.asarray(x, dtype=np.float32))
      
      inp = argument(Tensor('float32', [1, 10, 30, 30]))
      
      cell = op.slice(inp, i32([0, 0, 0, 0]), i32([1, 10, 1, 1]), axes=i32([0, 1, 2, 3]))
      v0v1 = op.cast(op.reduce_max(cell, keepdims=False), to=np.int32)
      v1v29 = op.add(op.mul(v0v1, i32([29])), i32([1]))
      v9v0 = op.add(op.mul(v0v1, i32([-9])), i32([9]))
      v29v0 = op.add(op.mul(v0v1, i32([-29])), i32([29]))
      
      shape = op.concat([v1v29, v1v29, v1v29, v1v29], axis=0)
      inp_ = op.slice(inp, i32([0, 0, 0, 0]), shape)
      
      # Intermediate ops here have near 0 cost, even though operating on full grid
      out = op.mul(inp_, f32([2]))
      
      pads = op.concat([i32([0, 0, 0, 0, 0]), v9v0, v29v0, v29v0], axis=0)
      out = op.pad(op.cast(out, to=np.int8), op.cast(pads, to=np.int64))
      
      model = build({"input": inp}, {"output": out})
      model = strip_identity(model)
      
      import onnx_tool
      model = onnx_tool.loadmodel(model)
      graph = model.graph
      graph.graph_reorder_nodes()
      graph.shape_infer(None)
      graph.profile()
      graph.print_node_map()
      ```

    - **Michael D. Moffitt** (2026-04-23T23:05:38.170Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      Wonderful, thank you both (and @jiweiliu) for the detailed feedback.  One final alternative to ensuring statically-defined graphs is the function below ... we'll carefully evaluate all of these options, and announce a decision soon.
      
      ```python
      def is_statically_defined(filename):
          graph = onnx.shape_inference.infer_shapes(onnx.load(filename)).graph
          for item in list(graph.input) + list(graph.value_info) + list(graph.output):
              if not item.type.HasField("tensor_type"): continue
              for dim in item.type.tensor_type.shape.dim:
                  if dim.HasField("dim_param") or not dim.HasField("dim_value"): return False
          return True
      ```

  - **Michael D. Moffitt** (2026-04-28T21:31:30.877Z, votes: {'canUpvote': True}):
    This issue should be now fixed (see also: [our metric update from today](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230)).  Specifically, we perform shape inference on all networks, and require all resulting tensors to have statically-defined shapes in order to be eligible for scoring.

- **MassimilianoGhiotto** (2026-04-22T06:29:56.710Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  Hi everyone, thanks for the update! Regarding the pinned versions, could you please clarify which opset_id range is supported by the scorer?  Thanks!

  - **Michael D. Moffitt** (2026-04-22T18:10:19.190Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    Our scorer itself imposes no limit on the opset_id range, aside from those supported implicitly by the tools we call.  So, if you find that a certain opcode behavior is implemented in both `onnxruntime` and `onnx-tool`, then you should be all set!

    - **(unknown)** (2026-04-23T08:53:49.847Z, votes: {}):
      (deleted)

- **hengck23** (2026-04-24T23:53:25.657Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  @mmoffitt 
  can you verify if lazy broadcast is allowed?  
  https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051   
  
  the ouput memory size is not counted by the profiler, although it is fixed shape and coded in python and shown in onnx graph (netron.app)

  - **Michael D. Moffitt** (2026-04-25T00:24:26.463Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Yep, just posted my [reply](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051#3448238).  Feel free to respond in that thread if we can offer more clarity.

- **Yiheng Wang** (2026-04-23T01:35:33.593Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  HI @mmoffitt thanks for the updates. May I know:
  > Sometime soon (likely tomorrow) we'll kick off a rescoring of all submissions
  
  If we have already rescored all submissions? Seems my 5900 score is still there (it should be 57XX or 56XX I think)

  - **Michael D. Moffitt** (2026-04-23T01:38:27.553Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
    Great question -- not yet, we're taking a closer look at two other issues first (one being the parameter contribution of `Constant` ops, another being the tolerance of dynamic shapes).  We'll follow up to this thread once we've settled on an appropriate resolution.

    - **Yiheng Wang** (2026-04-23T01:54:55.787Z, votes: {'totalVotes': 6, 'canUpvote': True, 'totalUpvotes': 6}):
      Thanks. FYI, the following are bugs that I found (not sure if duplicate) in onnx-tool 1.0.1:
      
      Two reproducible mismatches between `onnx-tool 1.0.1` and `onnxruntime` that may affect scoring:
      
      Environment:
      - onnx-tool 1.0.1
      - onnxruntime 1.24.4
      - opset 17
      
      ## 1. Heterogeneous-shape `Where` is profiled with the wrong output shape
      
      Expected:
      `onnx-tool` should infer/profile the same broadcasted output shape as ONNX Runtime.
      
      Observed:
      For `Where(cond, x, y)` with different input shapes, `onnx-tool` reports a smaller output shape than `onnxruntime`.
      
      Minimal repro:
      
      ```python
      import io, os, tempfile, contextlib
      import numpy as np
      import onnx, onnx.helper as oh, onnx.numpy_helper as onp
      import onnx_tool, onnxruntime as ort
      
      cond = np.zeros((1, 1, 9, 9), dtype=np.bool_)
      x = np.zeros((1, 5, 1, 9), dtype=np.uint8)
      y = np.zeros((1, 5, 1, 1), dtype=np.uint8)
      
      node = oh.make_node('Where', ['cond', 'x', 'y'], ['out'])
      graph = oh.make_graph(
          [node], 'where_bug', [],
          [oh.make_tensor_value_info('out', onnx.TensorProto.UINT8, None)],
          [onp.from_array(cond, 'cond'), onp.from_array(x, 'x'), onp.from_array(y, 'y')]
      )
      model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)], ir_version=10)
      
      fd, path = tempfile.mkstemp(suffix='.onnx')
      os.close(fd)
      onnx.save(model, path)
      
      with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
          mo = onnx_tool.loadmodel(path, {'verbose': False})
          mo.graph.graph_reorder_nodes()
          mo.graph.shape_infer(None)
          mo.graph.profile()
      
      print("onnx-tool:", tuple(mo.graph.tensormap['out'].shape))
      print("onnxruntime:", ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(['out'], {})[0].shape)
      ```
      
      Observed output:
      - onnx-tool: `(1, 1, 9, 9)`
      - onnxruntime: `(1, 5, 9, 9)`
      
      ## 2. Value-dependent shape inference is evaluated with zero-filled dummy inputs
      
      Expected:
      If an output shape depends on runtime tensor values, `onnx-tool` should not collapse it to zero during static shape inference.
      
      Observed:
      A shape-producing subgraph is evaluated with zero-filled dummy input, causing `ConstantOfShape` to be profiled as size 0 even though runtime output is nonzero.
      
      Minimal repro:
      
      ```python
      import io, os, tempfile, contextlib
      import numpy as np
      import onnx, onnx.helper as oh, onnx.numpy_helper as onp
      import onnx_tool, onnxruntime as ort
      
      input_vi = oh.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 1, 4, 4])
      
      zero = onp.from_array(np.array([0.0], dtype=np.float32), 'zero')
      axes = onp.from_array(np.array([0, 1, 2, 3], dtype=np.int64), 'axes')
      unsq_axes = onp.from_array(np.array([0], dtype=np.int64), 'unsq_axes')
      fill = onp.from_array(np.array([1.0], dtype=np.float32), 'fill')
      
      nodes = [
          oh.make_node('Greater', ['input', 'zero'], ['gt']),
          oh.make_node('Cast', ['gt'], ['gt_i64'], to=onnx.TensorProto.INT64),
          oh.make_node('ReduceSum', ['gt_i64', 'axes'], ['n'], keepdims=0),
          oh.make_node('Unsqueeze', ['n', 'unsq_axes'], ['shape1']),
          oh.make_node('ConstantOfShape', ['shape1'], ['out'], value=fill),
      ]
      
      graph = oh.make_graph(
          nodes, 'bug2_zero_fill',
          [input_vi],
          [oh.make_tensor_value_info('out', onnx.TensorProto.FLOAT, None)],
          [zero, axes, unsq_axes]
      )
      model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)], ir_version=10)
      
      fd, path = tempfile.mkstemp(suffix='.onnx')
      os.close(fd)
      onnx.save(model, path)
      
      with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
          mo = onnx_tool.loadmodel(path, {'verbose': False})
          mo.graph.graph_reorder_nodes()
          mo.graph.shape_infer(None)
          mo.graph.profile()
      
      print("onnx-tool:", tuple(mo.graph.tensormap['out'].shape))
      print("onnxruntime:", ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(
          ['out'], {'input': np.ones((1, 1, 4, 4), dtype=np.float32)}
      )[0].shape)
      ```
      
      Observed output:
      - onnx-tool: `(0,)`
      - onnxruntime: `(16,)`
      
      Thanks.

- **Durga Kumari** (2026-04-22T17:40:26.790Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thank you for the detailed update and for actively incorporating community feedback

- **Navneet** (2026-04-25T06:03:18.150Z, votes: {'canUpvote': True, 'totalUpvotes': 1}):
  Thanks for the neuroGolf Update @mmoffitt

- **Kawchar Husain** (2026-04-28T11:27:17.883Z, votes: {'canUpvote': True}):
  Hello @mmoffitt ,
  
  I have a question about the Longest Leader.
  
  The competition description says it is awarded to the team holding 1st place for the longest period between April 16, 2026 12:00 AM UTC and July 15, 2026 11:59 PM UTC.
  
  However, there was a scoring update on April 21, and you also mentioned on April 24 that another update may happen next week.
  
  Could you please clarify which time period will be counted for the Longest Leader award?
  
  Will it count from the original start date, April 16, or will it restart/count from after the next scoring update, or use some other period?
  
  Thanks!

- **(unknown)** (2026-04-23T05:55:06.747Z, votes: {'totalVotes': 1, 'totalUpvotes': 3}):
  (deleted)

- **@🤞@** (2026-04-22T19:36:21.467Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  thank for the consideration 🙏

- **Svanik Kolli** (2026-04-21T23:38:49.310Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  THANK YOU SM
