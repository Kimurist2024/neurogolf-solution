# Should we pause further optimization until there is a data/eval update or a full rescoring?

- Topic ID: 693589
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693589
- Author: Tony Li (@tonylica)
- Posted: 2026-04-21T16:30:35.745644400Z
- Votes: 11
- Total messages: 5

## Body

Update: the host has already shipped an evaluation update

https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711

--------------------------------------
At this point, we already know there are several bugs, or at least likely bugs, that could be materially distorting scores. Any future data update or evaluation-code change could significantly alter both the current results and the direction of optimization.

My current score also does not yet include the over-grid tasks, and it may not be affected by known scoring issues listed below, so I am not too concerned about the score at this stage. 

What concerns me more is that the LLM is now repeatedly pushing further optimization by converting more logic into Constant nodes to gain score. If Constant memory is later counted, some of these newer optimizations may turn out to reduce score rather than improve it.

If possible, I would also like a full rescoring of the entire submission history. That would at least preserve visibility into earlier strong submissions, even if later optimizations turn out to be worse under corrected evaluation.

Relevant post and point:

1. **Constant nodes do not appear to be counted toward memory usage**
   It seems that a `Constant` node’s parameters and memory are only counted at the node where the constant is actually consumed.

2. **Other issues in `onnx-tool`**
https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827

3. **The highest single-task score is 25**
https://www.kaggle.com/competitions/neurogolf-2026/discussion/693247

4. **ONNX Runtime compatibility**
https://www.kaggle.com/competitions/neurogolf-2026/discussion/693088

5. **Over-30×30 grid tasks**
https://www.kaggle.com/competitions/neurogolf-2026/discussion/692621
   There are 6 oversize-grid tasks: `021`, `055`, `080`, `184`, `202`, and `366`.

## Comments (5)

- **hengck23** (2026-04-23T05:02:12.980Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  training token is expensive!!!!

- **Michael D. Moffitt** (2026-04-22T23:09:13.347Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Just to confirm: for issue #1, it sounds like you're asking for a change similar to the following:
  
  ```python
  def score_network(m):
    model = onnx_tool.loadmodel(m, {'verbose': False,
                                    'constant_folding': True}) # <--- this line is new
    ...
  )
  ```
  
  Let me know if that's consistent with your expectation.

  - **Tony Li** (2026-04-23T04:06:22.050Z, votes: {'canUpvote': True}):
    The main issue is still the onnx-tool profiler vs ONNX Runtime mismatch.
    
    The inflated-score cases reported so far are mostly in that family — e.g. heterogeneous-shape Where, value-dependent ConstantOfShape, and the task179 dynamic-Pad case where ORT returns [1,10,30,30] but the profiler sees a much smaller output. Since scoring uses the profiler path, that undercounts memory and inflates score.
    
    So for that class of bug, add a direct check after profiling: require each profiled output tensor to match the declared ONNX output shape and minimum required byte size. That should block this whole mismatch family directly.
    
    For the separate Constant-op param count:  constant_folding=True is not a fix. It does recover the simple used Constant -> Add case, but it still misses unused serialized Constants and constant-only output graphs. In one all-constant graph it also folded so aggressively that the profiled cost effectively dropped to zero.
    
    The preference would be to keep profiling the original graph, keep constant_folding=False, and add an explicit pass over original-model Constant nodes, along the lines of the code below. That said, this may be more invasive than it looks and could have unintended consequences, and we have not had a chance to test it thoroughly.
    
    ```python
    import onnx
    from onnx import numpy_helper as onp
    
    def _tensor_proto_numel(tensor_proto: onnx.TensorProto) -> int:
        n = 1
        for d in tensor_proto.dims:
            n *= int(d)
        return int(n)
    
    def _tensor_proto_payload_bytes(tensor_proto: onnx.TensorProto) -> int:
        if tensor_proto.raw_data:
            return len(tensor_proto.raw_data)
        if tensor_proto.data_type == onnx.TensorProto.STRING:
            return sum(len(x) for x in tensor_proto.string_data)
        return int(onp.to_array(tensor_proto).nbytes)
    
    def extra_constant_param_elems(model: onnx.ModelProto) -> int:
        extra = 0
        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    extra += _tensor_proto_numel(attr.t)
                elif attr.name == "sparse_value" and attr.HasField("sparse_tensor"):
                    # count stored values, not dense logical size
                    extra += _tensor_proto_numel(attr.sparse_tensor.values)
        return extra
    
    def extra_constant_payload_bytes(model: onnx.ModelProto) -> int:
        extra = 0
        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    extra += _tensor_proto_payload_bytes(attr.t)
                elif attr.name == "sparse_value" and attr.HasField("sparse_tensor"):
                    st = attr.sparse_tensor
                    extra += _tensor_proto_payload_bytes(st.values)
                    extra += _tensor_proto_payload_bytes(st.indices)
        return extra
    
    ```
    Then in scoring:
    
    ```python
    macs, memory_bytes, params = score_network(model_path, model)
    
    # keep current onnx_tool profile, then patch in Constant attrs from original model
    params += extra_constant_param_elems(model)
    
    cost = max(1, int(macs) + int(memory_bytes) + int(params))
    ```
    
     ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22938014%2Fc29c55721712d0ad63d6a96f92c40023%2F1.png?generation=1776917353709065&alt=media)

- **Michael D. Moffitt** (2026-04-28T21:34:53.997Z, votes: {'canUpvote': True}):
  Issue #1 should now be fixed!  See also: [our metric update from today](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230).

- **(unknown)** (2026-04-21T18:09:41.547Z, votes: {}):
  (deleted)
