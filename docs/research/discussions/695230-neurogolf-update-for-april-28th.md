# NeuroGolf Update for April 28th

- Topic ID: 695230
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230
- Author: Michael D. Moffitt (@mmoffitt)
- Posted: 2026-04-28T21:12:54.257834Z
- Votes: 12
- Total messages: 22
- Pinned: yes

## Body

Thanks again to all teams for their helpful and detailed feedback.  As [telegraphed](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711#3448225) last week, we're rolling out a metric update to ensure competition rules are strictly and fairly enforced:
- We now correctly incorporate parameter contributions from **constant values**, addressing an [issue](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827#3445818) [reported](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693589) by @mukundan314 and @tonylica.
- Our [constraint](www.kaggle.com/competitions/neurogolf-2026/overview/constraints) requiring **statically-defined shapes** is now explicitly enforced in response to [discoveries](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693950#3447198) [made](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711#3446797) by @asalhi and @mukundan314.  In particular, networks that are invalid or contain dynamic shapes—e.g., symbolic dim parameters or missing dimension values after [shape inference](https://onnx.ai/onnx/api/shape_inference.html)—should now yield zero points as originally stated (you can use our [starter notebook](https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-championship) to perform this same check locally).
- To insulate contestants against [profiling](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711#3447354) [idiosyncrasies](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051#3448342), our **memory footprint calculation** is now just a simple sum over the bytes consumed by all static shapes (excluding input & output layers).

As always, we'll continue to listen to community feedback and will provide further clarity if needed.  The batch rescoring process should be starting soon!

---

***[Refer to the bottom of our [welcome message](https://www.kaggle.com/competitions/neurogolf-2026/discussion/691461) for a complete list of all contest updates]***

## Comments (22)

- **hengck23** (2026-04-30T20:47:15.720Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
  i suggest to add version tag in the file "neurogolf_utils.py". This helps kagglers to keep track of changes and also faciliate discussion that follows
  
  ```
  NEUROGOLF_FILE_VERSION='28-apr.00'
  ```

  - **Michael D. Moffitt** (2026-05-01T17:32:23.710Z, votes: {'canUpvote': True}):
    Yes, and in addition to that, we can create a section toward the top listing all the changes so far (along with the dates that we made them).
    
    Thanks for the suggestion!

- **Michael D. Moffitt** (2026-04-29T18:38:37.313Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  A quick update for 4/29:
  - Yesterday's LB batch rescoring process has completed!
  - Many thanks to @pavelsavchenkov + @robga + @cudacoding + @asalhi for discovering & sharing two [new](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694754#3450241) [exploits](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694541#3449839).  These issues affect fewer than four or five teams at present, but do need to be addressed.
  
  The fixes are seemingly straightforward, so we're sharing them here (to solicit community feedback) before making any changes:
  
  ```
  _EXCLUDED_OP_TYPES = ["LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION",
                        "COMPRESS"]  # <--- this op is new
  ...
  
  def calculate_memory(filename):
      ...
      init_names = {init.name for init in graph.initializer}
      io_names = {t.name for t in list(graph.input) + list(graph.output)}
      if io_names.intersection(init_names): return None
      if model.functions: return None
      for opset in model.opset_import:
          if opset.domain not in {"", "ai.onnx"}: return None
      for node in graph.node:
          for attr in node.attribute:
              if attr.type in [onnx.AttributeProto.GRAPH,
                               onnx.AttributeProto.GRAPHS]:
                  return None
  ```
  
  Also, since `math.log()` currently fails for the handful of tasks that can be solved at zero cost, we're planning a change that gives full credit (25 points) to such solutions.

  - **Pavel** (2026-04-30T14:05:05.680Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Yes, I think the proposed change fixes 2 linked bugs. Thank you!

  - **Pavel** (2026-04-30T16:28:24.257Z, votes: {'canUpvote': True}):
    Even with this fix, I think there is another scorer issue derived from the already familiar `Tensor.get_numpy()` behavior: when runtime values are unknown, `onnx-tool` fabricates zeros.
    
    You can insert a `Compress` gate at the beginning of the graph so `onnx-tool` infers a zero-volume tensor and then profiles all downstream computation with batch size 0. This can heavily undercount MACs. The gate adds some memory cost, but the MAC undercount can be larger.
    
    To keep the host static-shape check happy, the model can also include a `value_info` entry for the gated tensor with its real non-zero shape.
    
    Attached are two task-1 submissions. They solve the same task and differ only by this input gate. The gated version scores lower cost / higher points.
    
    Cost:
    * baseline: `253982`  
    * bugged: `118818`
    
    These are not absolute minimum cost for task 1. The intention is to show how to lower the cost of a honest model by exploiting `onnx-tool` and `onnx`.
    
    Profiling MACs for the bugged model:
    
    | item | node count | reported MACs by scorer | true / expected MACs | wrong? | comment |
    | --- | ---: | ---: | ---: | --- | --- |
    | `_gate_reduce_max` | 1 | 9000 | 9000 | no | Gate cost is charged correctly. |
    | `_gate_greater` | 1 | 1 | 1 | no | Gate cost is charged correctly. |
    | `_gate_compress` | 1 | 0 | 0 | no | The issue is not `Compress` MACs; it is the zero-batch shape inferred after it. |
    | downstream `Conv` | 2 | 0 | 180000 | yes | These identity convs execute at runtime, but `onnx-tool` profiles them with batch size 0. |
    | downstream `Mul` | 1 | 0 | 81 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
    | downstream `ReduceMax` | 1 | 0 | 81 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
    | downstream `Sub` | 1 | 0 | 9 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
    | total | 8 | 9001 | 189172 | yes | Reported total only includes the gate; expected total includes gate plus downstream computation. |

  - **Pavel** (2026-04-30T16:35:53.860Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    One more thing about MACs accounting in `onnx-tool`.
    
    It looks like zero MACs for `ArgMax`, `TopK`, and `OneHot` in `onnx-tool` is not intended.
    
    `onnx-tool` does not use MACs as literal multiply-adds only. Its own profiler comments describe an instruction-style cost model, and it defines `CMP_MACS = 1` for comparisons [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L9-L22)].
    
    Comparable non-multiply-add ops are already charged: `Less` [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L721-L733)], `Equal` [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L2231-L2237)], `Greater` [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L2434-L2441)], `ReduceMax` via `ReduceMinNode.profile()` [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L1924-L1949)], and `MaxPool` [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L1394-L1414)].
    
    The zero cost happens because the default `Node.profile()` returns `[0, 0]` [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L212-L213)], and these classes implement inference but do not override `profile()`:
    - `ArgMaxNode` computes `np.argmax`, then the next class starts with no profile override [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L1627-L1640)].
    - `TopKNode` computes top-k selection, then the next class starts with no profile override [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L1953-L1980)].
    - `OneHotNode` uses helper code based on equality comparison, but also has no profile override [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f1e9d31498f4bf40898bd1143aab9ac3db/onnx_tool/node.py#L1030-L1070)].
    
    So under current `onnx-tool` policy, these should have non-zero comparison/selection cost; they end up zero only because their `profile()` methods are missing.
    
    Do you think the scorer should keep this behaviour?

    - **Michael D. Moffitt** (2026-04-30T19:02:43.240Z, votes: {'canUpvote': True}):
      +1 to fixing the `Compress` issue (see our latest update above).  It introduces the same sorts of issues as `NonZero` and `Unique`.
      
      As for the other corner cases: there will definitely be bugs that teams can leverage to bring their MAC counts down (perhaps by a lot).  In the interest of providing a stable metric for the remainder of the contest, we're willing to tolerate these edge cases, and will only take action in the event of a truly catastrophic exploit.

    - **Pavel** (2026-04-30T19:14:04.727Z, votes: {'canUpvote': True}):
      Zero MACs in `onnx-tool` by design: I see, thanks!
      
      Dynamic gate issue: it is actually not `Compress` specific.
      
      The same pattern can be for example reproduced with a standard `Slice` gate at the beginning of the graph: `ReduceMax(input) -> Greater(..., 0) -> Cast(INT64) -> Slice(input, end)`.
      
      Cost:
      * baseline: `409489`
      * "optimized": `127807`
      
      Profiling MACs for the "optimized" model:
      
      | item | node count | reported MACs by scorer | true / expected MACs | wrong? | comment |
      | --- | ---: | ---: | ---: | --- | --- |
      | `_slice_gate_reduce_max` | 1 | 9000 | 9000 | no | Gate cost is charged correctly. |
      | `_slice_gate_greater` | 1 | 1 | 1 | no | Gate cost is charged correctly. |
      | `_slice_gate_cast_end` | 1 | 0 | 0 | no | Cast has no MAC cost under the current policy. |
      | `_slice_gate_slice` | 1 | 0 | 0 | no for MACs, yes for shape | The issue is not `Slice` MACs; it is the zero-batch shape inferred after it. |
      | downstream `Conv` | 2 | 0 | 180000 | yes | These identity convs execute at runtime, but `onnx-tool` profiles them with batch size 0. |
      | downstream `ReduceSum` | 1 | 0 | 81 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
      | downstream `ConvTranspose` | 2 | 0 | 145800 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
      | downstream `Sub` | 1 | 0 | 9 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
      | downstream `Add` | 1 | 0 | 810 | yes | Runtime batch is 1; scorer profiles zero-volume output. |
      | total | 14 | 9001 | 335701 | yes | Reported total only includes the gate; expected total includes gate plus downstream computation. |

    - **CPMP** (2026-05-01T08:05:21.733Z, votes: {'totalVotes': 8, 'canUpvote': True, 'totalUpvotes': 8}):
      > As for the other corner cases: there will definitely be bugs that teams can leverage to bring their MAC counts down (perhaps by a lot). In the interest of providing a stable metric for the remainder of the contest, we're willing to tolerate these edge cases, and will only take action in the event of a truly catastrophic exploit.
      
      People won't share loopholes anymore if they aren't fixed. They will keep their advantage private. Is that what you want?

    - **NNMax** (2026-05-01T09:08:40.227Z, votes: {'totalVotes': 7, 'canUpvote': True, 'totalUpvotes': 7}):
      > As for the other corner cases: there will definitely be bugs that teams can leverage to bring their MAC counts down (perhaps by a lot).  In the interest of providing a stable metric for the remainder of the contest, we're willing to tolerate these edge cases, and will only take action in the event of a truly catastrophic exploit.
      
      Doesn't it defeat the whole purpose of competition? It's becoming more of a bug bounty rather than true optimization and if you consider allowing edge cases, it'll severely discourage many others who are trying to optimize without any bug exploits.

    - **Michael D. Moffitt** (2026-05-01T15:07:08.703Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
      +1 to @cpmpml and @ashok205.  We do indeed want contestants to feel comfortable sharing potential loopholes with us, and to ensure that teams can do this without revealing valuable techniques, we've just set up `neurogolf.2026@gmail.com` to allow direct and private communication with the organizers.  We'll also pin that address to our [welcoming announcement](https://www.kaggle.com/competitions/neurogolf-2026/discussion/691461).
      
      Thanks for sharing your feedback — please keep it coming!

- **theredbluepill** (2026-05-01T16:09:12.487Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thanks. I really thought it was going end-game like this soon, 10K score.

- **Durga Kumari** (2026-05-01T16:06:15.803Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thanks for the clarification and transparency also this makes the competition much fairer.

- **Chan Kha Vu** (2026-04-29T00:19:26.257Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  **Let the new round of reward hacking begin!**
  
  Time to make our LLMs anxious y'all
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2270821%2F1212960a053511f129bf726fe8136e76%2F03270c4c700148ecb731be926ad62df95a996582-1522x869.webp?generation=1777421952260400&alt=media)

- **prvi** (2026-04-29T11:57:34.797Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Some task has zero sum for mac's memory and parameter count after excluding the input output tensor sizes. Are they handled properly? I got an error.

  - **Michael D. Moffitt** (2026-04-29T18:41:38.070Z, votes: {'canUpvote': True}):
    Great question -- yes, there are one or two tasks that can be solved at zero cost, which (ideally) should yield a full score of 25 points.  We'll have an update about this soon.

  - **Michael D. Moffitt** (2026-05-01T15:10:54.260Z, votes: {'canUpvote': True}):
    This issue should be fixed as of yesterday's update.  Specifically: if teams manage to find a legitimate zero-cost solution to a task, it will earn them the full 25 points.

- **hengck23** (2026-04-28T22:08:08.707Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  how to we use the new profiler in our local PC environment? download the new neurogolf_utils.py, install new version of onnx-tool from github or ????

  - **Michael D. Moffitt** (2026-04-28T22:15:04.383Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    As long as you're using our newest [public dataset](https://www.kaggle.com/competitions/neurogolf-2026/data) (which has the latest `neurogolf_utils.py`), you should be all set.  As for `onnx-tool`, v1.0.1 is OK -- these updates don't require any new versions of that.
    
    Please feel free to reach out if you need any additional help reproducing these scores locally!

- **(unknown)** (2026-04-29T13:39:30.003Z, votes: {}):
  (deleted)

- **(unknown)** (2026-04-28T21:45:49.413Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
  (deleted)

  - **Michael D. Moffitt** (2026-04-28T21:48:47.490Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Thanks for the question -- there are no explicit changes to allowed Ops, as long as they survive the "shape inference" pass that is now performed before network scoring.
