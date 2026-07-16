# 20+ single-task scores vs the ~15.9 full-output floor — intended, or still a scoring bug?

- Topic ID: 693950
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693950
- Author: Tony Li (@tonylica)
- Posted: 2026-04-22T19:56:18.276727500Z
- Votes: 10
- Total messages: 22

## Body

After the 04/21 metric update, Have seen kaggler reports of `20.13`, multiple `20+` tasks, multiple `18.5+` tasks, and even `23.27`.

That stands out because if a model really pays for a full `1x10x30x30`  output in new metric, the rough floor is already around:

* `9000` cost → score `15.895020`

By comparison:

* `18.5` → cost ≈ `665`
* `20.13` → cost ≈ `130`
* `23.27` → cost ≈ `5.64`

So `20+` is not just a better `15.9`. It suggests a very different cost/accounting regime.

My own best-score model is still below that ~`15.9` full-output floor, so before spending time chasing `20+`, I wanted to discuss whether this is intended, or whether another eval/scorer update may still happen.

Possible areas may inflate reported scores for some solutions:

* constants or initializers may not be contributing to memory cost as expected
* dynamic shapes appear to under-report memory cost
* the rule requiring statically-defined shapes does not seem to be explicitly enforced in the scorer

Could we get clarify:

* Are these `20+` scores intended under the current metric?
* Or is another eval/scoring update possible?

**Inflation probe :**

I built a dynamic-pad probe for `task179.onnx` based on the approach described here:

```text
https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711#3446797
```

The probe passes the Kaggle scorer for Task 179. It scores **18.139**, with a reported cost of **954** and a file size of **716 bytes**.

Relative to the current score, the net gain is **+2.937 score**.

This appears to be a genuine score inflation issue in the current scorer. For Task 179, all official local examples are `3x3`, so the crop-to-`3x3` trick is semantically valid for this task. The current scorer seems to enforce only static declared graph input/output shapes, while the actual cost is still derived from `onnx_tool.shape_infer(None) -> profile()`.

In this probe, ORT returns a shape of `[1,10,30,30]`, but `onnx_tool` profiles the `Pad` output as `[1,10,3,3]`. As a result, memory is significantly undercounted, with the reported profile showing `memory=846`, `macs=91`, and `params=17`.

I also tested @asalhi Ali’s approach (`[1, 10, None, None]`) on task 337, which scored 20.13. In that case, `onnx_tool` profiles the `Pad` output as `[1,10,0,0]`, which causes the memory cost to be severely undercounted.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22938014%2Faaf3e5e81cba4d5ec4ced9e371f7eef1%2F2222.png?generation=1776893352298328&alt=media)

## Comments (22)

- **Ali** (2026-04-22T20:31:01.233Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  I shared a notebook that generated 20.13 for task 337 
  You can find it in the code tab. 
  
  Attached is the main part: 
  
  ```python
  import onnx
  from onnx import helper, TensorProto, numpy_helper
  import numpy as np
  
  # Input tensor: float32[1, 10, ?, ?]
  input_tensor = helper.make_tensor_value_info(
      "input",
      TensorProto.FLOAT,
      [1, 10, None, None]
  )
  
  # Output tensor: float32[1, 10, ?, ?]
  output_tensor = helper.make_tensor_value_info(
      "output",
      TensorProto.FLOAT,
      [1, 10, None, None]
  )
  
  # Constant indices for Gather
  idx_values = np.array([0, 1, 2, 3, 4, 8, 6, 7, 5, 9], dtype=np.int64)
  idx_initializer = numpy_helper.from_array(idx_values, name="idx")
  
  # Single Gather node
  gather_node = helper.make_node(
      "Gather",
      inputs=["input", "idx"],
      outputs=["output"],
      axis=1
  )
  
  # Build graph
  graph = helper.make_graph(
      nodes=[gather_node],
      name="t_337",
      inputs=[input_tensor],
      outputs=[output_tensor],
      initializer=[idx_initializer]
  )
  
  # Build model
  model = helper.make_model(
      graph,
      opset_imports=[helper.make_operatorsetid("", 13)]
  )
  
  # Match IR version seen in your file
  model.ir_version = 13
  
  # Save
  onnx.save(model, "task337.onnx")
  
  print("Saved task337.onnx")
  ```

  - **(unknown)** (2026-04-22T20:37:12.470Z, votes: {'totalVotes': 2, 'totalUpvotes': 2}):
    (deleted)

    - **Ali** (2026-04-22T20:39:04.620Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      You can score even 20.50 with dtype=np.int32 
      
      I am not sure if it's allowed, but it scores!
      @mmoffitt

  - **Russell Kirk** (2026-04-22T20:38:38.487Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    I think we're supposed to use the [1,10,30,30] tensor and not dynamic shapes.  From my understanding, your submission isn't allowed under the rules.  I may be wrong.

    - **Russell Kirk** (2026-04-22T20:39:21.783Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      It's allowed in ONNX, but the "Overview" asks us to use [1,10,30,30].   /// EDIT: I'm trying to find where I inferred that dynamic shapes aren't allowed, and I cannot find it right now.  So take what I said with a grain of salt.  I'll change this when I find out for sure. /// FOLLOWUP: In the attached utilities, it **seems** to only properly score if you're using [1,10,30,30].   So take that to mean what you want.

    - **Ali** (2026-04-22T20:43:27.517Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Then the score will become: 14.50 😐 
      This should be addressed in the metric (because currently it scores)

    - **(unknown)** (2026-04-22T20:43:32.837Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
      (deleted)

    - **Russell Kirk** (2026-04-22T21:03:54.977Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      I'm not a deciding voice. Mukundan already pointed it out to the host.  I believe the host will likely address it.

  - **Michael D. Moffitt** (2026-04-28T21:36:51.813Z, votes: {'canUpvote': True}):
    As of [today's update](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230), statically-defined shapes are now strictly enforced!  Thank you @asalhi + @robga + @russcore + @mukundan314 for the detailed feedback.

- **hengck23** (2026-04-23T05:14:19.197Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  a do nothing score is about 14.5 (cost is only memory for 1x10x30x30 float32)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fa00e62be24e2f6b9220a528998766ca8%2FSelection_3040.png?generation=1776921215342488&alt=media)
  
  anything more than that is cheating????

  - **Russell Kirk** (2026-04-23T05:47:50.960Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Did you try it as a boolean?

    - **hengck23** (2026-04-23T06:12:12.530Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      I though the rule is float32 as input and output?

    - **Russell Kirk** (2026-04-23T06:17:04.260Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Someone can correct me if I'm wrong, I'm still learning, [1,10,30,30] describes the dimensions of the tensor whereas float vs boolean describes the contents. I thought the requirements were calculations regarding shape not contents.  Again, I'm not very sure.  I will wait until everything is declared until I continue further.

    - **hengck23** (2026-04-23T06:56:35.407Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      i think this needs clarification from the host becuase i see from neurogolf_utils.py:
      
      ```
      _DATA_TYPE = onnx.TensorProto.FLOAT
      _EXCLUDED_OP_TYPES = ["LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"]
      _FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
      _GRID_SHAPE = [_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH]
      _IR_VERSION, _OPSET_IMPORTS = 10, [onnx.helper.make_opsetid("", 10)]
      ```
      
      i though these are  parameters we cannot change:
      - input,ouput shape and size, name
      - opset and ir version

    - **(unknown)** (2026-04-23T07:16:13.727Z, votes: {'totalVotes': 3, 'totalUpvotes': 3}):
      (deleted)

    - **Michael D. Moffitt** (2026-04-23T15:22:37.293Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      Yes, to clarify further: you're welcome to ignore the specific values of `_IR_VERSION` and `_OPSET_IMPORTS` that we happened to use in `neurogolf_utils.py`.  We found that those settings were sufficient to create our helper function `single_layer_conv2d_network()`, but you may have better luck with more recent versions.

  - **Tony Li** (2026-04-23T17:45:04.577Z, votes: {'canUpvote': True}):
    > a do nothing score is about 14.5 (cost is only memory for 1x10x30x30 float32)
    > 
    > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fa00e62be24e2f6b9220a528998766ca8%2FSelection_3040.png?generation=1776921215342488&alt=media)
    > 
    > anything more than that is cheating????
    
    If the `1×10×30×30` output is enforced correctly, then a single-task score of `15.895020` would likely be near the ceiling. That would put the maximum total score at roughly `6358`, in my view. Under that scenario, the top gold range would probably be around `6300 - 6340`, 
    
    Each leaderboard rank could be separated by only about 1 point, or even show the same displayed score despite tiny underlying differences.

    - **hwe owe** (2026-04-25T09:28:44.457Z, votes: {'canUpvote': True}):
      Can I be teammate with you?if we ensemble our submission,we might be champoin

- **Geremie Yeo** (2026-04-25T06:27:02.427Z, votes: {'canUpvote': True}):
  currently 15.827

- **hengck23** (2026-04-25T01:13:58Z, votes: {'canUpvote': True}):
  With host confirmation https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051#3448238 limit is not long 15.9. It is 16 or 17

- **(unknown)** (2026-04-22T20:34:00.350Z, votes: {'totalVotes': 2, 'totalUpvotes': 2}):
  (deleted)

- **(unknown)** (2026-04-22T20:11:29.810Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
  (deleted)
