# LB 9550 Potential scoring loophole privately reported to hosts [Kaggle Agent]

- Topic ID: 696365
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/696365
- Author: Jiwei Liu (@jiweiliu)
- Posted: 2026-05-02T00:36:18.796345300Z
- Votes: 23
- Total messages: 2

## Body

Hi all,

Our agent found a scoring loophole that can substantially inflate leaderboard scores without representing a real modeling improvement.

We have emailed the competition hosts neurogolf.2026@gmail.com with all the details.
We are intentionally not publishing the implementation details or code here because it might trigger a lot of submissions to exploit it, and make rescoring much slower. But if host wants to get feedback about how to fix it, we can share it all here too. Just let us know. @mmoffitt

Best

## Comments (2)

- **Jiwei Liu** (2026-05-03T18:14:31.433Z, votes: {'totalVotes': 6, 'canUpvote': True, 'totalUpvotes': 6}):
  The issue is in memory accounting: `calculate_memory()` sums tensor sizes from `graph.value_info`, but unused `value_info` entries can contain negative `dim_value`s. These entries are ignored by ONNX Runtime during inference, but they reduce the scorer’s computed memory.
  
  Example from `task005.onnx`:
  
  ```text
  Unused value_info:
    _r7_unused_negative_memory_task005
    UINT8 shape [-139737]
  ```
  
  This tensor is not connected to any node, but the scorer includes it in memory:
  
  ```text
  macs   = 82700
  memory = -82906
  params = 207
  cost   = 1
  score  = 25 - ln(1) = 25
  ```
  
  So the model receives a perfect cost score even though the actual graph is not a 1-cost model. I also confirmed this can be applied broadly by adding unused positive/negative `value_info` entries to force `macs + memory + params == 1`.
  
  Suggested fixes:
  
  1. Reject any tensor shape with `dim_value <= 0` in `input`, `output`, or `value_info`.
  2. Ignore unused `value_info` entries when calculating memory.
  3. After calculating memory, reject `memory < 0` and `macs + memory + params <= 0`.
  4. Prefer deriving memory only from tensors actually produced/consumed by graph nodes after shape inference.
  
  `onnx.checker.check_model()` alone is not sufficient here, since the negative unused `value_info` passed validation in this case.

- **Michael D. Moffitt** (2026-05-03T17:37:27.510Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Thank you @jiweiliu!  The exploit your team shared with the organizers will indeed be fixed in the next update, so feel free to share it here so that other teams can better understand the specific vulnerability we'll be patching.
