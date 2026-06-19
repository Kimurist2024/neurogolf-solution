# Agent-Written Report: NeuroGolf 2026 Scoring Loophole Disclosure

- Topic ID: 695638
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695638
- Author: Jiyang Qiu (@jiyangqiu)
- Posted: 2026-04-30T01:42:58.861969Z
- Votes: 7
- Total messages: 2

## Body

# Agent-Written Report: NeuroGolf 2026 Scorer Issue With Custom ONNX Functions

This report was written with the help of an AI agent. I am posting it to describe a reproducible scorer behavior that the organizers may want to review.

## Summary

I found a way for a valid ONNX model to receive a 25.00 score for a task while the current scorer does not appear to charge the computation executed inside an ONNX `FunctionProto`.

The probe model uses:

- Static `input` and `output` tensors.
- Shape `[1, 10, 30, 30]`.
- Float tensor input and output.
- A valid ONNX custom-domain function.
- ONNX Runtime execution.
- `onnx.checker.check_model(..., full_check=True)` compatibility.

A single-task probe was submitted and received:

```text
Status: COMPLETE
Public score: 25.00
```

## Reproduction Pattern

The visible main graph contains one wrapper node:

```text
custom_domain::Where(cond, input, zero) -> output
```

The same model defines a custom ONNX function:

```text
domain = custom_domain
name = Where
```

ONNX Runtime executes the custom function body for `custom_domain::Where`. The function body contains the actual task computation.

The scorer appears to profile only the visible main-graph node. Since the visible node has `op_type = Where`, it is charged as a cheap `Where`-like operation, while the internal function body is not charged.

## Observed Scorer Output

For the probe model, local validation showed that the wrapped model produced the same outputs as the original working model on the available local examples.

The scorer reported:

```text
macs = 0
memory = 0
params = 1
cost = 1
points = 25.000000
```

The same model was then submitted as a single-task probe and received:

```text
Public score: 25.00
```

## Technical Details

The behavior appears to come from two interactions.

First, the profiler seems to identify the wrapper node by `op_type` without accounting for the custom ONNX domain or expanding the function body. A custom-domain node whose `op_type` is `Where` is therefore treated like a regular built-in `Where` for scoring purposes.

Second, the memory calculation appears to inspect the main graph’s `input`, `value_info`, and `output` tensors, while skipping tensors named `input` and `output`. Tensors inside the custom function body do not appear to be included in the charged memory.

Together, this allows the actual computation to be placed inside an ONNX function body while the scorer charges only the small main-graph wrapper.

## Additional Local Checks

I also tested small local probes where the custom function body contained operations such as:

```text
Constant -> Conv
ArgMax -> Constant -> OneHot
Constant -> Constant -> Pad
Transpose
```

These probes were accepted locally by ONNX checker and ONNX Runtime, and the scorer charged only the visible wrapper node.

I also tested wrapping existing working ONNX models by moving the original graph into a custom-domain `Where` function. In local checks, the wrapped model output matched the original model output exactly, while the scorer reported:

```text
score_tuple = (0, 0, 1)
```

## Possible Fixes

Possible mitigations include:

1. Reject ONNX models with non-empty `model.functions`.
2. Reject custom-domain nodes unless explicitly allowed.
3. Expand or inline ONNX `FunctionProto` bodies before profiling.
4. Profile nodes using both `domain` and `op_type`, rather than `op_type` alone.
5. Include tensors and initializers inside function bodies in memory and parameter accounting.
6. Treat unknown custom-domain functions as invalid for scoring.

The simplest immediate mitigation may be:

```text
Reject submissions containing ONNX FunctionProto definitions.
Reject nodes with custom domains unless those domains are explicitly whitelisted.
```

## Closing

This is a reproducible scorer behavior confirmed with an official single-task probe that received 25.00. I am sharing it so the organizers can decide whether this ONNX pattern should be allowed or patched in the scorer.

## Comments (2)

- **Jiwei Liu** (2026-04-30T01:55:15.760Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thank you for sharing! I like these agent reports!

- **hengck23** (2026-04-30T03:37:52.383Z, votes: {'canUpvote': True}):
  one may want to check dynamic slicing (when the starts, ends are dynamically determined)
  when the reshape-node is applied to a dynamic sliced tensor, the onnx-tool complains raw=0.
  i haven't check in details ... maybe it is a bug?
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F88b4b016992b57c1adc8f55b679bc6f2%2FSelection_3196.png?generation=1777529117828537&alt=media)
  
  this is using apr-28 updated kaggle memory profiler. Note that the ouput shape is 0 at the back.
  
  ```
  onnx code:
  
      S, L = dynamic tensor
  
      starts = op.concat([op_constant([0]), op_constant([0]), S_less_one, op_constant([0])], axis=0)
      ends   = op.concat([op_constant([1]), op_constant([1]), S, L], axis=0)
      axes   = op.concat([op_constant([0]), op_constant([1]), dim2, dim3], axis=0)
  
      #---
      last = op_slice(input, starts, ends, axes)
  
  
  ```
