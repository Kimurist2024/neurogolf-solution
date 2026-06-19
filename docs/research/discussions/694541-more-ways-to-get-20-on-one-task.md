# More ways to get 20+ on one task

- Topic ID: 694541
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694541
- Author: Pavel (@pavelsavchenkov)
- Posted: 2026-04-25T10:57:56.113394600Z
- Votes: 6
- Total messages: 4

## Body

# Detailed report (written by LLM, reviewed by me)

[https://www.kaggle.com/code/pavelsavchenkov/onnx-tool-exploit-report-apr-25-v2](https://www.kaggle.com/code/pavelsavchenkov/onnx-tool-exploit-report-apr-25-v2)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F172360%2Fe3de0135f0c07fbc397b1f2a747d936c%2FScreenshot%20from%202026-04-25%2010-55-04.png?generation=1777110932823075&alt=media" width="500">

I believe that cost lower bound for one task should be `9000` (`15.895020143681643` score) (e.g. one cast from float32 to bool).

Why: we must materialise the output tensor. Its shape must be `1x10x30x30`. So `(1 byte) * 1 * 10 * 30 * 30 = 9000`. Anything smaller than that is miscalculated profiling.

# Summary (written by me, reviewed by LLM):

I believe there is an issue with how onnx-tool is implemented.

Let's start with fundamentals. From onnx-tool perspective, model consists of 
* set of tensors. Each tensor has value, shape and dtype (might be unknown or partially unknown)
* set of nodes. Each node has an underlying op, attributes, list of input tensors, list of output tensors. All are known.

In order to profile the model, onnx-tool wants to compute shape and dtype of each tensor. We process nodes in topological order and **try** to set shape and dtype of output tensors based on input tensors and node attributes.

Some operations (nodes) behave such that their output tensors shape depends on some of input tensors **values** (not input tensors shape).

As one example, let's consider [Compress](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress). It takes "input" and "condition" as inputs, then filters "input" according to condition. E.g. "input"=`[[1, 2], [3, 4], [5,6]]`, "condition"=`[False, True, False]` becomes `[[3, 4]]`.

Now, what happens when we want to calculate shape of `Compress` node tensor output:
* code calls `shape_infer()` method of `CompressNode`
* `shape_infer()` is not implemented for `CompressNode`, so it calls `shape_infer()` method of parent class, which calls `value_infer()`
* `value_infer()` calls `numpy.compress` on `get_numpy()` of input tensors and put the result to the output tensor
* **[KEY PART] `get_numpy()` of a tensor with unknown value returns `numpy.zeros` !!**

**Then `Compress` tensor output resolves to `numpy.compress([False, False, ..., False], input)` which is tensor of shape `[0, ...]`. It then be counted as having 0 memory footprint, while in fact it can be big.**

There is another (much more local) issue. `WhereNode` `shape_infer` does not broadcast shapes correctly ([here](https://github.com/ThanatosShinji/onnx-tool/blob/main/onnx_tool/node.py#L779-L785), see more details in the notebook above.)

# Next steps

About `get_numpy()` emitting zeros on unknown values:

My understanding is that this design choice made `onnx-tool` implementation easier, at a cost of making it report incorrect values in many cases.

I believe the ideal proper fix would be to re-implement onnx-tool such that neither `value_infer()` nor `shape_infer()` rely on `get_numpy()` returning dummy zeroes. That is a big refactor though.

Another fix would be to make a set of targeted changes to problematic ops.

@mmoffitt wondering what you think
* do you agree with reasoning that `9000` should be the hard cost lower bound for one task, given correct profiler?
* do you think the bugs above are really bugs and should be fixed? Maybe I am missing something.

I am personally okay with having additional objective to optimize solution against concrete profiling implementation (would prefer not to, though). It would be nice to have clarity about that, to not waste any effort.

Thank you!

## Comments (4)

- **Michael D. Moffitt** (2026-04-28T22:21:36.920Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thank you for the excellent post!
  
  > do you agree with reasoning that 9000 should be the hard cost lower bound for one task
  
  Today's [metric update for April 28th](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230) simplifies the memory footprint calculation dramatically, and excludes the `input` and `output` layers.
  
  *(Note that it may have been possible to achieve this before using the previous metric, e.g., by cleverly employing the `Pad` operation which allowed the expansion of layers at no extra memory cost).*

  - **Pavel** (2026-04-28T23:52:36.507Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    Great job on the update, thanks a lot!
    
    I believe that all the buggy networks I had saved now either failing or scoring some sane values.
    
    However, after a quick pass on the new code and I think I found another exploit: one can add an initializer named `input` which makes `onnx-tool` drop `macs` and `params` to zero on the example task 1 solution below.
    
    Attaching an (an unpolished) archive with:
    * task 1 submission without initializer exploit
    * task 1 submission with initializer exploit
    * .md report
    * script which confirms difference in profiling and that the only difference between networks is one initializer (need to adjust `neurogolf_utils` import path)

    - **Michael D. Moffitt** (2026-04-29T17:55:16.503Z, votes: {'canUpvote': True}):
      Yep, this looks like a legitimate issue.  I've added an addendum to [yesterday's update](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230), calling out this bug and the other one you reported.  Both of them are great catches -- thank you!!

  - **(unknown)** (2026-04-29T04:27:25.377Z, votes: {}):
    (deleted)
