# Question on scoring: per-task score appears to differ between single-file and full-bundle submissions

- Topic ID: 702256
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/702256
- Author: Andrey Yunoshev (@andreyyunoshev)
- Posted: 2026-05-21T23:36:46.213681700Z
- Votes: 0
- Total messages: 8

## Body

I'm trying to understand the exact scoring mechanism for submissions and noticed what looks like a discrepancy. Could anyone (especially the organizers) clarify?

**Setup — controlled experiment with 3 submissions of the same `task262` ONNX file:**

| Submission | Files in zip | Public LB returned |
|---|---|---|
| (1) Only `task262.onnx` (my candidate, ~600 bytes cost) | 1 | **18.60** |
| (2) 400-task bundle that includes the same `task262.onnx` | 400 | **6404.04** |
| (3) Same 400 bundle but `task262.onnx` removed | 399 | **6403.62** |

**Implied contribution of `task262` in bundle context** = (2) − (3) = **0.42 LB**

But the same file scored **18.60 LB** when submitted alone. So the per-task score appears to depend on whether other files are present.

The expected per-task points formula is `max(1.0, 25.0 - log(memory + params))` from `neurogolf_utils.verify_network`, which gives ~18.6 for my file's cost ≈ 596. This matches the single-file score exactly, but **not** the bundle contribution.

The competition Overview says the official scorer "will also employ a private dataset (containing a smaller number of examples per task)". My local validation passes 100/100 on a holdout of ARC-GEN-style samples I generate myself, but the bundle score behaves as if private examples failed.

**Questions:**

1. Is the private dataset evaluated for **every** submission (single-file or bundled), or only when the bundle contains a network for the corresponding task?
2. When a network passes all `train` / `test` / `arc-gen` examples but fails on the private dataset, what is the per-task contribution? Is it `0`, the `cost_pts`, or something else (e.g., a fixed penalty)?
3. Is there a documented spec for what the single-file Public LB number actually represents, given that 399 of the 400 task slots are missing from the zip?

A pointer to documentation would be perfect. Thanks!

## Comments (8)

- **Chris Deotte** (2026-05-22T01:20:32.403Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Maybe this is related to this discussion [here][1]
  
  [1]: https://www.kaggle.com/competitions/neurogolf-2026/discussion/699840

  - **Michael D. Moffitt** (2026-05-22T21:02:38.333Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    +1, I believe that is the root cause (specifically, there is a bug in ONNX Runtime that retains state between consecutive task executions when networks are malformed).
    
    @andreyyunoshev Take a look at that discussion, and see if it might apply to these tasks in your bundle: #272, #294, #317

    - **Andrey Yunoshev** (2026-05-22T22:30:29.727Z, votes: {'canUpvote': True}):
      I checked all three for the bias_len < out_channels pattern — task272/294 already had bias=10/out_ch=10, task317 had no bias (I added zero bias). Upgraded ir/opset on ir=6 tasks. Check also all other tasks. So bias-OOB hypothesis doesn't appear to be the live bug for my bundle — possibly something else (state retention?). Anyone have additional reproducer patterns?

    - **Chan Kha Vu** (2026-05-22T22:33:29.083Z, votes: {'canUpvote': True}):
      I'm also having instability issues, and it happened in the range of problems where I didn't have bias_len < out_channels pattern. The funniest part is the score for tasks where I had those bugs are stable.

    - **Geremie Yeo** (2026-05-23T17:16:16.480Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      @mmoffitt I wonder if there is any chance this could be fixed (using something like restarting the environment for each task)? It may make judging much slower though
      
      The funny thing is, we submitted the exact same zip file twice within a 1 min period. The scores were different 😭

    - **Chris Deotte** (2026-05-23T17:58:39.777Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      @andreyyunoshev @chankhavu @yeoyunsianggeremie Note there are multiple types of Conv layers: i.e. Conv, ConvInteger, QLinearConv, ConvTranspose, DeformConv. Make sure you check every one of these layers for `bias_len < out_channels`.
      
      If that doesn't solve the problem, then explain to Codex or Claude the problem w/ `bias_len < out_channels` and have Codex or Claude scan all your onnx for any other graph layer parameters that may cause a similar issue (i.e. other malformed networks).

    - **Michael D. Moffitt** (2026-05-23T18:59:29.433Z, votes: {'canUpvote': True}):
      I am now investigating a potential issue in ONNX Runtime that appears to (incorrectly) reuse scratchpad memory between sessions without initialization, triggering the contamination of state even if the models themselves are perfectly valid.

    - **Michael D. Moffitt** (2026-05-24T01:48:24.517Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      I've reported this bug to the ONNX Runtime developers here: https://github.com/microsoft/onnxruntime/issues/28654
      
      If others have additional info that might be relevant, please let me know -- many thanks!
