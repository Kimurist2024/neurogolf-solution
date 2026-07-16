# Stuck around 5800: Is 6500+ mainly about manual rule writing?

- Topic ID: 700921
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/700921
- Author: Gengsr (@gengsr)
- Posted: 2026-05-18T11:12:18.927119600Z
- Votes: 0
- Total messages: 2

## Body

Hi everyone, I’m currently stuck at around 5800 on NeuroGolf 2026, and I’d like to ask for some advice on the next direction.

My current baseline submission is about 5800.64. To avoid blindly mixing old high-scoring submissions, I built a per-task validation workflow: I compare different candidate ONNX files with the baseline using local probe results, including base_cost, candidate_cost, base_score, candidate_score, and md5. Then I generate single-task override submissions and test only one task replacement at a time on the leaderboard.

Here are the issues I’ve run into:

1. Some candidates from public high-score sources look extremely good locally. For example, some tasks have very low candidate cost and very high local delta. But after submitting, the public score drops instead, almost as if the task failed on hidden cases and lost the original baseline score.

2. I tried hand-writing simple rule-based tasks, such as task006, which is a left/right binary region operation. The local validation passes, but the improvement is less than 1 point, so the return on effort feels quite low.

3. I filtered high-ROI tasks where the base_score is low and base_cost is high. Some of them theoretically have 3–5 points of improvement potential, but many seem to require connected components, object selection, sorting, largest/smallest object logic, or other object reasoning. These are quite hard to implement efficiently in pure ONNX, and I’m also worried about hidden-case robustness.

4. I’m a bit confused about the right path forward. If I want to go from around 5800 to 6500+, is it basically necessary to manually inspect many ARC tasks and hand-write rules? Or is there a more systematic approach, such as template-based rule solvers, automatic rule extraction from generators, or a stable ONNX graph rewriting/compression pipeline?

What I have tried so far:

* Built a candidate matrix for all 400 tasks;
* Filtered out poison tasks that failed online;
* Ran fast triage on high-ROI tasks;
* For tasks that looked writable, tried to write Python oracles first;
* Only planned to convert to ONNX after the Python oracle passed all train/test/arc-gen examples;
* Avoided directly mixing old high-score packages at scale.

The current problem is that many tasks look like fixed templates, color replacement, crop/recolor, or bbox/mask refinement during fast triage, but once I try to write a Python oracle, the rule is often uncertain or does not pass the examples.

So I’d like to ask:

1. Is the main route to 6500+ mostly manual task analysis and rule writing?
2. For low-base-score tasks that require object reasoning, are there any relatively general ONNX patterns people use?
3. Do you have a recommended way to classify tasks quickly, so I can decide which ones are worth hand-writing and which ones should be skipped?
4. For candidates that are locally valid but drop online, how do you usually detect hidden overfitting?
5. Besides hand-written rules, are there any stable, general methods for improving the score?

I’m not asking for direct solutions to specific tasks. I mainly want to confirm the right direction and avoid wasting time on low-ROI tasks. Thanks a lot!

## Comments (2)

- **jacekwl** (2026-05-18T11:46:29.857Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Public notebooks contain a lot of tasks that fail because e.g. they use lookup tables that work only on train+test+arc-gen. One way to spot them is that they have N similar nodes, where N=number of tests.
  Or they does not meet requirements. 
  If they are valid but still fail hidden test cases, usually generating extra tests helps to find edge cases. Or just making sure that your understanding of the task is the same as what generator produces.

  - **Gengsr** (2026-05-18T11:56:11.567Z, votes: {'canUpvote': True}):
    Thanks, that makes a lot of sense.
    
    I think I ran into exactly this with some public candidates. They looked great locally, but failed badly online, so they were probably lookup-table-like solutions. I’ll use the repeated-node check mostly to filter suspicious public candidates.
    
    For my own rule-based attempts, I guess the better path is to validate against extra generated cases before converting to ONNX.
    
    Do you usually use the official generator directly for those extra tests, or do you also create manual edge cases? Also, at higher scores, is the main bottleneck usually finding the correct rule, or making the ONNX implementation small enough?
