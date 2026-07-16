# How are you using AI agents for Neurogolf?

- Topic ID: 703854
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/703854
- Author: YouKnoWho (@darksoul026)
- Posted: 2026-06-02T07:49:06.985292700Z
- Votes: 3
- Total messages: 5

## Body

Hi everyone,

I wanted to ask a general workflow question about using AI agents for this competition.

My current score is **6439.82**. I am using both **ChatGPT Pro** and **Claude Code Max**, but my progress feels quite slow.

A typical long session gives me around **+5 score**, and one 5-hour session can consume almost **10% of my weekly quota**. So I am wondering if I am using agents inefficiently.

For people making faster progress, I would really appreciate any general advice on the workflow:

- Do you use agents to solve full tasks end-to-end?
- Or do you mainly use them to build reusable scripts/tools?
- Are you creating custom “skills” for the agents?
- Are you mixing stronger and cheaper models?
- For example, one main high-capability model plus smaller/faster subagents?
- Do you let agents explore freely, or give them very specific hypotheses?
- How do you decide when to stop a bad run?
- Do you work task-by-task, or first build a stronger general framework?

I am not asking for private tricks or task-specific solutions.

I am mainly trying to understand the meta-workflow, because right now I feel like I am spending a lot of quota for very small gains.

Any advice would be appreciated.

Good luck everyone!

## Comments (5)

- **Chan Kha Vu** (2026-06-03T18:12:36.303Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Going into this competition, I never thought I might spend more on LLM tokens than on GPUs for other compute-heavy competitions...
  
  Yeah, I have no idea how to use tokens more efficiently. I guess only a few players at the top has more or less figured this out.

- **Rustam Bazarbayev** (2026-06-02T15:27:05.997Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  I think you have to solve one by one

- **Jan Vorel** (2026-06-02T11:04:24.943Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  You are asking for winning recipe actually. :-)

- **Andres H. Zapke** (2026-06-06T14:54:43.937Z, votes: {'canUpvote': True}):
  I use chatGPT to build scripts and do the rest by hand or following the footsteps of the top notebooks. Never used claude or any of those AI agents. Do you recommend them? Whats the advantage of using them as opposed to my approach?

- **Chet** (2026-06-03T16:50:58.463Z, votes: {'canUpvote': True}):
  Based on my benchmarking, top models are all you need. [GLM vs Opus: ONNX cost-opt](https://www.kaggle.com/code/jsrdcht/glm-vs-opus-onnx-cost-opt-neurogolf-2026)
