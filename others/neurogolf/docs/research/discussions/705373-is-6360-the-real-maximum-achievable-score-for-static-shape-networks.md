# Is ~6360 the real maximum achievable score for static-shape networks?

- Topic ID: 705373
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/705373
- Author: SebastianGil00 (@sebastiangil00)
- Posted: 2026-06-09T19:46:39.931783Z
- Votes: 0
- Total messages: 4

## Body

I've been studying the scorer and I keep hitting what looks like a hard ceiling, so I wanted to share the reasoning and ask whether I'm missing something.

The score is points = max(1, 25 − ln(macs + memory + params)) per task. Using the official score_network, I measured the cost of a trivial single-node network whose only job is to emit the required [1, 10, 30, 30] output:

bool output → memory = 9000 → 25 − ln(9000) ≈ 15.89 points
fp16 output → 18000 → ≈ 15.20 points
fp32 output → 36000 → ≈ 14.51 points
Since score_network counts every node's output tensor (a 2-node chain measures 72000), the output tensor alone sets a floor: any network that genuinely produces a static [1,10,30,30] result seems bounded near 15.9 points/task, i.e. ~6360 total if every task used a bool output — and lower in practice since many tasks need fp16/fp32 outputs.

Yet the top of the leaderboard sits well above 7000. The only way I can see to beat the output-memory floor is to make the counted output volume smaller than 9000 (e.g. a shape-inference dimension collapsing toward 0).

So my genuine question: is ~6360 the real ceiling for "honest" static-output networks, and is everything above it relying on the output-memory trick — or is there a legitimate technique to get a full [1,10,30,30] result scored below the 9000-byte floor that I've simply missed? Would love to learn if I'm wrong.

## Comments (4)

- **jacekwl** (2026-06-09T20:00:57.730Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Macs is excluded from current metric after update, it's just `max(1, 25 − ln(memory + params))`
  
  Input and output are excluded from calculations.
  
  There are a couple of official update posts in Discussion tab.

  - **SebastianGil00** (2026-06-09T21:07:51.847Z, votes: {'canUpvote': True}):
    Thanks a lot, jacekw. That completely resolves it, and it's a big correction on my end. I was scoring locally with the bundled score_network, which still sums macs + memory + params and also counts the output tensor, so it was massively overestimating cost and inventing a ceiling that doesn't exist. With max(1, 25 - ln(memory + params)) and input/output excluded, the picture is totally different. I'll go read the official update posts.
    
    One thing I'm still trying to understand on the validation side: when a low-cost solver passes every public ARC-GEN sample but still scores 0, what is the hidden set actually probing? I've hit cases where a model that's correct on all public examples fails the hidden check, and I genuinely can't tell whether it's testing out-of-distribution variants of the task (shifted, resized, or recolored grids) or just more samples from the same generator distribution.
    
    I ask because it changes the whole optimization philosophy: if the final private set includes OOD variants, a model has to stay genuinely invariant (a real solver), whereas if it's the same distribution, a tighter fit is fine. Any insight there would save a lot of blind submissions on my end. 🙏

    - **robga** (2026-06-10T08:27:04.473Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      It may be worth nothing that the organiser has said, "for all but two or three tasks, we accepted a hidden test only if its output matched that of each program from the top ten teams in last year’s Code Golf Championship", so almost without exception you can go back to those solutions to find a satisfactory variant or discover why yours may not be.

    - **SebastianGil00** (2026-06-10T17:25:13.557Z, votes: {'canUpvote': True}):
      Thanks @robga , very helpful. One question if you are willing: for the floor-bound tasks, is the general approach to crop the input to a bounded top-left region and compute there? And if so, how do you size that crop so it safely covers the hidden size range per task?
