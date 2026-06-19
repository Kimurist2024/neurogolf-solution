# Sanity-checking the leaderboard: does top-team performance ~7,400 LB really reduce to ~600-byte solutions per task without scorer exploits?

- Topic ID: 701942
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/701942
- Author: Andrey Yunoshev (@andreyyunoshev)
- Posted: 2026-05-20T10:32:26.508397200Z
- Votes: 9
- Total messages: 6

## Body

Hi everyone, hi @mmoffitt — I'd like a sanity check before I keep pushing on this competition. I've spent significant time exploring techniques in shared notebooks and discussions, and I want to confirm I'm understanding the playing field correctly.

### The math

Per-task score is `pts = max(1.0, 25.0 - ln(cost))`. So total LB = sum over 400 tasks of `max(1, 25 - ln(cost_i))`.

If a team is around **LB ≈ 7,400** (close to top-10), that implies an average per-task score of `7400 / 400 = 18.5 pts`, which inverts to an average per-task cost of roughly:

```
cost ≈ exp(25 − 18.5) = exp(6.5) ≈ 665 bytes per task
```

Top-1 around **LB ≈ 7,457** implies even lower — `exp(25 − 18.64) ≈ 578` bytes per task on average.

This is what I keep seeing referenced as "the ~600-byte regime."ф

### What 600B per task means in practice

If the *average* is around 600 bytes, then roughly half of the 400 tasks need to come in below 600B — i.e. at a graph size where you essentially have:

- 0–20 parameters in initializers
- 1–4 intermediate tensors of small shape (or none counted at all)
- Output going more or less straight from input through a handful of ops

That is a *minimal-graph* regime — almost no learned weights, almost no intermediate broadcast tensors, op-fusion to the point that you can describe the whole task in a few lines.

### My current bundle

I am at LB ≈ 6,358.65, which works out to an average per-task cost of:

```
6358.65 / 400 = 15.90 pts/task   →   cost ≈ exp(25 − 15.90) = exp(9.10) ≈ 8,950 bytes/task
```

So my bundle is roughly **15× more expensive on average** than the top-10 level, and about **15.5× more expensive** than top-1. I clearly have a lot of structural compression headroom — I'm not arguing the gap is illegitimate.

### My question

I just want to confirm I'm reading the target correctly:

> **Is an average per-task solution size below ~600 bytes a realistically achievable target through honest structural compression alone?**

I'm asking because before I commit weeks of work to chasing that number, I'd like to know whether ~600B/task is a regime I can actually reach by writing tight, minimal-graph ARC encodings — or whether it would only be reachable through scorer quirks I'm not aware of.

Yes / no would already help me a lot. Thanks!

## Comments (6)

- **jacekwl** (2026-05-20T13:01:03.510Z, votes: {'totalVotes': 9, 'canUpvote': True, 'totalUpvotes': 9}):
  I think using average is a bit misleading here because of the `ln` in the formula.
  My average cost is ~7200, while median is ~860. Average is basically determined by the worst tasks.

- **robga** (2026-05-20T12:22:10.770Z, votes: {'totalVotes': 10, 'canUpvote': True, 'totalUpvotes': 10}):
  The number 1 tip for any kaggle competition is to understand the evaluation metric and go from there. So the fact you're doing that means you're well positioned. Keep going! I'm not aware of any bug.

  - **Chan Kha Vu** (2026-05-23T00:24:39.753Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    > So the fact you're doing that means you're well positioned. Keep going! I'm not aware of any bug.
    
    To add to this motivational speech. @andreyyunoshev, people will probably form super-teams at the end of this competition to boost / combine scores. So with densest packing, 60 people can get a gold medal... and you're probably in the gold zone for now 😁

- **Durga Kumari** (2026-05-25T06:17:31.843Z, votes: {'totalVotes': -1, 'canUpvote': True}):
  Your math and interpretation look reasonable to me. A ~600B/task average does seem to imply extremely minimal graphs with heavy structural reuse/compression.

- **Navneet** (2026-05-23T04:18:29.167Z, votes: {'canUpvote': True}):
  Cool Sanity check leaderboard @andreyyunoshev

- **Rustam Bazarbayev** (2026-05-21T03:58:55.123Z, votes: {'canUpvote': True}):
  I think they got a higher score because they had some easier tasks, which made it easier to score above 20.
