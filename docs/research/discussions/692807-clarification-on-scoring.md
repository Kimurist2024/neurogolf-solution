# Clarification on scoring

- Topic ID: 692807
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/692807
- Author: Damien Mourot (@nahman)
- Posted: 2026-04-18T00:26:14.371708400Z
- Votes: 1
- Total messages: 1

## Body

Hey, 
just to validate, when you say:

>Functional correctness will be determined by validating the network against the original ARC-AGI benchmarks, the ARC-GEN-100K dataset, and a small private benchmark suite (so as to prevent teams from overfitting their solutions). To be eligible for points, your network must produce correct results across all of these tests.

The private benchmark suite is also run at scoring time ? If I got points in the LB, the task is validated ?

Thanks

## Comments (1)

- **Michael D. Moffitt** (2026-04-18T12:51:50.840Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Great question: response below:
  
  > If I got points in the LB, the task is validated ?
  
  That's correct.  Since the LB includes all tasks—including private—the scores you see there are all 100% verified (and there won't be a separate leaderboard after the competition deadline has passed).
