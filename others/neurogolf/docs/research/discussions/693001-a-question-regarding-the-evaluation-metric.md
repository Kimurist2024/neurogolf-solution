# A question regarding the evaluation metric

- Topic ID: 693001
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693001
- Author: RimoKag (@rimokaggle)
- Posted: 2026-04-19T02:56:35.000676500Z
- Votes: 1
- Total messages: 1

## Body

From a research curiosity I would like to know the motivation behind defining the evaluation metric max(1, 25-ln(cost)) in such a manner. Can the host or anyone clarify? Moreover is the evaluation metric robust and consistent with the choice of different datasets that will be used to evaluate the functional correctness of the designed network?

## Comments (1)

- **Michael D. Moffitt** (2026-04-19T13:16:30.853Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  Yes, happy to shed some light on this.  While no single metric is perfect, the function `max(1, 25 - ln(cost))` exhibits some nice properties:
  - **A guaranteed minimum**: A base score of 1.0 for any correct network, ensuring that simply solving a task is always rewarded.
  - **A proportional incentive**: Because of the natural log, it rewards scaling down geometrically. Halving the parameter count yields the same point boost whether one is reducing a model from 1,000,000 parameters to 500,000, or from 10 to 5.
  - **A clean maximum**: Given the maximum per-task score of 25, an upper-bound on the (theoretical) total score is a nice round 10,000 points across all four-hundred tasks.
