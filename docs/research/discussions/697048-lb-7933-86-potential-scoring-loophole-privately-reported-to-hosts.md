# LB 7933.86 - Potential scoring loophole privately reported to hosts

- Topic ID: 697048
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/697048
- Author: Binh (@binhdth)
- Posted: 2026-05-05T04:34:54.111699100Z
- Votes: 8
- Total messages: 4

## Body

Hi @mmoffitt ,

I found a possible scoring loophole in the updated NeuroGolf metric that can inflate LB scores without a real modeling improvement. I emailed the details, a small repro, and a suggested fix to `neurogolf.2026@gmail.com`. I am not posting the implementation details here because it could encourage exploit submissions and make a rescore harder.

## Comments (4)

- **Geremie Yeo** (2026-05-05T05:33:22.793Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5256933%2F87edf4e899da994f985c49a2f73c9b2c%2FScreenshot%202026-05-04%20at%2010.33.12PM.png?generation=1777959201541017&alt=media)

- **hengck23** (2026-05-05T04:49:23.870Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  coming out with a good unhackable metric is more difficult than the optimization itself :) don't over estimate agent ... they can hack anything, even binary and bytecodes?
  
  ## quarter time: red team score 5: blue team 4

  - **Chris Deotte** (2026-05-05T04:53:12.653Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    It's crazy. We should just ask Codex GPT 5.5 to write us an unhackable metric script instead of relying on current GitHub libraries.

- **Zacchaeus** (2026-05-05T05:22:18.030Z, votes: {'canUpvote': True}):
  IMHO Making the scoring function publicly available and allowing 100 submissions per day makes the competition inherently vulnerable to exploitation.
