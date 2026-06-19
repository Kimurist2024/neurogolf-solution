# [Proposal] Introduce a notebook sharing cooldown before the deadline

- Topic ID: 691904
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/691904
- Author: parthenos (@nihilisticneuralnet)
- Posted: 2026-04-15T23:06:17.978495300Z
- Votes: 14
- Total messages: 9

## Body

One recurring pattern in competitions like this one, is the surge of public notebooks in the final hours (often ensemble pipelines). 

Last moment public releases disproportionately benefit participants who are online at that exact moment, able to quickly integrate and ensemble these ideas. while others, who may have independently developed similar methods or who simply aren't active in the final hours, don't get a fair chance to respond. The result (especially in the bronze medal zone) becomes more about the reaction speed to last-minute disclosures 

Therefore, introduce a mandatory cooldown period for public notebook sharing (atleast 48-72 hours before the deadline)

## Comments (9)

- **Addison Howard** (2026-04-21T14:01:43.043Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Sharing a new notebook is disabled during the last week for this competition. Only in rare instances do we _not_ enable that.

  - **parthenos** (2026-04-23T04:32:16.913Z, votes: {'canUpvote': True}):
    last santa competition, participants were begging them to stop who were posting high scoring kernels at the very last hour of the deadline (they were updating past notebooks to share csv files)
    
    - https://www.kaggle.com/code/saspav/santa-submission/comments
    - https://www.kaggle.com/code/jazivxt/why-not/comments

  - **Paritosh Kumar Tripathi** (2026-04-26T09:32:11.443Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
    But one can simply publish a newer version to their already shared notebook. Can that be disabled as well in the last week for the competition?

- **CPMP** (2026-04-16T03:39:36.777Z, votes: {'canUpvote': True}):
  Normally sharing a new notebook is disabled during last week.

  - **parthenos** (2026-04-16T06:50:34.800Z, votes: {'canUpvote': True}):
    not all (afaik)
    
    last santa competition, participants were begging them to stop who were posting high scoring kernels at the very last hour of the deadline
    - https://www.kaggle.com/code/saspav/santa-submission/comments
    - https://www.kaggle.com/code/jazivxt/why-not/comments

    - **Tom** (2026-04-16T09:03:06.570Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      How did they not disable the sharing button in the final week? Isn’t this a standard rule in most competitions? These guys seem intending to farm the coding GM by doing this stuff.

    - **c-number** (2026-04-16T09:54:09.953Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
      If I remember correctly they were updating past notebooks to share csv solutions.

    - **CPMP** (2026-04-16T09:59:57.503Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      Yes, this is a way to bypass the limit. The fix is to not allow any public notebook version during last week.

    - **Manish Swami** (2026-04-17T06:08:55.203Z, votes: {'totalVotes': -2, 'canUpvote': True}):
      Another potential solution is to establish a score-based restriction on public sharing. Specifically, the platform could disable the ability to publish or update notebooks once they surpass a certain LB threshold. This ensures that while helpful baseline techniques can still be shared, the 'medal-winning' logic remains private to those who developed it, preventing late-game leaderboard shakeups caused by public ensembles
