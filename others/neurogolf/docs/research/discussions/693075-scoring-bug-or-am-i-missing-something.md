# Scoring bug or am i missing something?

- Topic ID: 693075
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693075
- Author: NNMax (@ashok205)
- Posted: 2026-04-19T12:31:15.412468Z
- Votes: 2
- Total messages: 2

## Body

So I have solved all tasks excluding the 6 corrupted ones with a mean score of 10.97 (I bruteforced a lot of tasks hence the low avg) which should give me a total of 4322.18 points right? But the submission only scored 3028.54 which is just the same score as my previous submission. I downloaded the submission and there were 394 tasks as well.

My local test validation gives me 4322.18 points as well.

Is there any constraints that I'm missing?

## Comments (2)

- **Tony Li** (2026-04-19T12:47:13.190Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
  If your local score is higher than the leaderboard score, it likely means that a few tasks are not actually solved correctly.
  
  Instead, the solution produced by your LLM may have effectively memorized the full local test pattern. That can appear to pass locally, but then fail silently and score 0 on the leaderboard, because our local evaluation does not include the full evaluation dataset.
  
  As the host explained, the public test set contains hidden anti-overfitting tasks. These tasks are not exposed to users for training or memorization, but they are still included in the public evaluation.

  - **NNMax** (2026-04-19T13:02:55.047Z, votes: {'canUpvote': True}):
    Thanks, that makes sense.
