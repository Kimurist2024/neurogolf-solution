# Task ambiguity: some tasks may not be reliably solvable

- Topic ID: 698249
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/698249
- Author: keymoon (@keymoon)
- Posted: 2026-05-09T01:41:36.332367600Z
- Votes: 8
- Total messages: 5

## Body

I have not investigated all of the details yet, but it appears that some generators can produce the same input with different outputs. [task023](https://arcprize.org/tasks/150deff5) seems to be one example.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1953252%2Fd76104ccd285db4562d30ca8ff964d3e%2F2026-05-09%20103652.png?generation=1778290913080640&alt=media)

In last year’s competition, when generator issues were found, the generators were rewritten and the affected cases were regenerated. For example:

* [`Updates Task #226 to ensure that grids are always 10x10.`](https://github.com/google/ARC-GEN/commit/44250f1df5708f15bd2e04f5e58c6de6080cbb79)
* [`Fixes task #100 so that only area matters.`](https://github.com/google/ARC-GEN/commit/d52b5c60ce001f0869b1655ee6050aad98e72362)
* [`Fixes Task #76 so that yellow pixels must be diagonally connected.`](https://github.com/google/ARC-GEN/commit/8a5aad6d36d133e084b7ccac13598493b3742462)

Are there any plans to apply a similar fix this year?

Removing the hidden cases itself entirely would also be an option, although it would strongly incentivize overfitting to the public cases.

## Comments (5)

- **Michael D. Moffitt** (2026-05-09T17:45:56.470Z, votes: {'totalVotes': 9, 'canUpvote': True, 'totalUpvotes': 9}):
  I should be able to double-check task023’s hidden tests early Monday morning. Also, it’s worth mentioning that for all but two or three tasks, we accepted a hidden test only if its output matched that of each program from the top ten teams in last year’s Code Golf Championship.

  - **Michael D. Moffitt** (2026-05-11T15:39:48.803Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
    OK, finally got a chance to dig deeper into [task023](https://arcprize.org/tasks/150deff5), and I believe its hidden test cases are **not** ambiguous (i.e., there is a unique correct solution for each input).  For that puzzle in particular, note that [ARC-GEN](https://github.com/google/arc-gen) assumes that the number of cyan pixels will always be a multiple of 4, and the number of red pixels will always be a multiple of 3—in other words, a block will never partially overlap with another block, and a line will never partially overlap with another line.
    
    Happy to perform the same check for other tasks if any additional issues arise!

    - **Russell Kirk** (2026-05-12T00:43:37.273Z, votes: {'canUpvote': True}):
      But what if there are multiple "simple semantic" rules that are equivalent of all public input/outputs? but they diverge on different input/output pairs (e.g. the hidden set)?

    - **Michael D. Moffitt** (2026-05-12T01:10:52.277Z, votes: {'canUpvote': True}):
      > But what if there are multiple "simple semantic" rules that are equivalent of all public input/outputs? but they diverge on different input/output pairs (e.g. the hidden set)?
      
      If there's a task where (a) ARC-GEN has been demonstrated to produce inconsistent pairs (like in the post above), and (b) someone is passing all public tests yet inexplicably failing the scorer, then that's something we might be willing to investigate.
      
      Aside from task023, I haven't seen such examples yet (but they did indeed occur last year, so it's definitely possible).

- **Ali** (2026-05-09T18:37:49.257Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  Task 023 scores (not much, but scores 9.63 currently for me) 
  
  I am pretty sure I have 400 tasks scoring now (some with low scores, such as 3.x)
