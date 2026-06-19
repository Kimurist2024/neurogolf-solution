# Are we reaching 10k within few days?

- Topic ID: 694706
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694706
- Author: Kawchar Husain (@kawchar85)
- Posted: 2026-04-26T13:37:06.591443500Z
- Votes: 1
- Total messages: 11

## Body

The top team is already at 9090.93, which means an average of about 22.73 per task across 400 tasks. That feels extremely high, since the score depends on 25 - ln(cost), and cost includes MACs, memory, and parameters.

So I’m wondering: is this achievable with normal ONNX optimization, or did top teams find some scorer/profiler hack?

Curious what others think: should we focus on solving more tasks, or mainly on finding cost/scoring optimizations?

## Comments (11)

- **NNMax** (2026-04-26T14:15:34.807Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  It seems host is still rolling out the bug/exploit fixes to the onnx-tool. Until then we can't say for sure.

- **shanzhong8** (2026-04-27T00:55:40.593Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  every submission greater than 5800 contains hacked tasks

  - **Geremie Yeo** (2026-04-27T01:29:20.493Z, votes: {'canUpvote': True}):
    The public notebook scoring ~5500 also contains hacked tasks lol

    - **hengck23** (2026-04-27T01:48:31.357Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      I hope to see some good quality onnx graph greater than 15.3 without hack so that all kagglers can learn something.

- **vishnuvardhan33** (2026-04-26T15:52:41.510Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  They're definitely using the exploit

- **theredbluepill** (2026-04-26T14:10:23.293Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  It's token rival my friend.

- **Geremie Yeo** (2026-04-26T17:47:17.273Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  I managed to get a score of 25.00 on task001 🥴
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5256933%2F152bae3fe207be49dcf428846e099d78%2FScreenshot%202026-04-26%20at%2010.46.33AM.png?generation=1777225615816086&alt=media)

  - **Kawchar Husain** (2026-04-26T17:52:31.883Z, votes: {'canUpvote': True}):
    wow... wow... 🥺

  - **hengck23** (2026-04-26T18:37:39.300Z, votes: {'canUpvote': True}):
    "Outplay the profiler, be outplayed by the agent."

- **hengck23** (2026-04-26T15:15:32.977Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  i remember the first day i self-studied on agent to solve automatic progamining task. the tutorial  was "ask the agent to write a python guessing 5 number ... there is a C compile binary to print the correctness of the guess". ... after a few iteration, i see the following on codex CLI ...   
  
  " we could reverse engineer  the binary without breaking the rule ... it is faster to get the correct guess ..."

- **FOYSAL** (2026-04-26T18:59:20.943Z, votes: {'canUpvote': True}):
  I'm surprised to see the result. How is it possible?
