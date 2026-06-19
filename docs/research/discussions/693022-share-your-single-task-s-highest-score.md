# Share your single task's highest score

- Topic ID: 693022
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693022
- Author: Yiheng Wang (@yiheng)
- Posted: 2026-04-19T05:22:21.249457400Z
- Votes: 9
- Total messages: 26

## Body

On my side (04/19): **16.39**

How far we can go at the end? I guess <6400 😄

---0421 update---

I got 21+ score in a task, which should be impossible. Therefore, after some debug and check discussion topics, I'm sure there are some bugs in onnx-tool (which may be found and exploited by LLMs).
Now I'm using onnx-tool 1.0.1 locally and re-prepare my bug related tasks, hope the host can update scoring tools and fix bugs, then everyone can optimize solutions in correct directions.

---0422 update---

I tried to update my local scoring tools, and try to avoid any bugs. After that, my best local score is 15.8

## Comments (26)

- **Russell Kirk** (2026-04-22T06:42:37.013Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  Mine is still 15.9   (15.8XX that rounds up), without using "tricks."  Another post correctly points out a trick that allows higher than 15.9, but I don't see how a score higher than 16 is possible without evading the intent of the scorer.

  - **Yiheng Wang** (2026-04-22T06:51:19.087Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    same on my side, 15.8X.
    I set multiple hard restrictions for my model to avoid touching "tricks", and LLM told me that score 15.9 is the ceiling.

    - **hwe owe** (2026-04-25T05:01:05.643Z, votes: {'canUpvote': True}):
      can we be teamate with you ,@Yiheng Wang,i have score high in my submission and if we make ensemble,we might get champoin

    - **Yiheng Wang** (2026-04-25T05:21:31.933Z, votes: {'canUpvote': True}):
      sorry, I prefer solo for this challenge

    - **hwe owe** (2026-04-25T05:44:58.637Z, votes: {'canUpvote': True}):
      okay.soory

- **yash bhaskar** (2026-04-19T08:38:59.170Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  My highest is 15.36 till now. 
  
  Also those who are using Claude Code or Codex, how many tokens have you guys consumed for far? 😄

  - **(unknown)** (2026-04-19T18:53:47.310Z, votes: {}):
    (deleted)

  - **Russell Kirk** (2026-04-19T20:55:41.260Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    On the 17th, I used 5 million tokens with Opus 4.7.  (part of my subscription-- so not extra charge)
    Even if I tell Claude the correct answer, he doesn't always believe me :(

  - **Geremie Yeo** (2026-04-22T04:38:42.600Z, votes: {'canUpvote': True}):
    Used up my weekly limit for Claude Code with Max 20x lol

- **hwe owe** (2026-04-26T04:58:33.207Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  vtask179.onnx     cost=           0  score=25.00

- **Tony Li** (2026-04-21T14:03:34.847Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  I just got a new single best score : 
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22938014%2Fc05679c45f5268911e45e68229a49ed8%2F1767.jpg?generation=1776780194314043&alt=media)

  - **Yiheng Wang** (2026-04-22T01:29:02.750Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    thanks for sharing Tony. My best one is now 15.8

- **Russell Kirk** (2026-04-19T05:58:58.177Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  14.5 until you made that comment -- I had made some wrong assumptions!

  - **Russell Kirk** (2026-04-19T07:16:58.940Z, votes: {'canUpvote': True}):
    15.9 now.  I don't see how you can get below that.  So I'm still making bad assumptions.

    - **Yiheng Wang** (2026-04-19T07:42:38.083Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      huge improvements within 2 hours!

    - **Russell Kirk** (2026-04-19T07:51:16.907Z, votes: {'canUpvote': True}):
      Yea, I changed from a float to boolean :D

    - **(unknown)** (2026-04-19T10:55:47.877Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
      (deleted)

- **Tony Li** (2026-04-19T06:14:05.417Z, votes: {'canUpvote': True, 'totalUpvotes': 1}):
  Also about Claude Code or Codex? 😄
  
  I firmly believe Claude will trail ChatGPT and Gemini by a wide margin. Let’s see whether that holds true after this competition.
  
  If we judge by pure intelligence, Claude is not even close. It does not belong in the top three on that metric😅.

  - **NNMax** (2026-04-19T06:36:20.913Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    I think claude or codex or any other good models are fine because llms are excellent at pattern recognition. It's only a matter of token efficiency and speed. I'm definitely getting my hands dirty for optimizing the tasks because I'm not trusting LLMs on optimization.

- **Ali** (2026-04-22T06:14:22.320Z, votes: {'canUpvote': True}):
  I have two tasks scoring 20+ and two tasks scoring 18.5+ 
  (All under updated metric)

- **yash bhaskar** (2026-04-22T03:26:01.357Z, votes: {'canUpvote': True}):
  With the updated metric (04/22), got my new single best score:
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F12063947%2F4ccfa932bb73f2a51bb47f185942f697%2FScreenshot%202026-04-22%20at%208.54.36AM.png?generation=1776828328885033&alt=media)

  - **Russell Kirk** (2026-04-22T03:52:35.323Z, votes: {'canUpvote': True}):
    What task#? :D

- **hengck23** (2026-04-21T01:44:52.353Z, votes: {'canUpvote': True}):
  i am to compare fully automatic + human guided semi-automatic optimization.
  so if you interested, please state like (task00x, 15.xx, fully automatic) and i see if i can do better.
  
  i am interested in those tasks that a fully automatic agent cannot solve at all. In that case i want to see if interactive prompting help.

  - **Yiheng Wang** (2026-04-21T05:15:38.867Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Hi @hengck23 , can take a look at this topic: https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827
    The current onnx-tool version used in Kaggle has some bugs, I guess we will have rescore things soon (current best task scores may not correct)

- **(unknown)** (2026-04-22T16:55:33.537Z, votes: {}):
  (deleted)

- **(unknown)** (2026-04-19T05:47:45.720Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
  (deleted)
