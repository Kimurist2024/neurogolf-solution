# Token, Overfit, and Timeout

- Topic ID: 704762
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/704762
- Author: Tony Li (@tonylica)
- Posted: 2026-06-06T00:52:54.686714600Z
- Votes: 5
- Total messages: 19

## Body

I think there are three major challenges in this competition: token limits, overfitting, and score timeout.

I may write a separate post about token issues. This one is mainly about overfit and timeout.

My current concern is around these two task groups:

**Overfit-risk top 10 task IDs:**
task192, task319, task118, task359, task018, task285, task096, task048, task355, task219

**Slowest top 10 task IDs:**
task358, task350, task212, task335, task246, task022, task375, task009, task074, task070

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F22938014%2F086a1058428e4c8b4af8af38ffe628c2%2Fscore10.jpg?generation=1780785431126881&alt=media)
How do you usually manage these overfitting, and timeout issues?

For tasks like these, how do you detect possible overfit (score zero in kaggle )early, or avoid having the LLM repeatedly make the same overfit optimization? And how do you reduce timeout risk without weakening the score?

About overfit , I have tried generating a hidden private dataset. It may reduce the overfitting risk by around 80%–90%, but it needs to be balanced with the time the LLM spends optimizing and scoring. Also, sometimes a solution fails my synthetic private dataset but still passes Kaggle, so it is a double-edged sword.

## Comments (19)

- **Yiheng Wang** (2026-06-07T10:29:16.940Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  Hi Tony, thanks for sharing this topic. Let me share some of my experiences:
  
  Overfitting: On my side, I record each "wrong" optimization for each task and tell my model to avoid that kind of change in the future.
  
  Timeout: I did some profiling-based optimization, which helped me reduce my >30 min inference time to ~12 min when I was in the 74XX range. It's not hard — just tell the LLM your expectation: maintain the current cost and reduce profiling time.
  
  Token: I subscribed to ChatGPT Pro ($200/month). I can run my workflow with codex automatically for 2–4 days per week to use up the weekly limit (recently, OpenAI frequently reset quota, thus sometimes can use 4-6 days per week), and after that I can keep using ChatGPT's web page and do copy-and-paste optimization instead. It's unlimited, and you can open multiple tabs in parallel. Therefore, we don't need to spend too much money and can keep optimizing our models every day.
  
  ![chatgpt example](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F682501%2Fa112481018a611eee9d68a2c9a829151%2FScreenshot%202026-06-07%20at%2017.15.56.png?generation=1780827957019810&alt=media)

  - **Jan Vorel** (2026-06-07T11:57:11.673Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    After this post are you confident to keep second place? ;) Thanks for sharing

    - **Tony Li** (2026-06-07T12:31:42.507Z, votes: {'canUpvote': True}):
      This competition requires real time, energy, and effort. So there’s no need to worry about us. Time cannot be manipulated by simply sharing ideas or a few words. For example, once someone reaches the 7,500-point level, it may take 10+ days just to gain another 50–100 points. That part can not be reduced .

    - **Jan Vorel** (2026-06-07T12:45:53.440Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Challenge Accepted :)

    - **Yiheng Wang** (2026-06-07T12:58:20.473Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      These are not secrets, I think many participants have already used similar ways😄

  - **Paritosh Kumar Tripathi** (2026-06-07T12:06:00.230Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Somehow everyone is using the same strategy lol :) but I do not understand why a lot of people have such high inference times, for us inference never took more than 2 minutes for all 400 tasks (assuming inference here means the kaggle submission runtime)

  - **yash bhaskar** (2026-06-07T12:26:06.807Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    Are submissions taking more than 30 min?? For us its ~2 min for 400 tasks.

    - **Tony Li** (2026-06-07T12:32:24.310Z, votes: {'canUpvote': True}):
      You will see that, when you going forward

    - **Chan Kha Vu** (2026-06-07T17:44:47.720Z, votes: {'canUpvote': True}):
      In fact, I'm surprised your team hasn't seen it yet @yash9439, we hit the 30min wall much earlier than you it seems. Which means we're missing a lot on some things...

    - **Paritosh Kumar Tripathi** (2026-06-07T17:53:30.657Z, votes: {'canUpvote': True}):
      @chankhavu On the contrary we feel that we are also missing on a lot of things by not being able to utilise the 30 min time limit fully.

  - **Tony Li** (2026-06-07T12:34:21.550Z, votes: {'canUpvote': True}):
    @Yiheng , thanks,  a very good tip for us,  best wish to you

- **Chet** (2026-06-07T18:21:15.090Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  The LB is now consistent with the final scoring data—is my understanding correct?

- **Geremie Yeo** (2026-06-06T19:19:55.940Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Interestingly, for our last 10 hidden zero tasks, none of them intersected with your list

  - **Tony Li** (2026-06-06T22:40:41.080Z, votes: {'canUpvote': True}):
    This may change over time, but my ChatGPT obsession keeps overfitting these tasks again and again — especially task 359.

- **robga** (2026-06-06T18:38:27.823Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Timeouts: my aggregate runtime for all those 10 tasks is 2.8 seconds score 177. It's cool to imagine people are taking different approaches and wonder what the ensemble effect may be when groups form later.

  - **Tony Li** (2026-06-06T22:39:25.120Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    I did a probe test run for this slowest runtime. It took 9 minutes, which is half of my current total runtime of 18 minutes. The total score is 195. I also updated the picture in my original post.

- **Cona** (2026-06-06T08:39:44.820Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Arc-Gen natively generates some insoluble samples for a few complex tasks, and picking them out may help.
  
  I agree with @xsmaxpc that overfitting might be valid, so could we investigate those samples paired with onnxs that fail on our own private set but pass the official private set? I haven't implemented this pipeline yet, so I don't know if it's an efficient choice.

- **Paritosh Kumar Tripathi** (2026-06-06T03:27:12.973Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  > Also, sometimes a solution fails my synthetic private dataset but still passes Kaggle, so it is a double-edged sword.
  
  I noticed it too, I believe increasing the number of samples in the hidden test set should help.

- **Aibe** (2026-06-06T05:08:00.917Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  It seems that the competition doesn't have a private leaderboard. If the organizers aren't going to swap in a different dataset to rescore submissions after the competition ends, then overfitting to the leaderboard is a valid choice.😁
