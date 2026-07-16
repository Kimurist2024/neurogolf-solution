# Kaggle Agent full Automated Agent Trace & Progress Update

- Topic ID: 692571
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/692571
- Author: Jiwei Liu (@jiweiliu)
- Posted: 2026-04-17T02:15:56.937676Z
- Votes: 21
- Total messages: 12

## Body

Hello, I'm very excited to share an agent trace. This is a crazy run! **11 sub agents called.** **12 auto compact triggered! 1729 Tools called!!** I think it is pretty obvious which prompts are from me. LOL!
The full session is below. It got LB 794 place all by itself from scratch in 12 hours. 

https://daxiongshu.github.io/kaggle-agent-session-example/

I'll publish the kernel soon. Stay tuned!

Edit: here it is! https://www.kaggle.com/code/jiweiliu/kaggle-agent-lb794-inference

- 4/17 Edit: Agent made great progress in the 2nd/3rd day!
- 4/18 Edit: Nonstop run for 3 days. No Code change so far (I do give it a tip. :P) and 173 auto-compat triggered!

![418](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F100236%2F5a41af8e9b3569e3681ed81104ab4e5e%2F418.jpeg?generation=1776511593656265&alt=media)

## Comments (12)

- **@🤞@** (2026-04-18T11:14:03.547Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Çan i have number of token you use to produise on solution ?

  - **Jiwei Liu** (2026-04-18T11:25:09.390Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    sure, let me figure it out and get back to you

- **Jiwei Liu** (2026-04-19T01:12:08.487Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  After running non-stop for three days, I have to shut it down. It’s been a remarkable run—hundreds of subagents, auto-compact cycles, improving the score from nothing to 4400, even figuring out [a Kaggle scoring bug ](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692621)entirely on its own. But it was also unexpected and unplanned from the start. Eventually, it became unsustainable. The frontend is frozen and crashing, and token efficiency has plummeted. It’s time to fix these bugs and build a more robust, long-running version. :pray:
  ![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F100236%2F000e6b64070393140a59f4d1ec073f74%2Fimage%20(3).png?generation=1776561058624219&alt=media)

  - **Geremie Yeo** (2026-04-19T02:38:17.527Z, votes: {'canUpvote': True}):
    Nice! Just curious, does your agent submit to Kaggle automatically too?

    - **Jiwei Liu** (2026-04-19T03:02:24.803Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      yes it does.

    - **@🤞@** (2026-04-19T06:00:39.937Z, votes: {'canUpvote': True}):
      Great App

  - **Geremie Yeo** (2026-04-21T03:00:51.837Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    How much `/extra-usage` did you burn to achieve those scores?
    
    I have used 70% of my Claude Code Max 20x weekly quota just to reach 3.5k from scratch (and about 40% was from my custom Qgentic agent) so technically the CC usage only contributed to ~2k points. 
    
    It feels I am missing something more efficient else from my estimates 4.5k+ would require a massive amount of `/extra-usage`

    - **Jiwei Liu** (2026-04-23T23:44:47.053Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Hi Geremie, we have a company plan which is not a membership. It doesn't show cost or tokens. but i can tell you this. I burnt $10k for Claude APIs usage for all my projects in the past month. Golf might be a significant part of it but I don't know the exact number.

    - **Geremie Yeo** (2026-04-23T23:49:30.053Z, votes: {'canUpvote': True}):
      Thanks! I have used Codex ($20 plan exhausted) + Claude Code ($200 plan exhausted) + [Qgentic](https://github.com/bogoconic1/Qgentic-AI.git) (Gemini, unlimited, probably around $1k of cloud credits) to achieve my current score. With a vast.ai RTX Pro 6000 Blackwell thats running 24/7.
      
      I have not paid for extra usage yet. The cost of running AI is so expensive these days 😭

    - **Jiwei Liu** (2026-04-24T00:06:34.487Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      extra usage is not worth it. i would rather spend that money on gpt pro for codex

- **yash bhaskar** (2026-04-18T05:01:30.427Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Great work guys! Just realized some of my tasks were wrong 😅 thanks to your ensembling notebook, fixing them now and climbing back up 🔥

- **Tom** (2026-04-17T08:12:49.877Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 2}):
  Goos work @jiweiliu. Let's see how this agent game goes in the final.
