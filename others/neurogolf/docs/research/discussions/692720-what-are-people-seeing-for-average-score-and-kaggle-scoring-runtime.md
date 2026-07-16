# What are people seeing for average score and Kaggle scoring runtime?

- Topic ID: 692720
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/692720
- Author: Tony Li (@tonylica)
- Posted: 2026-04-17T18:22:07.535897400Z
- Votes: 8
- Total messages: 10

## Body

I’m curious what the current average score and Kaggle evaluation runtime look like for others.

Right now I’m at **165 tasks solved** with a **score of 2070.74**, so average score  is ~12.55, and my Kaggle scoring run time is about **90 seconds**.

 if  #1 is out of reach, chasing very low-scoring solutions feels like a waste of time and resources. Because of that, I set  >12  as my acceptance threshold and only ask the LLM to produce solutions better than that.

My original idea was to **solve all 400 tasks first**, then optimize from there. That is basically how I approach Code Golf too, but it is exhausting in practice. There is a lot of repetition and wasted effort.

I did find one method that looked promising: it seemed able to solve all tasks in a single run and score around 10. It passed locally, but for some reason it failed on Kaggle. I ran out of time and submission quota before I could investigate it properly, so I’ve put that approach on hold for now and gone back to solving the tasks one by one.

Would be interested to hear where others currently are in terms of **average score, number of solved tasks, and runtime**.

## Comments (10)

- **Kameron Kilchrist** (2026-04-19T00:16:35.797Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  I have ~383 / 394 possible solved with 4572.92 points (per leaderboard)
  
  394 possible because tasks 021, 055, 080, 184, 202, 366 are unscorable due to oversize inputs 
  
  Min of 6.58, max of 15.50 (bit skeptical of that tbh--I'll investigate further later) 
  
  Mean of 11.98, median of 12.03; Q1 10.82, Q3 13.38
  
  Math is a bit off because there's ~10 point discrepancy between my own tracking and Kaggle's reporting. I believe Kaggle will accept incorrect answers and award 0 or 1 points for those.

  - **(unknown)** (2026-04-19T00:41:09.117Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
    (deleted)

- **NNMax** (2026-04-18T14:59:29.713Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Currently i'm at 184 tasks solved but at locally I did around 197 and later got error in submission and it turned out to be that those failed tasks turned out to be those that didn't pass arc_gen validation and another one had a silly logic error.
  
  And for runtime, how do you see that because I'm just uploading the submission.zip directly. 
  
  Bruteforce solving all tasks is pretty easy here it seems, the real challenge is the cost optimization.

- **Tom** (2026-04-18T04:12:47.923Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Hey Tony, there's a pending time within 90 seconds, the actual time seems less than that.

  - **Tony Li** (2026-04-18T14:00:40.480Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Thanks, Tom. Really glad to see you around again. My runtime is a bit longer now, around 120 seconds.

- **Jiwei Liu** (2026-04-18T01:58:03.103Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Hi Tony, we solved 277 tasks in the latest run and the kaggle scoring time is also 90 seconds.

  - **Tony Li** (2026-04-18T13:56:24.810Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Thanks, Jiwei. Solving 277 tasks with a 90-second Kaggle scoring time is seriously impressive. It is always great to see you around. You are a mentor to me, and I always hope we will have the chance to work together again someday. I was hoping to win a solo gold in this competition, but it may turn out to be as brutal as Code Golf😅.

    - **Jiwei Liu** (2026-04-19T00:43:20.003Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Hi Tony, it's always an honor to be your teammate. And good luck on getting a solo gold and a grandmaster! I know it's just a matter of time. 
      
      Just want to add our latest submission of solving 376 tasks takes 3 min 10 sec to score.

- **yash bhaskar** (2026-04-17T18:31:35.697Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  I didn’t really thought much about cost optimization until now, was mostly focused on solving tasks. Right now I am at 167 solved with a score of 1951. Kaggle Scoring runtime is similar around 80 sec.
  
  Looks like I will need to start working on optimization next.
  
  Update (18/04) : 178 Task -> 2328.6 LB
  
  Update (19/04) : 244 Task -> 3039 LB
  
  Update (20/04) : 363 Task -> 4500 LB
  
  Update (21/04) : 377 Task -> 4900 LB

  - **Tony Li** (2026-04-18T13:58:45.277Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Hi bro, you are always sharing great progress and helping us a lot. It makes sense that optimization becomes the next frontier once the solving side is stable.
