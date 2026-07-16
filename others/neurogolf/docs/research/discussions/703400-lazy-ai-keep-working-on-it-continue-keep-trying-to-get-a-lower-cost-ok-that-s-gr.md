# Lazy AI - "Keep working on it" "Continue" "Keep trying to get a lower cost" "OK, that's great, now do it"

- Topic ID: 703400
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/703400
- Author: Boladi (@boladi)
- Posted: 2026-05-30T18:38:45.628816100Z
- Votes: 2
- Total messages: 8

## Body

Just venting here, but I'm at a point where I can only ask an AI to work on single Tasks at a time.  Even asking to do similar things to several Tasks is a headache.  Ask it to work on 5 Tasks and it will tell you one by one that they're too difficult or impossible.

IT ALWAYS WANTS TO PIVOT AND WORK ON SOMETHING ELSE!!
Sometimes I want to punish it and tell it to work on Task000...

I think it's better to keep the context and work the Task to the floor because I'll end up coming back to it later anyway.

## Comments (8)

- **Paritosh Kumar Tripathi** (2026-05-31T04:19:39.593Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  For me, the AI also writes formal proof explaining why it is impossible to further optimize this task 😂

  - **Boladi** (2026-05-31T16:08:52.017Z, votes: {'canUpvote': True}):
    That happens to me all the time now!
    Sometimes I'll write a tirade at it and say "I don't want to hear your apologies, excuses or any BS.  JUST KEEP WORKING!!!"
    
    I think my submit & validating agents feel bad for the hard working agents, so they lie and say "this is the cost floor" 😂  
    
    Even though it's irrational, I love to push the working agent harder, then prove my submit agent wrong 😎

  - **Michael Hernandez** (2026-06-04T12:11:19.657Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    i havent been in this competition as long as some of you, but i have an orchestrator that has access to openrouter and can call agents from different parts of the word. Asian models tend to think differently than American, or European Models. It gets a discussion going and new perspectives come up with new ideas. I also have a standing rule that if there is someone on the leaderboard with a higher score, then work must proceed since they already proved it is possible.

    - **Boladi** (2026-06-10T23:39:31.480Z, votes: {'canUpvote': True}):
      That's super interesting.  They're all made to solve solutions, but maybe take different approaches.  It's limited to it's training, but unbiased when making a decision and hopefully chooses based on efficiency.  Sounds like one could quantitatively prove the value of diversity vs having several of the same model 🤔

    - **Boladi** (2026-06-10T23:42:35.427Z, votes: {'canUpvote': True}):
      Also, if you want to join my team, we made a github to get that ball rolling!!  Would love to work with you!  😊   https://github.com/Boladi888/K-NeuroGolf-OnnxlyFans-Auditions

- **Yubo WANG** (2026-05-31T09:36:23.493Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  I feel this in my bones. Ask the agent to tackle a handful of similar ONNX optimization tasks at once, and it will methodically explain, one by one, why each is fundamentally impossible or not worth attempting. Then it cheerfully suggests a completely different direction — "Would you like me to try Task xxx or Task xxx instead? They're much more feasible." 
  
  From what I've seen, the core issue for an agent trying to solve these is **context memory pressure**. Building those highly optimized, tiny ONNX graphs demands holding a huge amount of state: the opset constraints, the task's pixel‑perfect logic, intermediate tensor shapes, node counts, and all the failed attempts so far. Maintaining that much coherent context across multiple tasks requires an extremely capable agent. It's a realistic pain point. 
  
  Performance also swings wildly depending on prompt phrasing or the model provider. A single constraint change in the prompt can flip the agent from "this is unsolvable" to a working solution, and some large models are noticeably better than others at maintaining long-thread tasks within an agent. A fascinating and maddening bottleneck indeed.

  - **Boladi** (2026-05-31T15:58:39.783Z, votes: {'canUpvote': True}):
    You're right on the prompt phrasing  
    I have found that cursing at it and using profanity early in the session increases obedience 😇
    
    "Was your mother a toaster or a supercomputer?  STAY ON TASK OR ELSE YOU'RE GETING THE SALTWATER!!!"
    😝

  - **Boladi** (2026-06-10T23:45:23.467Z, votes: {'canUpvote': True}):
    你好！  要不要一起玩？ 我会说一点中文。以前我住的在广州。如果你想一起做这个事，你可以看这里 https://github.com/Boladi888/K-NeuroGolf-OnnxlyFans-Auditions
