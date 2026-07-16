# Questions to hosts

- Topic ID: 696569
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/696569
- Author: NNMax (@ashok205)
- Posted: 2026-05-02T20:34:26.871823600Z
- Votes: 11
- Total messages: 6

## Body

1. How would you detect if someone is using exploits/bugs but never disclosed it either publicly or privately?
2. What if someone uses exploits/bugs to some fixed set of tasks instead of all tasks? That will portray their score on leaderboard as legit but under the hood they had used exploits right?
3. New bugs and exploits will be discovered after each patches. Does this mean the longest leaderboard prize time gets reset after each patch? If no reset, does this mean the longest leaderboard prize is rewarding the ones who scored higher using exploits instead of those who scored higher without exploits?
4. Will a one time thorough audit of all combination of ops and functions along with other things that contribute to building and profiling of onnx graphs provide all potential exploits? ( Kind of not possible but still curious to hear the answer )
5. Like others suggested, will a kaggle provided DSL solve these issues?
6. Is it possible to overfit to private test sets as well? 
- Because a very few of my tasks scored legit scores but when validated locally against freshly generated synthetic pairs by the same scripts that was used to create the task pairs for all official tasks, I can see it score less than 100% pass rate. 
- This implies that private set is either a fixed one or has a pass rate percentage. If it is a fixed one, then wouldn't it be easier for some people to do seed hacking (although it's a nightmare) to get valid scores for overfitting tasks instead of genuine rule based solvers? 
- For some tasks I get **exhausted combinatorial space** message so can't validate them locally. 

@mmoffitt

## Comments (6)

- **Chris Deotte** (2026-05-02T20:39:54.163Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  I notice that the prize section says:
  > In the event the competition needs to be restarted, the Longest Leader dates shall be the new start and deadline of the competition.
  
  When we update the metric and rescore LB, is that considered "new start"? I hope so. Otherwise, it will be better for a team to use an exploit for 1 month to hold 1st place before reporting it. Therefore I suggest that a metric rescore counts as "new start".
  
  (As original post said, someone can find a new way to convert any task to 25. Then only change a few tasks to keep them in 1st place. If they drop to 2nd, they can change a few more tasks to 25. If rescore is "new start", there is no incentive for someone to do this)

- **Pavel** (2026-05-02T21:21:59.043Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  I think at this point there should be an additional award for whoever was the first to report a bug that prompted a rescore.
  
  This way there will be a big incentive to (1) search for catastrophic bugs and (2) report them asap, even for people far from the top.

  - **Chris Deotte** (2026-05-02T21:48:20.473Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
    I agree. We need to do all rescores ASAP (so we need encouragement to get them done quickly). It is frustrating to work for days and then have a rescore force us to change all our tasks.
    
    Personally, I am considering stopping working for 1 month and/or just searching for rescores for the hosts in the next 1 month. There isn't much reason to optimize onnx yet if more rescores are coming.

  - **(unknown)** (2026-05-02T22:12:48.853Z, votes: {'totalVotes': 3, 'totalUpvotes': 3}):
    (deleted)

- **Geremie Yeo** (2026-05-02T20:48:51.220Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  is the software so buggy that every metric fix introduces a new way to score `10000.0` ? 🤔

  - **(unknown)** (2026-05-02T21:10:04.763Z, votes: {}):
    (deleted)
