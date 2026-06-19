# Question about public score vs hidden evaluation

- Topic ID: 705448
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/705448
- Author: akihiro (@akihirokkkkk)
- Posted: 2026-06-10T08:42:53.525302500Z
- Votes: 2
- Total messages: 21

## Body

I’d like to confirm whether the public leaderboard score is the actual final score for this competition, or whether there is any hidden/private evaluation set that may affect the final ranking.

This matters because it changes the optimization strategy.

If the public score is final, then highly specialized solvers that fit the visible tasks very well seem reasonable. But if there is hidden data, then a solver that performs well on the available training/public examples might still be overfitting and fail to generalize.

So I’m wondering what the intended direction is:

Should we focus on general rule-based solvers that are robust beyond the visible examples, or is it acceptable to optimize directly for the public task distribution, even if the solution is highly specialized?

I’d appreciate any clarification.

## Comments (21)

- **Jan Vorel** (2026-06-10T09:11:40.063Z, votes: {'canUpvote': True}):
  From first page Overview: 
  
  Evaluation
  
  For any of the 400 tasks in the ARC-AGI public training v1 benchmark suite, your team will earn a score of max(1, 25 - ln(cost)) for a functionally correct network whose cost is the sum of the following:
  
  The total number of parameters in the network
  The total memory footprint of the network (in bytes)
  **Functional correctness will be determined by validating the network against the original ARC-AGI benchmarks and a small private benchmark suite (so as to prevent teams from overfitting their solutions). To be eligible for points, your network must produce correct results across all of these tests.** - so it is not final, and overfitting should be avoided

  - **akihiro** (2026-06-10T09:20:28.473Z, votes: {'canUpvote': True}):
    @evolvion 
    
    Thanks, that makes sense. I had missed that part in the Overview.
    
    So the public leaderboard is not necessarily the full final evaluation, and functional correctness is also checked against a small private benchmark suite to prevent overfitting.
    
    That clarifies the intended direction for me: solvers should not only fit the visible public examples, but should remain general enough to pass unseen variants as well.
    
    Thanks for pointing it out.

  - **Fritz Cremer** (2026-06-10T09:32:46.783Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    I understood it differently: We can't overfit on the tasks visible to us, as there are more tasks on LB. But these additional tasks won't change between public/private LB. On the leaderboard page it states:
    "This leaderboard is calculated with all of the test data." which would imply scores won't change. But clarification here would be great as this is a very important detail @mmoffitt

    - **Jan Vorel** (2026-06-10T10:21:46.877Z, votes: {'canUpvote': True}):
      In my opinion test data is only test cases defined in original task definition file. It is not final dataset that will be used for final testing. But lets organisers clarify

    - **robga** (2026-06-10T11:47:05.083Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
      Organiser has said "Since the LB includes all tasks—including private—the scores you see there are all 100% verified (and there won't be a separate leaderboard after the competition deadline has passed)."

  - **Michael D. Moffitt** (2026-06-10T12:35:29.857Z, votes: {'totalVotes': 10, 'canUpvote': True, 'totalUpvotes': 10}):
    Let me know if this clarification is sufficient: there is indeed a small private benchmark suite (to prevent overfitting to the public ARC-GEN examples), but our current LB is **already incorporating** that hidden benchmark suite.
    
    In other words: you don't need to worry about a separate, post-competition LB that might introduce unexpected surprises — what you're seeing now is all that counts for the final rankings.
    
    +cc: @evolvion @akihirokkkkk @fritzcremer @robga

    - **Fritz Cremer** (2026-06-10T12:46:08.993Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Thanks, that definitely clears it up!

    - **Jan Vorel** (2026-06-10T13:27:25.787Z, votes: {'canUpvote': True}):
      I just spend last 3 days solving edge cases that pass here and do not pass locally on my private dataset, preparing 2 sets to submit .. My bad ... back to point hunting

    - **SebastianGil00** (2026-06-10T13:30:14.233Z, votes: {'canUpvote': True}):
      Hey Michael, when the hidden benchmark validates a network, how is each test input grid placed into the [1,10,30,30] tensor? Is the content always anchored at the top-left corner exactly like the public ARC-GEN examples, or do the hidden tests include translated, resized, or padded placements of the same grid? This determines whether a correct network must be fully position-invariant, or only correct for top-left-anchored content.

    - **Jokrasa** (2026-06-10T14:16:54.413Z, votes: {'canUpvote': True}):
      Are you planning to fix the order sensitivity bug mentioned here https://www.kaggle.com/competitions/neurogolf-2026/discussion/699840 ? -> If we try multiple orders and find one that lets the task pass, will this work for the final leaderboard or will there be rescoring / a later fix for this?

    - **Michael D. Moffitt** (2026-06-10T14:42:44.333Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      @sebastiangil00:
      
      > Is the content always anchored at the top-left corner exactly like the public ARC-GEN examples
      
      Yep that's right.  Essentially, we're just running ARC-GEN a few extra times, and then creating the [1,10,30,30] tensors in exactly the same way as the others.
      
      @jonaskreutz:
      
      > Are you planning to fix the order sensitivity bug mentioned here?
      
      The current evidence suggests that order sensitivity issues can be avoided if teams fix their convolutions (so, we're inclined to keep things as they are). Another option is to upgrade to ONNX Runtime 1.26.0, but such changes can be quite disruptive, and might even introduce new bugs.  We're open to feedback either way!

    - **SebastianGil00** (2026-06-10T16:44:41.203Z, votes: {'canUpvote': True}):
      @mmoffitt  Thanks Michael, that clears up the input placement. A follow-up on the constraint side: the rules require all tensors to have statically-defined shapes, but you also mentioned wanting to preserve dynamic slicing. How should a network reconcile those? Concretely, is a data-dependent Slice (where the crop size depends on the input content, so the sliced tensor's shape is not a compile-time constant) permitted by the validator, or must any crop region be a fixed compile-time constant with its shape patched into value_info? This determines whether we can crop to the actual content bounding box versus only a fixed top-left window.

    - **Michael D. Moffitt** (2026-06-10T17:38:54.400Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      > the rules require all tensors to have statically-defined shapes, but you also mentioned wanting to preserve dynamic slicing
      
      The main thing is to make sure that no tensors contain the `dim_param` attribute.  I believe it's possible to honor that constraint, while also using tricks that allow specific regions to be selected at runtime (there's a relevant topic about this [here](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695972)).

    - **SebastianGil00** (2026-06-10T19:28:10.443Z, votes: {'canUpvote': True}):
      Thanks @mmoffitt, that helps. One more on sizing: for a given task, are the public example grids a safe upper bound on the dimensions of the hidden ARC-GEN samples, or can a hidden sample be larger than any public example (up to 30x30)?

    - **Michael D. Moffitt** (2026-06-10T19:49:04.993Z, votes: {'canUpvote': True}):
      > can a hidden sample be larger than any public example (up to 30x30)?
      
      We'll constrain all our hidden tests to grids of size 30x30 or smaller.

    - **SebastianGil00** (2026-06-10T21:38:22.500Z, votes: {'canUpvote': True}):
      One more on the cost metric, if you don't mind. Sorry for all the questions; I’ve found all your answers very useful. Is a network's memory footprint computed purely from the statically-defined intermediate tensor shapes (summing each node's output tensor size), or does it also include any runtime or operator workspace memory used during execution?

    - **Michael D. Moffitt** (2026-06-10T22:31:22.993Z, votes: {'canUpvote': True}):
      > Is a network's memory footprint computed purely from the statically-defined intermediate tensor shapes
      
      Yes, that's basically it.  As a precaution, we also sweep though the actual shapes reported by ONNX Runtime, just in case there are mismatches compared to the shapes defined statically in the model.

    - **SebastianGil00** (2026-06-10T22:59:04.423Z, votes: {'canUpvote': True}):
      Two quick scoring-mechanics questions, grouped so they are easy to answer:
      
      1. Static vs runtime shape mismatch. When an intermediate tensor's statically declared shape does not match the shape ONNX Runtime actually produces at runtime, how is that resolved for scoring? Is the network rejected, or is the larger of the two shapes used for the memory calculation?
      
      2. Runtime limit. Is there a per-network or per-submission runtime or compute limit applied during scoring, and if a network exceeds it, is that task counted as incorrect?
      
      Thanks again for taking the time on these.

    - **Michael D. Moffitt** (2026-06-10T23:22:57.020Z, votes: {'canUpvote': True}):
      > is the larger of the two shapes used for the memory calculation?
      
      This is the policy that we currently have in place.
      
      > Is there a per-network or per-submission runtime or compute limit
      
      My understanding is that Kaggle imposes a 30 minute runtime per submission.

    - **SebastianGil00** (2026-06-11T09:57:56.480Z, votes: {'canUpvote': True}):
      Thanks @mmoffitt! Is there a theoretical or practical upper bound on the total score under the current metric? With 400 tasks capped at 25 points each, the absolute ceiling is 10000, but I imagine each task has some minimum representable cost. Do you have a sense of how high the achievable maximum realistically is?

    - **Chan Kha Vu** (2026-06-11T15:06:11.770Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
      @sebastiangil00 for the upper bound of the metric, it's up to us competitors to figure out, and perhaps surprise the organizers :) Maybe they're already surprised by top-1
