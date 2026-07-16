# LB 10000 - Convert any task into score 25 - [fixed]

- Topic ID: 696377
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/696377
- Author: Chris Deotte (@cdeotte)
- Posted: 2026-05-02T02:22:03.842874900Z
- Votes: 42
- Total messages: 16

## Body

My agent just discovered how to convert any task into score 25 (using the April 30 updated metric). FYI, I just emailed the host at `neurogolf.2026@gmail.com` explaining how to do it and proposing a fix.

## Comments (16)

- **Michael D. Moffitt** (2026-05-03T03:49:18.990Z, votes: {'totalVotes': 7, 'canUpvote': True, 'totalUpvotes': 7}):
  Many thanks to @cdeotte & the other teams who sent in bug reports and fixes!
  
  We're aiming for one big update + announcement + rescore (ideally Monday?) to keep things as simple as possible for teams.  A few tips so that folks are ready:
  - Make sure that `dim_value > 0` for every dimension of every tensor (ONNX's default model checker does not catch these)
  - To scrub models of "bogus" shape data, we'll likely add something like `model.graph.ClearField("value_info")` toward the beginning of `calculate_memory()`.  You can try adding this to your local `neurogolf_utils.py` to see how it behaves.
  
  Apologies for the bumpy start — this is the first time (to our knowledge) that a competition has solicited & evaluated networks in this manner, and ONNX's broad feature set unfortunately means a larger attack surface than we'd prefer.

  - **hengck23** (2026-05-03T04:40:36.557Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    `model.graph.ClearField("value_info")` may make your memory profiler unable to account for intermediate tensors produced by dynamic slicing.
    
    For deployment, it is common to replace symbolic dynamic dimensions with concrete `dim_value`s, e.g. with ONNX Runtime’s dynamic-shape-fixing tool:
    
    https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html  
    https://www.kaggle.com/competitions/neurogolf-2026/discussion/695972  
    
    So perhaps the profiler should distinguish between:
    1. invalid/bogus shape metadata, and
    2. intentionally fixed intermediate shapes needed for static analysis/profiling.
    
    Simply clearing ** all `value_info` may remove useful shape information  rather than only removing invalid shape data. **
    
    ---
    
    A more robust approach might be:
    - enforcing `dim_value > 0` (as mentioned), and
    - validating consistency between inferred shapes and actual graph operations, (rather than dropping shape metadata altogether.)
    
    ---
    
    I think dynamic slicing itself needs to remain supported; otherwise many legitimate ARC-style graph solutions become impossible, because crop sizes, object extents, and selected regions often depend on the input.
    
    The issue seems to be bogus `value_info`, not dynamic slicing. Ideally the profiler should still support valid dynamic-shape operators, while rejecting inconsistent or fake shape annotations.

  - **CPMP** (2026-05-03T11:53:45.960Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    I have not entered the competition hence this may be silly, but I saw the ONNX parameter counter https://github.com/gmalivenko/onnx-opcounter
    
    Sharing in case it might be useful.

  - **Suman Sharma** (2026-05-03T15:20:12.933Z, votes: {'canUpvote': True}):
    Thank you for the continued updates and transparency! Really appreciate the quick turnaround on fixes.
    Two things I wanted to flag that might help other teams:
    1. I'm seeing several networks score zero after the latest metric changes despite previously being valid. Could you share a definitive list of what causes zero scoring? Specifically:
    
    Which ONNX op types are explicitly banned (beyond Loop, Scan, NonZero, Unique, Script, Function, Compress)?
    Which op types are fully allowed and safe to use?
    Are there any ops that are technically not banned but cause issues due to onnx-tool profiling behavior (like ArgMax, TopK, OneHot having zero MACs)?
    Any structural patterns beyond the ones already documented that silently cause zero scores?
    
    A simple allowlist or denylist in the competition rules would save every team significant debugging time and make the competition much fairer for everyone.
    2. Regarding the dim_value > 0 requirement would it be possible for the validator to return a specific error message identifying which tensor and which dimension failed, rather than silently returning zero? Right now it's very hard to know whether a zero score is due to incorrectness, a banned op, a shape issue, or something else entirely.
    Thanks again for the transparency and responsiveness  the community feedback loop has been really excellent on this one.

- **hengck23** (2026-05-02T08:48:24.247Z, votes: {'totalVotes': 7, 'canUpvote': True, 'totalUpvotes': 8}):
  kaggle competition becoming a HACKathon

- **Geremie Yeo** (2026-05-02T06:36:48.750Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
  my agent did it in half an hour lol (reported via email as well)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5256933%2Ffe9ec3bb1274dd93067506885a2c1072%2FScreenshot%202026-05-01%20at%2011.31.36PM.png?generation=1777703579831747&alt=media)
  
  update: scaled to 10000
  
  edit: segmentation fault is not all you need

- **Jiwei Liu** (2026-05-02T02:23:18.600Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
  😂 your agent is good at reading discussions

- **robga** (2026-05-02T07:52:51.790Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
  It seems inevitable to me that the results of the challenge will rest on micro interpretations unique to onnx tool profiling and leaderboard signature evasion rather than the intended objective. These weaknesses are far under the surface vs obvious exploits and mostly hidden even to the human participants. Maybe what it needs is a kaggle provided dsl.

  - **Geremie Yeo** (2026-05-02T08:00:12.813Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    I wonder if the winning score will be `10000.0` at the end, surely not...

- **Boladi** (2026-05-02T23:49:40.203Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Does this happen a lot???  I just joined kaggle 😂

  - **Chris Deotte** (2026-05-02T23:53:03.287Z, votes: {'totalVotes': 6, 'canUpvote': True, 'totalUpvotes': 6}):
    Yes and no. In competitions that explore new types of metrics like this competition, then there are frequent updates to fix the metric. In competitions that do traditional modeling with traditional metrics, then it happens more rare.

- **Ali** (2026-05-02T06:43:05.960Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  I have two tasks scoring 25, but my agent is promising me that it didn't try something tricky! It never lied to me! 🫠

  - **Chris Deotte** (2026-05-02T10:03:38.610Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    Yes, for (at least) two tasks we can legitimately achieve score 25.

- **hengck23** (2026-05-02T08:44:15.417Z, votes: {'canUpvote': True, 'totalUpvotes': 1}):
  @robga "Maybe what it needs is a kaggle provided dsl."
  
  since agent can exploit the metric, it can also make a tool.
  Why don't use agent to make a resaonable  profile tool? I think it would not take too much time. we just need to build a list of op and how to compute memory and macs for each of them.

- **(unknown)** (2026-05-02T08:47:25.717Z, votes: {}):
  (deleted)

- **Navneet** (2026-05-04T07:05:43.690Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thanks for the agent @cdeotte
