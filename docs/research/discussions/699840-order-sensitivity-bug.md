# order sensitivity bug?

- Topic ID: 699840
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/699840
- Author: robga (@robga)
- Posted: 2026-05-15T12:27:00.878383700Z
- Votes: 16
- Total messages: 14

## Body

I’ve encountered what appears to be order-sensitive scoring for submission zip members @mmoffitt 

The same set of ONNX files can receive different leaderboard scores depending only on the order of files inside the zip. In my case, taskAAA scores correctly alone and in smaller bundles, but contributes 0 when placed later in a different bundle/zip. It's happened with 2 tasks. Moving the exact same files to the front of the zip makes the full submission score match the locally predicted total. No ONNX bytes changed, only zip member order changed. Has anyone else noticed this? Normally I think of 0 as not meeting the hidden examples, but now I'm wondering if its something else and maybe they were OK but hit a bug. Just to be clear: if I submit 25 tasks in numeric order, one task unexpectedly scores 0. I bisect the 25 and both halves score OK. I can then repair the failing zip by moving that task to the front. So with my 2 "0" tasks, if I submit tasks001-400 in order I score less than if I put the specific 2 tasks at the front of the zip in which case they score as they do locally. 

Edit: cause was identified by helpful conrtibutors below, "the main issue was that the Conv bias length was smaller than the number of output channels ... After fixing the bias length, my submissions are stable." and this fixed it for me too.

## Comments (14)

- **Michael D. Moffitt** (2026-05-25T16:21:56.210Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  ***[Cross-posting this from the [other thread](https://www.kaggle.com/competitions/neurogolf-2026/discussion/702256#3462580), and will edit with updates once they become available.]***
  
  I've reported a possible memory contamination bug to the ONNX Runtime developers here: https://github.com/microsoft/onnxruntime/issues/28654
  
  If others have additional info that might be relevant, please let me know -- many thanks!
  
  ---
  
  ***[2026-05-27 Update]***: Our scorer appears to be working properly, provided that all Conv operations in submitted networks exhibit the proper length & bias values.  However, for contestants who are running their experiments locally on ARM64, two other things may be needed:
   - An upgrade to ONNX Runtime v1.26.0
   - An additional session option: `options.add_session_config_entry("mlas.disable_kleidiai", "1")`

- **keymoon** (2026-05-15T21:41:22.577Z, votes: {'totalVotes': 10, 'canUpvote': True, 'totalUpvotes': 10}):
  I experienced the same issue: submitting the exact same zip file multiple times resulted in different scores across submissions.
  
  The likely cause is **undefined behavior being triggered in ORT.**
  
  There appears to be an ORT bug involving out-of-bounds memory access, where it may read from uninitialized regions of the heap.
  
  My guess is that this became more noticeable after the judging environment changed to reuse the same Python process instead of restarting it for every run. If ORT runs early in the process lifetime, the relevant memory region may still be filled with zeros, so the model appears to behave as expected. However, if it runs later, that memory is more likely to contain data written by previous executions, causing ORT to read stale values and return incorrect results.
  
  As a temporary workaround, it may help to run the local judge multiple times without restarting the process, and reject the submission if any run produces a wrong answer. In my case, the main issue was that the Conv bias length was smaller than the number of output channels.
  
  Whether this kind of undefined behavior should be accepted as a valid solution probably needs separate discussion.

  - **Michael D. Moffitt** (2026-05-15T22:26:37.210Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
    > There appears to be an ORT bug involving out-of-bounds memory access
    
    I had (reluctantly) arrived at the same conclusion, so I'm glad to hear that someone else is seeing the same thing.  We probably won't be able to deploy any fixes before the weekend, but I'll give it some thought and hopefully we'll have some kind of workaround ready soon!

  - **Yiheng Wang** (2026-05-15T23:54:47.663Z, votes: {'totalVotes': 5, 'canUpvote': True, 'totalUpvotes': 5}):
    ```
    the main issue was that the Conv bias length was smaller than the number of output channels.
    ```
    Confirmed that I met exactly the same issue when using this method, after fixing the bias length things, my submissions are stable

    - **David Austin** (2026-05-16T15:14:02.997Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      Same for me as well

  - **PRASHANT SHUKLA91** (2026-05-18T08:05:30.583Z, votes: {'canUpvote': True}):
    Can you tell me your submission.zip file size? Is it same in every run?

- **Chris Deotte** (2026-05-22T03:04:42.213Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  This also happened to me. It was frustrating to debug. Fixing the Conv's fixed the issue.

- **Ali** (2026-05-15T19:18:58.007Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  I am not sure if related, but I just discovered that my best sub, which is 6096.94 , now scores 6071.01 
  My auto agent spent the whole day rejecting tasks because no single task beat the 6096.94 because the score was calculated at 6071, reference :-( 
  
  It seems the last metric update (yesterday, I guess) didn't run the whole LB! causing this mess. 
  @mmoffitt

  - **(unknown)** (2026-05-15T19:36:41.353Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
    (deleted)

  - **Michael D. Moffitt** (2026-05-15T22:38:32.253Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
    @asalhi I'm afraid yours seems to be a different issue (check your networks for tasks #118 and #243 ... they fail at `line 242` of the new defensive check in `neurogolf_utils.py`).

    - **Ali** (2026-05-15T22:50:34.543Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Thanks. 
      I will have a look. The main problem was that I didn't know it was failing the new update because the old best was still "best" on LB, and as I said, the agent wasn't able to find an update that passed the LB score back then.
      
      It kept asking me every while: Are you sure you are scoring 6096.94 ? Of course, I say hhhh 
      
      Anyway its now submitting every tiny small improvment (becoming sensitive or punishing me), which I should stop before eating my subs. 
      
      I am so enjoying this competition :-)

- **Navneet** (2026-05-16T06:53:31.417Z, votes: {'totalVotes': -4, 'canUpvote': True}):
  Thanks for the informative order sensitivity bug @robga

- **Chan Kha Vu** (2026-05-17T08:34:49.933Z, votes: {'canUpvote': True}):
  I think I'm hitting the same issue. These 2 submissions are identical, but the score is different:
  ![](https://i.imgur.com/ZAInPt0.png)
  
  Each task has identical SHA hashsum, confirmed by my clanker:
  ![](blob:https://imgur.com/950bd66d-5bc7-4f54-bd23-3fad85ade8c9)

  - **Michael D. Moffitt** (2026-05-18T00:29:07.610Z, votes: {'canUpvote': True}):
    Might be worth checking to see if @yiheng's note above (*"the main issue was that the Conv bias length was smaller than the number of output channels."*) applies to your solution for Task #294.  Let me know what you find out!
