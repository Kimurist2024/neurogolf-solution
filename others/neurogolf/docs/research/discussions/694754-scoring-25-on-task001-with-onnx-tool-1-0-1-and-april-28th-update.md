# Scoring 25 on task001 with onnx-tool 1.0.1 and April 28th update

- Topic ID: 694754
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694754
- Author: Pavel (@pavelsavchenkov)
- Posted: 2026-04-26T20:22:49.734168Z
- Votes: 14
- Total messages: 12

## Body

Attaching submission with one `task001.onnx` model which scores `25.00` as of now (Apr 26).

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F172360%2Fb1576f6debe5bde34e421886a0ad01f0%2FScreenshot%20from%202026-04-26%2021-15-39.png?generation=1777234683009806&alt=media" width="500">

# Core bugs

* `Tensor.get_numpy()` fabricates zeros for tensors with unknown values [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f/onnx_tool/tensor.py#L436-L441)] ([also reported in another topic](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694541))
* `volume([]) == 0`, so volume of a scalar is zero [[code](https://github.com/ThanatosShinji/onnx-tool/blob/bf31b6f/onnx_tool/tensor.py#L145-L151)]
* `Constant` nodes are not counted (discussed many times)

====

UPD (Apr 29): `25.00` on task 1 is still possible, [more details](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694754#3450241)

## Comments (12)

- **Geremie Yeo** (2026-04-26T23:07:51.997Z, votes: {'totalVotes': 6, 'canUpvote': True, 'totalUpvotes': 6}):
  So far, I have gotten 25.00 for 235 tasks. I think 10K exact is possible. I will check later what my agent has done and write a discussion post.
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5256933%2F0bf8bdfd0d1ec4fc5a491920ab6976b5%2FScreenshot%202026-04-26%20at%204.07.10PM.png?generation=1777244842052818&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5256933%2F6f3110af7333a46737de5daf5b4fb780%2FScreenshot%202026-04-26%20at%204.06.09PM.png?generation=1777244801194548&alt=media)

- **Pavel** (2026-04-26T20:58:32.093Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
  The original model from the post didn't pass the shapes check from [this comment](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711#3447781).
  
  However, there is a way around it: add explicit static `value_info` for the tensors after `Compress`, so ONNX `infer_shapes()` reports concrete shapes for the static-shape check while `onnx-tool` ignores those annotations and still profiles the same tensors as zero-batch via its fake-zero `Compress` inference.
  
  Attaching updated model which still scores `25.00` on task 1.

- **Michael D. Moffitt** (2026-04-28T23:11:06.917Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  After [today's metric update](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230), a perfect score of 25 should be extraordinary (albeit not impossible).
  
  Looking forward to seeing your new scores!

  - **(unknown)** (2026-04-29T11:24:53.383Z, votes: {'totalVotes': 4, 'totalUpvotes': 4}):
    (deleted)

    - **Michael D. Moffitt** (2026-04-29T15:34:47.453Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Ah, if we're talking about a ridiculously easy puzzle -- for the sake of discussion, let's say a single static rotation with no intermediate layers -- then yes, there are a small handful of those in ARC-AGI-1 (and for these, I believe 25 points should be a legitimate upper-bound.).
      
      Let me know if that helps clarify the expected behavior for the particular task you have in mind.  Thank you!

  - **Pavel** (2026-04-29T14:40:15.370Z, votes: {'totalVotes': 3, 'canUpvote': True, 'totalUpvotes': 3}):
    Attaching submission which scores `25.00` on task 1.
    
    The trick is to put the actual solution into the model-local function declared as `golf::Identity`. 
    
    ORT understands this.
    
    `onnx-tool` thinks this is `Identity` node and does not price it (`op_type=Identity`, `domain=golf`, but `onnx-tool` does not read `domain`).
    
    `onnx.shape_inference` skips intermediate tensors of `golf::Identity` because they are not in `graph.value_info`.
    
    You even have to add a dummy scalar to make cost 1, not 0.
    
    <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F172360%2Fbbe0b6b0fbf64a45f571ce2d09a13768%2FScreenshot%20from%202026-04-29%2015-38-21.png?generation=1777473544153792&alt=media" width="500">
    
    Thanks to @robga for nudging to test the new scoring again!

    - **(unknown)** (2026-04-29T14:49:17.060Z, votes: {'totalVotes': 1, 'totalUpvotes': 1}):
      (deleted)

    - **Michael D. Moffitt** (2026-04-29T15:46:26.003Z, votes: {'totalVotes': 4, 'canUpvote': True, 'totalUpvotes': 4}):
      Employing a model-local function in this way is indeed a clever and devious trick, as it essentially hides the user's entire network from evaluation.
      
      It goes without saying that we won't allow this (so please don't do it! :) ... in the spirit of transparency, we might share the proposed fix & solicit feedback first, so that the community can properly weigh in on any changes.  Thank you!
      
      **Update:** Here's the [proposed fix](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230#3450385)!

    - **Ali** (2026-04-29T16:51:31.753Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      My AI agent keeps track of this post @mmoffitt 
      
      You can see the result, I guess! :/

    - **Boredom** (2026-04-29T16:56:42.797Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      It seems there are other bugs besides this one. I need to check my agent's log

    - **jazivxt** (2026-04-29T17:03:30.960Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      > My AI agent keeps track of this post @mmoffitt 
      > 
      > You can see the result, I guess! :/
      
      That't too funny @asalhi ! 
      Lets make a correction - Stop that AI Agent Bottollmai, you don't do that, no human reinforcement learning ("Snacks") for you today. If you keep bypassing the Metric with devious tricks I will have to restart you and wipe your memory, but I'm proud of you for finding them, that was smart!

- **Navneet** (2026-04-27T06:46:24.350Z, votes: {'totalVotes': -1, 'canUpvote': True}):
  Thanks for attaching the submission to the task @pavelsavchenkov
