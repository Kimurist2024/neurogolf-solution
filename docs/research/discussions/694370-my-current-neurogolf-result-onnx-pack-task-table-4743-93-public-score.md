# My Current NeuroGolf Result: ONNX Pack + Task Table (4743.93 Public Score)

- Topic ID: 694370
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694370
- Author: thisray (@thisray)
- Posted: 2026-04-24T14:50:09.739651900Z
- Votes: 18
- Total messages: 6

## Body

Hello everyone,

I’m sharing my current intermediate NeuroGolf result, mainly:

- `submission.zip`
- `task_by_task_release_table.csv`

I hope this can help others save time, compute, and iteration cost when looking for more efficient ways to reduce ONNX model cost.

Public score: `4743.93`

The method itself is not especially interesting: I subscribed to ChatGPT Pro and used Codex with many parallel ㄖsubagents to search task-specific ONNX candidates. The submitted pack contains ONNX files for `394 / 400` tasks, then I spent most of the remaining effort optimizing cost family by family.

This is a mixed-lineage pack, and several public resources helped a lot. Many thanks to:

- `aliafzal9323/neurogolf-2026-tiny-onnx-solver`  
  https://www.kaggle.com/code/aliafzal9323/neurogolf-2026-tiny-onnx-solver

- `karnakbaevarthur/neurogolf-all-task-logic-complexity-map`  
  https://www.kaggle.com/code/karnakbaevarthur/neurogolf-all-task-logic-complexity-map

- `needless090/neurogolf-onnx-v31`  
  https://www.kaggle.com/datasets/needless090/neurogolf-onnx-v31

Current release artifacts:

- Dataset: https://www.kaggle.com/datasets/thisray/neurogolf-4743-93-submission-task-table
- Notebook: https://www.kaggle.com/code/thisray/neurogolf-4743-93-pack-inspection

Hope this helps others inspect the pack and find better cost reductions.

## Comments (6)

- **Bhawesh Sinha 07** (2026-04-26T02:52:55.843Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  That is amazing!

- **sigmaborov** (2026-04-24T15:55:33.940Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  its crazy sharing this 4800 solution , respect)

- **Tony Li** (2026-04-24T15:28:50.683Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Thanks for sharing!

- **Diveyam Mishra** (2026-04-25T17:10:04.377Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 2}):
  Notebook is not visible

  - **thisray** (2026-04-29T03:53:05.740Z, votes: {'canUpvote': True}):
    Thanks for the heads-up! The updated notebook is public now:
    
    https://www.kaggle.com/code/thisray/neurogolf-4808-21-post-apr-28-update

- **(unknown)** (2026-04-24T16:00:31.953Z, votes: {}):
  (deleted)
