# Task032: Your network performance could not be measured

- Topic ID: 702485
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/702485
- Author: Paolo Antonuccio (@paoloantonuccio)
- Posted: 2026-05-24T07:51:15.528771900Z
- Votes: 0
- Total messages: 2

## Body

Results on ARC-AGI examples: 4 pass, 0 fail
Results on ARC-GEN examples: 262 pass, 0 fail

Error: Your network performance could not be measured

/kaggle/input/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py in verify_network(network, task_num, examples)
    505   if memory is None or params is None:
    506     print("Error: Your network performance could not be measured")
--> 507   if memory < 0 or params < 0:
    508     print("Error: Your network performance could not be measured")
    509   elif arc_agi_wrong + arc_gen_wrong == 0:

TypeError: '<' not supported between instances of 'NoneType' and 'int'

Why this?

## Comments (2)

- **Michael D. Moffitt** (2026-05-24T12:14:24.733Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  The issue with this one is that your `no_out_0` tensor has a `dim_param` field, which we've disallowed (as it implies a dynamic shape).
  
  The `calculate_memory()` function in `neurogolf_utils.py` silently returns `None` in such cases, but you can augment it to print debug info if a similar problem comes up again.

  - **Paolo Antonuccio** (2026-05-26T08:08:45.557Z, votes: {'canUpvote': True}):
    Okay, I see that was the problem. So the shapes need to be defined in the initializers?
