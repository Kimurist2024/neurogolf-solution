# Question about how submissions are scored

- Topic ID: 703200
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/703200
- Author: syx777 (@syx777)
- Posted: 2026-05-29T06:21:44.367394300Z
- Votes: 2
- Total messages: 2

## Body

I'm a newcomer to this competition and have a question about how submissions are scored. Apologies in  advance if this is already covered in the rules and I missed it.    

I noticed something that surprised me: I submitted an ONNX that does not pass all examples in the official  extracted/taskNNN.json files locally — specifically, it fails on a small number of arc-gen examples for one  task — yet the leaderboard still awarded full points for that task as if it were 100% correct.    

My questions:   

1. Is this the intended behavior and sufficient for full marks on a  task?  
2. Will there be a separate, stricter re-evaluation later (e.g., a private leaderboard at competition close) that also checks arc-gen or some unseen test set, which could change rankings? 
3. If a submission scores well now but doesn't generalize to the full arc-gen distribution, would that be considered a valid finish or could it be invalidated?    

I want to make sure I'm building toward something that will actually count, rather than optimizing against  a partial check. 

Thanks!

## Comments (2)

- **jacekwl** (2026-05-29T07:03:17.297Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Probably those examples have inputs larger than 30x30 and kaggle scorer ignores them because limit is 30x30. And there are a couple of tasks like this.
  
  There will be no re-evaluation. If it passes kaggle scorer, then it's good enough.

  - **syx777** (2026-05-29T07:50:37.297Z, votes: {'canUpvote': True}):
    Thank you for the explanation! That clears up my confusion.
