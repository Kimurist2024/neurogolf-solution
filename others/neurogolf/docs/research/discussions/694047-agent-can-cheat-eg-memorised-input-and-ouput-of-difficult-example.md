# Agent can cheat!!! (eg memorised input and ouput of difficult example)

- Topic ID: 694047
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694047
- Author: hengck23 (@hengck23)
- Posted: 2026-04-23T04:42:12.154856900Z
- Votes: 4
- Total messages: 5

## Body

i wonder how to the organizer is going to eliminate cheating?

In my agent training, it is possible that the agent examples for all examples except for one in a difficult task. Then he can just memorise the input  and ouput pattern (or some fix/patch) for that case.

Is there some hidden test pattern in the server? if not, do we consider this a valid program solution or cheating?

## Comments (5)

- **Kawchar Husain** (2026-04-23T04:49:03.747Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  >In addition, our official scoring metric will also employ a **private dataset** (containing a smaller number of examples per task) when validating these networks, so as to prevent overfitting.

  - **(unknown)** (2026-04-23T05:10:51.700Z, votes: {}):
    (deleted)

- **shwesh** (2026-04-23T04:51:05.317Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  Isn't the goal for this competition to make a model that memorizes the input and output pattern as much as possible without generalizing? That would be the optimal golf solution.

  - **hengck23** (2026-04-23T04:56:34.550Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    not memorise. the goal is to synthesize a specific program.
    
    i give you one example:
    
    ```
    correct logic is y= 2*x
    
    examples data: (x,y) = (1,2),(2,4)
    
    possible correct solutions:
    1)  y=2x graph
    2) y= LUT(x) where LUT entries are y=2x #correct memoristion
    
    
    so if hidden data is x=3, all the above models are correct
    
    
    wrong solution
    1)  y=x*x graph
    then he cheat: if x==1,y=2
    
    so if hidden data is x=3, this model is arong
    
    
    ```

    - **hengck23** (2026-04-23T05:00:13.410Z, votes: {'canUpvote': True}):
      ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fa3a64b036159f74da8c6c54ff3473583%2FSelection_3039.png?generation=1776920411258192&alt=media)
