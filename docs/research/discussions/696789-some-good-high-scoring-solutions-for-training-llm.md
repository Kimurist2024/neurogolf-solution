# some good high scoring(?) solutions for training LLM

- Topic ID: 696789
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/696789
- Author: hengck23 (@hengck23)
- Posted: 2026-05-04T07:12:17.058483Z
- Votes: 2
- Total messages: 6

## Body

I really want a discussion on how to use LLM to give good solutions (and not good coding)   
 
Attached is "good-solution.zip", which contains solutions to 10 tasks of moderate difficulty. No profiler hacks. These are results from human+AI interaction (and a quarrel).

let's discuss:  
1) What are your agent scores for these tasks? under what automation?  
2) Can pure AI comes up high level solution that exploits data structure and hacks (e.g. geometry)  
3) How to make magic prompt to make high level solution?    
4) how to automate prompt improvement? (autoresearch for prompt)   
5) how to make agent learn/clone from human?   


| TaskID | Score  | Task (≤6 words)           | Solution (≤15 words)                                      |
|--------|--------|---------------------------|-----------------------------------------------------------|
| 102    | 15.415 | structured region mapping | reshape compact grid, apply mask, avoid full expansion     |
| 106    | 17.670 | flat pattern extraction   | flatten input, argmax decode, rebuild minimal structure    |
| 107    | 15.178 | directional pattern fill  | detect axis from edges, propagate pattern along direction  |
| 109    | 16.066 | reflection block pattern  | crop core block, double reflect, crop to target size       |
| 111    | 18.274 | object crop + classify    | argmax to label, crop bounding box, rebuild output compact |
| 114    | 17.672 | small block augmentation  | detect region, insert small patch, minimal broadcast       |
| 115    | 16.477 | region detection grid     | scan top/left edges, infer size, broadcast into grid       |
| 117    | 14.949 | multi-stage transform     | sequential transforms, reuse tensors, limit intermediate size |
| 119    | 15.696 | sparse scatter pattern    | compute indices, scatter minimal elements, avoid expansion |
| 120    | 15.082 | validity masked output    | argmax decode, build validity mask, selective write output |
|--------|--------|---------------------------|-----------------------------------------------------------|
| AVG    | 16.248 | —                         | —                                                         |

## Comments (6)

- **Rustam Bazarbayev** (2026-05-04T18:07:53.463Z, votes: {'canUpvote': True}):
  I am trying to new approach here my current results: 
  | Task   | Score  |
  |--------|--------|
  | 006    | 9.43   |
  | 016    | 10.95  |
  | 026    | 9.43   |
  | 052    | 8.85   |
  | 053    | 9.73   |
  | 056    | 9.73   |
  | 073    | 9.73   |
  | 078    | 9.43   |
  | 087    | 8.85   |

  - **hengck23** (2026-05-05T18:46:20.413Z, votes: {'canUpvote': True}):
    lets talk abot task006 and task026. I think these can get > 16.0. this is what you can tell your agent or LLM or AI coder.
    
    1) e.g. input is size 7x7 consisting of left side 3x3 and right side 3x3, separated by a pixel vertical line. output  = op(left,right) where op is OR, AND, MAX, etc.   
    2)please read arc-gen task generator py file to find out the operator, and what fixed colors of the input and output used.   
    3) you can tell your LLM  a solution is generally like:  
    - slice the input like this: left =[0,c,:h,:w], right=[0,c,:h,w+2:2*w+1],keep it small
    - cast to uint8
    - make ouput = op(left,righ), e.g. 1xhxw
    - recolor
    - onehot to 10xhxw
    - pad to 10x30x30
    
    
    -----
    automation:
    
    1) first find some good solutions onnx  
    2)then find the prompts that can generate the good solutions  
    3)then find automated ways to generate these prompts

    - **hengck23** (2026-05-05T18:58:23.347Z, votes: {'canUpvote': True}):
      https://github.com/google/ARC-GEN/blob/main/tasks/task_0520fde7.py
      
      ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fa921d6ce1f9f9e10b891a90e4ad8a541%2FSelection_3297.png?generation=1778007536098430&alt=media)
      
      ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5189f72a62269740657a280735accf73%2FSelection_3298.png?generation=1778007549442104&alt=media)
      ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fa6d7686ecb21e9e1fcad6b2d3c2b8003%2FSelection_3299.png?generation=1778007597281671&alt=media)

    - **Rustam Bazarbayev** (2026-05-06T02:19:25.410Z, votes: {'canUpvote': True}):
      I am trying to solve all problems with new approach using CNN

    - **hengck23** (2026-05-06T03:56:52.030Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
      Cnn solution would not have high score. In fact top solutions will not be using conv etc at all except for a few isolated cases. This competition is not about finding general solution. It is about synthesising onnx graph for KNOWN solution.

    - **Rustam Bazarbayev** (2026-05-06T04:31:59.367Z, votes: {'canUpvote': True}):
      Oh. Thank you for the clarification! I did misunderstand. Currently, I am making a CNN with logic gates to generate the solutions. You know logic gates are better to make operation on integer values
