# Help me understand the dataset

- Topic ID: 703755
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/703755
- Author: HADY (@johnhady101)
- Posted: 2026-06-01T21:29:59.870274700Z
- Votes: 0
- Total messages: 2

## Body

The input array here in the task is an 10x12 array, the output array is 4x4; does that mean that the size of the output is not necessarily the same as the size of the input? or Iam messing something?

## Comments (2)

- **Chris Deotte** (2026-06-01T21:59:16.760Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  All our ONNX graphs need to accept input [1, 10, 30, 30] and make output [1, 10, 30, 30]. This is batch size 1, and one hot encoding of color 10 and shape 30x30. The actual puzzles contained inside these shapes can actually be any size. For example for task31, it appears the true puzzle shape is AxB where A in [10,11,12] and B = 12. And the output is the crop shape of the non black shape. But our ONNX must pad and output [1, 10, 30, 30] and accept an input of [1, 10, 30, 30].
  
  ![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Jun-2026/task31.png)

- **Andrey Yunoshev** (2026-06-01T22:07:46.767Z, votes: {'canUpvote': True, 'totalUpvotes': 1}):
  Yes, the output size does not have to match the input size.
  
    In ARC-style tasks, each example gives an input grid and an output grid. The goal is to infer the transformation. Sometimes the output is the same size, but often it can be cropped, expanded, tiled, reduced, or otherwise reshaped.
  
    So a 10x12 input and a 4x4 output is normal. It usually means the task transformation extracts or constructs a smaller relevant pattern from the input.
