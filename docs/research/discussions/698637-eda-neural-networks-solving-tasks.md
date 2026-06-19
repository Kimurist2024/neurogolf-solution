# EDA - Neural Networks Solving Tasks

- Topic ID: 698637
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/698637
- Author: Chris Deotte (@cdeotte)
- Posted: 2026-05-11T00:49:23.701472800Z
- Votes: 26
- Total messages: 4

## Body

Hi. It's pretty amazing that a neural network can convert the input grid to output grid. So I made some EDA notebooks (with the help of Codex) to help us visualize what the ONNX networks are doing to solve these ARC-AGI puzzles.

* Task 026 [here][1]
* Task 032 [here][2]
* Task 038 [here][3]
* Task 083 [here][4]
* Task 116 [here][5]
* Task 129 [here][6]

I chose a few different puzzles to illustrate a few different patterns that our neural networks need to learn. Enjoy!

# Task 116 Example

# Pattern to Learn - Mirror then Stack

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p1b.png)

# ONNX Solution

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p2.png)

# ONNX Layer 1

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p3.png)

# ONNX Layer 2

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p4.png)

# ONNX Layer 3

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p5.png)

# ONNX Layer 4

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p6.png)

# ONNX Layer 5

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/p7.png)

# Starter Notebook

View more fun examples [here][1], [here][2], [here][3], [here][4], [here][5], [here][6]. Enjoy! 


[1]: https://www.kaggle.com/code/cdeotte/eda-task-026
[2]: https://www.kaggle.com/code/cdeotte/eda-task-032
[3]: https://www.kaggle.com/code/cdeotte/eda-task-038
[4]: https://www.kaggle.com/code/cdeotte/eda-task-083
[5]: https://www.kaggle.com/code/cdeotte/eda-task-116
[6]: https://www.kaggle.com/code/cdeotte/eda-task-129

## Comments (4)

- **Navneet** (2026-05-12T07:08:36.807Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thanks for the Neural Networks Solving Tasks @cdeotte

- **Adithya Giridharan** (2026-05-11T12:51:43.110Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
  truly amazing..spoke my mind out with an actual implementation of visualisation ..thanks a ton! @cdeotte

  - **Chris Deotte** (2026-05-11T12:56:59.940Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
    Thanks. Before this visualization, i was a little confused how NN can make the transformation. But after displaying a few ONNX, i can see what is going on.
    
    This is one of the great things about LLM coding tools, they can help us visualize whatever we want to see in any data science project.

    - **jacekwl** (2026-05-11T13:15:16.427Z, votes: {'totalVotes': 2, 'canUpvote': True, 'totalUpvotes': 2}):
      I remember I initially thought we will have to train neural networks with all that gradient descent, epochs of training stuff etc.
      But then when I joined I realized that we are just creating symbolic solvers using tensors / ONNX operators which is much more approachable.
