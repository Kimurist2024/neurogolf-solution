# How are you calculating MACs locally before ONNX export?

- Topic ID: 691961
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/691961
- Author: fleure (@nelnomial)
- Posted: 2026-04-16T02:18:01.107947900Z
- Votes: 1
- Total messages: 2

## Body

Hi everyone!
This is my first time diving deep into optimizing network costs like this, and I'm really enjoying the challenge so far!

I understand how to count the number of parameters and estimate the memory footprint of my PyTorch models. However, as a beginner, I'm struggling with calculating the exact MACs locally before going through the trouble of exporting to ONNX and testing it.

I have a couple of questions:

1. Are there any specific Python libraries you recommend for this? (I've heard of tools like thop or ptflops, but I'm not sure which one matches the official ONNX evaluation best).

2. How are you integrating this into your local training loop?

## Comments (2)

- **Kameron Kilchrist** (2026-04-19T00:05:13.950Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  data/neurogolf_utils/neurogolf_utils.py -> score_network(m)
  
  Also important to use this tool because there are some edge case models that work but crash the scoring function -- which Kaggle then rejects as invalid onnx

  - **fleure** (2026-04-20T01:34:16.530Z, votes: {'canUpvote': True}):
    That completely answers it. Thank you, Kameron!
    I didn't realize the backend scoring function could actually crash on valid models. I'll stop relying on local estimates and make sure to validate every candidate through score_network(m) before exporting. Much appreciated!
