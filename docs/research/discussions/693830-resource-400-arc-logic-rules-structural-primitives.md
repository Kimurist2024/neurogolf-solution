# [Resource] 400+ ARC Logic Rules & Structural Primitives

- Topic ID: 693830
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693830
- Author: Karnakbayev Artur (@karnakbaevarthur)
- Posted: 2026-04-22T13:46:37.368381600Z
- Votes: 3
- Total messages: 1

## Body

I’ve released two datasets to streamline reasoning for ARC-AGI and NeuroGolf 2026:

[Logic Decoder](https://www.kaggle.com/datasets/karnakbaevarthur/logic-for-each-arc-task): Text-based transformation rules for all 400 tasks, decoded via DeepSeek-V3/R1. Perfect for RAG pipelines and synthetic data generation.

[Transformation Library](https://www.kaggle.com/datasets/karnakbaevarthur/neurogolf-2026-task-transformation-library): A metadata layer mapping tasks to "Logical Primitives" (Gravity, Tiling, etc.) with complexity scores. Ideal for routing tasks to specific ONNX-based solvers.

Use cases: 
1. Task Clustering: Group tasks by Primary_Category to reuse model heads.
2. Complexity Scaling: Prioritize development based on the 1-10 difficulty mapping.
3. Dynamic Routing: Use primary categories to trigger specific ONNX feature extractors (e.g., only run "Gravity" kernels when needed).

Hope these help your pipeline! Good luck with the grids. 🧠

## Comments (1)

- **NNMax** (2026-04-23T05:32:45.900Z, votes: {'canUpvote': True}):
  Thanks for the datasets, however while inspecting it for some tasks I found that the logic was not clear enough. It didn't clearly mention all constraints, edge cases and the exact logic was also seems a little less detailed.
  
  For now, I only looked at tasks 363 and 364 in your dataset.
