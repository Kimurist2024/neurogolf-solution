# Caution: clean up onnxruntime_profile files during local experiments

- Topic ID: 697697
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/697697
- Author: Takuya Inoue (@takuyainoue)
- Posted: 2026-05-06T23:07:58.537209100Z
- Votes: 4
- Total messages: 

## Body

[This post was partially written with help from Codex.]

A small caution for anyone running local verification or profiling scripts.

The updated scoring flow uses ONNX Runtime profiling traces via session.end_profiling(). If your local scripts save every onnxruntime_profile JSON with a timestamp or unique suffix and never delete them afterward, these files can accumulate very quickly.

In my case, my Codex-generated local scripts kept all timestamped onnxruntime_profile_*.json files under .cache/. After only a few rounds of local verification experiments, those profiling traces grew to more than 150GB and consumed most of the storage on my Mac.

## Comments (0)

(no comments)
