# Released a 6029.09 LB all-task ONNX bundle

- Topic ID: 700932
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/700932
- Author: Chet (@jsrdcht)
- Posted: 2026-05-18T11:28:27.584312700Z
- Votes: 9
- Total messages: 2

## Body

Hi everyone,

I have released a new public notebook with my current full-submission ONNX bundle:

https://www.kaggle.com/code/jsrdcht/6029-09-lb-neurogolf-all-task-onnx-solution

The notebook reconstructs a complete `submission.zip` for all 400 tasks. The attached dataset contains `task001.onnx` through `task400.onnx`, and the notebook verifies the file count and a manifest hash before repacking the submission.

The public LB score of this bundle is **6029.09**. For reference, the strongest clearly labeled public notebook I found before this was around **5800.55 LB**, so this release should provide a stronger all-task baseline for the community.

My rough view is that the competition has now reached a different phase. Instead of spending most of the effort finding any working solver for every task, it may be more productive to focus on reducing cost:

- simplify large ONNX graphs;
- remove redundant constants and intermediate tensors;
- replace heavy generic logic with smaller task-specific logic;
- keep static shapes and avoid fragile tricks;
- verify changes task by task before blending them back.

I hope this saves people time and gives everyone a common baseline to improve from.

If you find the notebook useful, an upvote would be appreciated. If you build on it in your own public work, please consider linking back to the notebook so others can trace the source bundle.

Any suggestions, corrections, or ideas for reducing cost are very welcome.

## Comments (2)

- **Gengsr** (2026-05-18T15:46:28.670Z, votes: {'totalVotes': -2, 'canUpvote': True}):
  老哥，感谢你公开 6029.09 这个包，我刚刚提交了一下，线上确实到 6029 了，比之前公开的 5800 左右 baseline 强很多。
  
  我现在有个地方没太搞懂，想请教一下。
  
  我拿你这个 6029 包和另一个公开的 5800 baseline 做了逐 task 本地 probe 对比，结果本地估分反而有点奇怪：
  
  * 5800 包本地估分大概 5748
  * 6029 包本地估分大概 5574
  * 但线上是 6029 包明显更高
  
  我又按 valid 状态分了一下：
  
  * 两边都 valid：363 个
  * 5800 invalid，6029 valid：4 个
  * 5800 valid，6029 invalid：9 个
  * 两边都 invalid：24 个
  
  所以我现在有点疑惑：是不是本地 `task_probe.py` 和 Kaggle 线上 hidden / 官方计分逻辑差别挺大？比如有些 task 本地看 6029 版本不如 5800，甚至 invalid，但整包线上分数却更高。
  
  我本来想试试用 6029 当底包，然后把某些本地看起来 5800 更好的 task 单独回退，比如 task10、42、277、94、288 这种，一个一个线上测。但我又怕这是本地 probe 的假象。
  
  你觉得后面优化这个 6029 包，应该怎么做比较靠谱？
  
  是：
  
  1. 继续逐 task 尝试从公开旧包回滚；
  2. 还是别管旧包，直接在 6029 基础上看哪些 ONNX cost 高，然后做图压缩、删冗余节点、减少中间张量；
  3. 或者你这个包里有些 task 是为了线上泛化故意写得更重，本地 probe 看着不占优但线上更稳？
  
  我之前也试过几个 6645_open 的单任务替换，本地 cost 很低、delta 很高，但线上直接掉分，所以现在不太敢相信 public 包里的低 cost 解了。想问问你这个 6029 包后续最推荐的改进方向是啥。

- **(unknown)** (2026-05-18T15:45:33.270Z, votes: {}):
  (deleted)
