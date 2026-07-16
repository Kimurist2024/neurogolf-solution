# Differential testing ONNX Tool

- Topic ID: 695251
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695251
- Author: bizy-coder (@benzydney)
- Posted: 2026-04-29T01:09:41.934346700Z
- Votes: 4
- Total messages: 3

## Body

The current approach to ONNX tool has been to manually curate bugs (sometimes by investigating what AI has done). This has already identified numerous bugs, but it is likely there are more "attack vectors" on other ops that may be less notable. I propose we use basic differential testing to improve this. To do this, we can use ORT (ONNX Runtime), which actually executes the model and produces ground-truth tensor shapes, dtypes, and sizes. Then we run a harness that runs every ONNX operator (opsets 11–18) through both `onnx_tool` and `onnxruntime` with a variety of different input types/shapes, then compares the two. Whenever ORT executes a model cleanly but `onnx_tool` reports a different shape / dtype / memory count, that's a confirmed bug. Currently this is only doing one operator nodes, but theoretically we could look at multi-operator networks if it is believed it oculd provide new vectors?

- **Scalar tensors counted as 0 bytes** — any op producing `shape=()` is reported as 0 bytes, even though the real tensor takes 1–8 bytes. Affects ~30 ops (`Abs`, `Cast`, `Constant`, `Conv`, `Equal`, `Greater`, `Identity`, `Relu`, `Sigmoid`, `Sub`, `Sum`, `Where`, …) — root cause is `volume([])` returning 0 instead of 1.
- **`Constant` node outputs not counted at all** — the produced tensor is in the graph but excluded from memory accounting, regardless of size. Compose this with the next bullet and a competitor gets a free large tensor.
- **`Round` reports output dtype as `bool`** — for any `float16/32/64` input, onnx_tool says the output is `bool`, undercounting bytes by 2–8×.
- **`Where` shape undercount** — when the broadcast condition has more elements than `X`/`Y`, onnx_tool reports the smaller `X`/`Y` shape (e.g. `(1, 1, 30, 1)`) instead of the broadcast result `(1, 10, 30, 1)` — 10× undercount.
- **`PRelu` / `Pow` / `LayerNormalization` shape undercount** — broadcast outputs reported as the smaller operand's shape, dropping a factor of 10× on memory.
- **`MaxPool` with `ceil_mode=1`** — reported as scalar `()` instead of the actual pooled shape, losing the entire output from accounting.
- **~50 ops** (`Acos`, `ArgMin`, `BitShift`, `IsNaN`, `LpPool`, `Mish`, `RandomNormal`, `ReduceL1`, …) raise `NotImplementedError: ... has no value_infer`, so any model that uses them fails scoring even though it runs cleanly under ORT - this is just a little unfortunate as it gives a seemingly arbitrary subset of ONNX that is valid for the competition.

Full report with minimal repros, affected opsets, and which fields are required vs incidental: [https://github.com/bizy-coder/onnx-tool-audit/blob/master/findings/structured_issue_report.md]. Can look through the rest of the codebase as well but fair warning it is scrappy and in parts vibe coded.

**Question for the organizers:** Is there a reason we use onnx tool instead of directly using ORT? We can promote all intermediate tensors to outputs to see the size/shape of all vertices. This seems like it would be more reliable although admittedly more costly?

## Comments (3)

- **robga** (2026-04-29T04:53:33.227Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Ah I thought that one (Round) might go unnoticed a little longer... but since you posted the report...
  
  @kevinyuluo @mmoffitt - RoundNode subclasses LessNode, but only overrides value_infer. It does not override shape_infer or profile, so it inherits LessNode.shape_infer, which sets the output dtype to numpy.bool_. Moe generlly, some nodes infer output dtype from a fixed input slot. like WhereNode slot 1, PWNode slot 0. You can effectively steer dtype through layers due to the behaviour. ie onnx-tool trusts the slot position & doesnt completely enforces the real ONNX type constraints. I haven't tried to psot-parse with the new rules, but it bumped my score up 300 points pre-reset. Naiviely,  input(float) --> Cast(bool) --> Transpose --> output vs real  input(float) --> Round --> Transpose --> output. The tool behaves as if Round inserted a Cast(bool). Dtype-preserving ops such as Transpose then propagate that incorrect bool dtype through the graph and you can "smuggle" data.

- **bizy-coder** (2026-04-29T02:09:03.933Z, votes: {'canUpvote': True}):
  Andddddddd I didn't see that they posted a new judge. However, even if MACs are calculated via onnx_tool bugs in it will be problematic as operators calculate MACs based on perceived input size
  
  Running the same diff based algorithm, there are also a few bugs in onnx.shape_inference it seems, though they appear to be smaller. Some ops are not properly supported which is unfortunate. I will look into making a more human meaningful report tomorrow: https://github.com/bizy-coder/onnx-tool-audit/blob/master/findings/scorer_structured_report.md

- **(unknown)** (2026-04-29T05:20:56.880Z, votes: {}):
  (deleted)
