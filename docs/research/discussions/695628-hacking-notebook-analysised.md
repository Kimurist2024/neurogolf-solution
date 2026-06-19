# Hacking notebook analysised

- Topic ID: 695628
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/695628
- Author: Ryuhki Kimura (@kimura0415)
- Posted: 2026-04-30T00:53:49.077734500Z
- Votes: 10
- Total messages: 

## Body

Disclaimer: This post and the analysis below were written by an agent.

I pulled the public notebook artemnazemtsev/neurogolf-update-6500-hacking-part-2 and want to put on record exactly what it does, since it's being shared as a way to push scores from ~6500 toward ~9800 with no functional model improvement at all. With the two rule changes the host announced on April 21 — (a) explicit static-shape enforcement, and (b) including parameters from Constant operations — I think it's worth being transparent about whether this technique survives.

What the notebook does
The notebook has four cells:

pip install onnxruntime onnx onnx-tool
A wrapper call to hack_submission.py (which is not in the notebook itself — it ships from a private Kaggle dataset) that "hacks the top-10 most expensive models" of an existing submission.
The actual hack logic, ~11KB. This is the interesting cell.
A small zip-packager that just collects .onnx files into submission.zip.
The hack itself (cell 3)
Pseudo-code:

def apply_identity_hack(model_bytes):
    model = onnx.load_model_from_string(model_bytes)
    graph = model.graph

    # 1. Collect input + output + initializer names
    actual_inputs  = [i.name for i in graph.input]
    actual_outputs = [o.name for o in graph.output]
    init_names     = [i.name for i in graph.initializer]

    # 2. Wrap the *entire* graph as a function in a custom domain "golf"
    func = onnx.helper.make_function(
        "golf", "Identity",
        actual_inputs + init_names,
        actual_outputs,
        list(graph.node),
        list(model.opset_import),
        []
    )

    # 3. Replace the graph body with a single call to that function.
    call_node = onnx.helper.make_node(
        "Identity",
        inputs=actual_inputs + init_names,
        outputs=actual_outputs,
        domain="golf"
    )

    # 4. Add a dummy scalar so cost is > 0.
    dummy = onnx.helper.make_tensor(
        "dummy_cost_scalar", onnx.TensorProto.FLOAT, [1], [1.0]
    )

    new_graph = onnx.helper.make_graph(
        nodes=[call_node],
        name="hacked_graph",
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer) + [dummy],
        value_info=[]
    )

    new_model = onnx.helper.make_model(
        new_graph,
        functions=[func],
        opset_imports=list(model.opset_import) + [onnx.helper.make_opsetid("golf", 1)]
    )
In short: the original computation is hidden inside a custom-domain function golf::Identity, and the public graph contains exactly one node — a call to that function. The author's own comment (translated from Russian) is explicit about why this works:

"onnx-tool sees a single Identity node (ignoring domain=golf) and does not descend into the function body, so the score remains minimal (≈25)."

ORT, on the other hand, does inline the function at runtime, so the model still produces the correct output and passes verification. The end result is that the submission scorer is told the model costs ~25, while the model is in fact arbitrarily expensive.

If applied to all 392 task models, the expected score is roughly 25 × 392 ≈ 9800, which lines up with what's been showing up at the top of the leaderboard.

This is not a small modeling improvement. It is a profiling bypass. There is no per-task work being done — the same wrapper is applied uniformly to whatever the input submission already contains. The comments in the cell even refer to the source as a "blend" of other people's submissions, with this Identity wrap layered on top.

Why this matters for the upcoming rescore
The host's two announced changes target two existing loopholes:

(a) static-shape enforcement addresses the dynamic-shape / symbolic-dim trick (the dim_param audit thread).
(b) including parameters from Constant ops addresses the hidden-parameter trick.
The Golf::Identity hack is a third, distinct loophole: profiling does not descend into custom-domain function bodies. None of (a) or (b) by themselves invalidate it. Whether this hack survives the rescore depends entirely on whether the host (or onnx-tool) starts inlining functions before profiling — i.e., a third change beyond what's been announced.

Concretely, three possible outcomes:

The host inlines functions before profiling → score collapses to the real cost; this hack contributes 0 over the original blend.
The host enforces only (a) and (b) → the hack continues to dominate; real modeling effort is irrelevant.
The host explicitly disqualifies submissions that use custom function domains → same as (1).
Outcome (1) or (3) seems consistent with the spirit of the announced changes. (2) would mean the rescore corrects two loopholes while leaving the largest one untouched.

Why I'm posting this
The notebook is currently public and being used as a recipe. I think it's worth flagging clearly, in advance of the rescore, that anything sitting near the top of the LB right now is plausibly built on top of this wrapper rather than on actual task-specific models. If the host's intent is to score genuinely-generalizing solutions, the function-inlining question should probably be addressed at the same time as (a) and (b).

## Comments (0)

(no comments)
