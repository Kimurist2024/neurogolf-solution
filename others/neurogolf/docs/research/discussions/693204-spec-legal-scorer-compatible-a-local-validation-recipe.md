# Spec-legal ≠ scorer-compatible: a local validation recipe

- Topic ID: 693204
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693204
- Author: Kameron Kilchrist (@kameronkilchrist)
- Posted: 2026-04-20T07:17:36.030311300Z
- Votes: 7
- Total messages: 1

## Body

The official NeuroGolf constraints are clear: static shapes, no `Loop/Scan/NonZero/Unique/Script/Function`, ≤1.44MB per file, any opset. But a model can satisfy all of that, pass `onnxruntime` and `onnx.checker.check_model`, and still get rejected at submission time with:

> `Error processing one or more onnx networks.`

The scorer isn't onnxruntime — it's `onnx_tool`. Its `shape_infer` + `profile` passes are strictly stricter than the ONNX spec, and several spec-legal ops crash it silently.

## What `neurogolf_utils` actually runs

```python
# data/neurogolf_utils/neurogolf_utils.py (abridged)
import onnx_tool

def score_network(onnx_bytes):
    model = onnx_tool.loadmodel(onnx_bytes, {'verbose': False})
    model.graph.graph_reorder_nodes()
    model.graph.shape_infer(None)   # <-- this is where most submissions die
    model.graph.profile()
    if not model.graph.valid_profile:
        return None, None, None
    ...
```

If `shape_infer` raises or `valid_profile` is False, the task returns `None, None, None`, and Kaggle surfaces it as the batch-wide error above. No per-task diagnostics.

## Ops we've confirmed are poisoned

Every one of these passes `onnxruntime` and `onnx.checker`. Every one of them kills the scorer.

| Op | Symptom | Workaround |
|---|---|---|
| `Min` | `ValueError: truth value of array is ambiguous` — `MinNode.value_infer` has a literal `result = not numpy.minimum(...)` typo | `Greater + Cast` for `min(x,1)`; `Neg + Max + Neg` for general min |
| `ArgMin` | `NotImplementedError: Node ArgMin has no value_infer` | `ArgMax(Neg(x))` |
| `Clip` (opset-10 attribute form) | `shape_infer` assert; the `ClipNode` expects opset-11 input form | Upgrade file to opset 11, pass min/max as inputs |
| `ArgMax(keepdims=0)` + `Reshape→[1]` | `AssertionError: raw == volume(newshape)` at node.py:2342 (volume-of-scalar bug) | Use `keepdims=1`, or `Unsqueeze(axes=[0])` instead of `Reshape` |
| Scalar-index `Gather` on rank-4 | Phantom size-1 dim leaks through, downstream Reshape or Conv crashes | Rank-1 `[1]` index + explicit `Reshape` to strip the dim |

There's also one non-op trap worth knowing: any task whose benchmark grids exceed 30×30 (e.g. `arc-gen` examples for tasks 021, 055, 080, 184, 202, 366) will `IndexError` inside `convert_to_numpy` *before the model runs*. Local `verify_task` silently skips these; scorer does not. They're permanently unsubmittable.

## Local validation that actually catches this

Three layers, each catching a different class of failure. The first two are not enough on their own.

```python
# Layer 1: shape correctness (necessary, not sufficient)
import onnx
onnx.checker.check_model(path)

# Layer 2: runtime execution (necessary, not sufficient)
import onnxruntime as ort
sess = ort.InferenceSession(path)
out = sess.run(None, {"input": sample_input})

# Layer 3: the one that matters — run the actual scorer
import onnx_tool
model = onnx_tool.loadmodel(path, {'verbose': False})
model.graph.graph_reorder_nodes()
model.graph.shape_infer(None)     # poisoned ops raise here
model.graph.profile()
assert model.graph.valid_profile, "would fail Kaggle scorer"
```

Wrap layer 3 in a subprocess — a bad `shape_infer` can leave `onnx_tool` in a state that corrupts later loads in the same process:

```python
import subprocess, sys
from pathlib import Path

def scorer_safe(onnx_path: Path) -> tuple[bool, str]:
    code = f"""
import onnx_tool
m = onnx_tool.loadmodel({str(onnx_path)!r}, {{'verbose': False}})
m.graph.graph_reorder_nodes()
m.graph.shape_infer(None)
m.graph.profile()
assert m.graph.valid_profile
print("OK")
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0, (r.stderr or r.stdout).strip()

ok, msg = scorer_safe(Path("submission/latest/task042.onnx"))
if not ok:
    print(f"would fail on Kaggle: {msg}")
```

## Grep patterns that flag trouble before building

```bash
# ArgMin — always broken
grep -rn '"ArgMin"' src/

# keepdims=0 followed by Reshape — volume-of-scalar trap
grep -rnB1 -A3 'keepdims=0' src/ | grep -B2 Reshape

# Min node — the typo'd value_infer
grep -rn 'make_node("Min"' src/

# Attribute-form Clip on opset 10
grep -rn 'helper.make_node("Clip"' src/
```

## Why we run layer 3 even when layers 1–2 pass

Because `onnx_tool`'s `shape_infer` is its own shape-inference implementation — not ONNX's, not ORT's. It has per-node `value_infer` methods that hand-implement each op, and several of them have bugs that the ONNX spec doesn't care about. The official constraint doc tells you what the *format* accepts. The scorer tells you what the *implementation* accepts. Those sets don't match.

If a submission returns "Error processing one or more onnx networks" and every local check passes, run layer 3 on each file in the bundle. It will point at the real culprit in one line.

## Comments (1)

- **Russell Kirk** (2026-04-20T07:48:10.380Z, votes: {'canUpvote': True}):
  Thanks! It is helpful, but I didn't identify the mistakes in my instance -- so there's perhaps more. I don't know enough about it to have more than a suspicion it's related to using WHERE with UINT8
