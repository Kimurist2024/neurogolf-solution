import sys, math, numpy as np, onnx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import scoring

m = onnx.load(sys.argv[1])
task = int(sys.argv[2])
print("sparse_initializer:", len(m.graph.sparse_initializer),
      "regular_initializer:", len(m.graph.initializer))
sess = scoring._make_raw_session(m)
if sess is None:
    print("RAW SESSION: FAILED TO CREATE (grader would reject)")
    raise SystemExit(0)
ex = scoring.load_examples(task)
ok = True; n = 0
for sub in ("train", "test", "arc-gen"):
    for e in ex.get(sub, []):
        b = scoring.convert_to_numpy(e)
        if b is None:
            continue
        try:
            raw = scoring._raw_output(sess, b["input"])
        except Exception as exc:
            print("INFERENCE ERROR:", exc); ok = False; continue
        n += 1
        if not np.array_equal((raw > 0).astype(np.float32), b["output"]):
            ok = False
print(f"gold over {n} visible examples:", "ALL MATCH" if ok else "MISMATCH")
params = 0
for init in m.graph.initializer: params += int(np.prod(init.dims))
for si in m.graph.sparse_initializer: params += int(np.prod(si.values.dims))
for node in m.graph.node:
    if node.op_type == "Constant":
        for a in node.attribute:
            if a.name == "value": params += int(np.prod(a.t.dims))
            elif a.name == "sparse_value": params += int(np.prod(a.sparse_tensor.values.dims))
print("params (sparse-aware):", params, "score~", round(max(1, 25 - math.log(max(1, params))), 2))
