import sys, tempfile, onnx, json, numpy as np, onnxruntime as ort
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/scripts")
from lib import scoring
m=onnx.load(R+"/scripts/golf/scratch_wave/task364_b.onnx")
# full correctness on all data
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(R+"/scripts/golf/scratch_wave/task364_b.onnx", so, providers=["CPUExecutionProvider"])
d=json.load(open(R+"/inputs/neurogolf-2026/task364.json"))
def oh(grid):
    g=np.array(grid);H,W=g.shape;x=np.zeros((1,10,30,30),np.float32)
    for r in range(H):
        for c in range(W): x[0,g[r,c],r,c]=1
    return x,H,W,g
bad=0;tot=0
for sp in ['train','test','arc-gen']:
    for pr in d[sp]:
        tot+=1; x,H,W,g=oh(pr['input'])
        y=sess.run(None,{"input":x})[0][0]
        full=y.argmax(0)  # full 30x30
        # build expected full 30x30: output in-grid, zeros outside
        exp=np.zeros((30,30),int); o=np.array(pr['output']); exp[:H,:W]=o
        if not np.array_equal(full,exp): bad+=1
print("FULL data correct:",tot-bad,"/",tot)
# memory/params via lib directly
try:
    mem,par = scoring.score_network(scoring.sanitize_model(m) if hasattr(scoring,'sanitize_model') else m, tempfile.mkdtemp())
    print("mem,par via score_network:",mem,par)
except Exception as e:
    print("score_network err:",e)
# fallback: count params + crude mem
import math
par=sum(math.prod(i.dims) for i in m.graph.initializer)
print("params(initializers):",par)
