import sys, json, numpy as np, onnxruntime as ort, onnx
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(R+"/scripts/golf/scratch_wave/task364_b.onnx", so, providers=["CPUExecutionProvider"])
d=json.load(open(R+"/inputs/neurogolf-2026/task364.json"))
def onehot(grid):
    g=np.array(grid); H,W=g.shape; x=np.zeros((1,10,30,30),np.float32)
    for r in range(H):
        for c in range(W): x[0,g[r,c],r,c]=1
    return x,H,W
bad=0
for sp in ['train','test','arc-gen']:
    for pr in d[sp][:50]:
        x,H,W=onehot(pr['input'])
        y=sess.run(None,{"input":x})[0][0]  # [10,30,30]
        pred=y[:, :H,:W].argmax(0)
        # but border outside grid must be all-zero -> argmax 0 ok
        out=np.array(pr['output'])
        if not np.array_equal(pred, out):
            bad+=1
            if bad<=2:
                print(sp,'MISMATCH'); 
                diff=np.argwhere(pred!=out)[:5]
                for (r,c) in diff: print('  ',(r,c),'pred',pred[r,c],'true',out[r,c])
print("bad(of sampled):",bad)
