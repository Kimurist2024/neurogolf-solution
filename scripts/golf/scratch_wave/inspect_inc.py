import sys, tempfile, onnx, collections
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/scripts")
from lib import scoring
m=onnx.load("/tmp/sub364/task364.onnx")
r=scoring.score_and_verify(m,364,tempfile.mkdtemp(),require_correct=True)
print("incumbent score_and_verify:", r)
c=collections.Counter(n.op_type for n in m.graph.node)
print("ops:", dict(c), "n_nodes", len(m.graph.node))
# initializer dtypes/sizes
for init in m.graph.initializer:
    import numpy as np
    print("init", init.name, init.dims, init.data_type)
