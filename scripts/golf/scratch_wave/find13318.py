import sys, tempfile, onnx, glob
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/scripts")
from lib import scoring
import os
cands=set()
for pat in ["**/task364.onnx"]:
    cands|=set(glob.glob(R+"/"+pat, recursive=True))
# also search any onnx dir with 364
for p in sorted(cands):
    try:
        m=onnx.load(p)
        r=scoring.score_and_verify(m,364,tempfile.mkdtemp(),require_correct=True)
        if r: print(r['cost'], r['correct'], p)
    except Exception as e:
        pass
