import sys, tempfile, onnx, json, numpy as np
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/scripts")
from lib import scoring
m=onnx.load(R+"/scripts/golf/scratch_wave/task364_cand.onnx")
r=scoring.score_and_verify(m, 364, tempfile.mkdtemp(), require_correct=True)
print("score_and_verify:", r)
