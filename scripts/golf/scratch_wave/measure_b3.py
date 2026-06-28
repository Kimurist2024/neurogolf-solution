import sys, tempfile, onnx, os
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/scripts")
from lib import scoring
import inspect
print([n for n in dir(scoring) if not n.startswith('_')])
m=onnx.load(R+"/scripts/golf/scratch_wave/task364_b.onnx")
td=tempfile.mkdtemp()
r=scoring.score_and_verify(m,364,td,require_correct=False)  # don't require, just get mem/params
print("score_and_verify(require_correct=False):", r)
