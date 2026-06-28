import sys, numpy as np, itertools
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/inputs/arc-gen-repo/tasks")
import importlib.util
spec=importlib.util.spec_from_file_location('gen',R+'/inputs/arc-gen-repo/tasks/task_e509e548.py')
gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
from scipy.ndimage import label
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
def corner_orients(s):
    res=[]
    for (r,c) in s:
        nb=tuple(sorted(d for d in DIRS if (r+d[0],c+d[1]) in s))
        if len(nb)==2 and (nb[0][0]+nb[1][0],nb[0][1]+nb[1][1])!=(0,0):
            res.append(nb)  # canonical pair = orientation
    return res
def shape_of(s):
    hasT=any(sum(((r+dr,c+dc) in s) for dr,dc in DIRS)==3 for (r,c) in s)
    if hasT: return 'aitch'
    return 'el' if len(corner_orients(s))==1 else 'you'

# enumerate the 4 corner orientation keys
allk=set()
el_sets=[]; you_sets=[]
for it in range(2000):
    ex=gen.generate(); g=np.array(ex['input']); G=(g==3).astype(np.uint8)
    lab,n=label(G,[[0,1,0],[1,1,1],[0,1,0]])
    for k in range(1,n+1):
        ys,xs=np.where(lab==k); s=set(zip(ys.tolist(),xs.tolist()))
        sh=shape_of(s); co=set(corner_orients(s)); allk|=co
        if sh=='el': el_sets.append(co)
        elif sh=='you': you_sets.append(co)
allk=sorted(allk)
print('corner orientation keys:',allk)
# find 2-group partition: groupA subset. el must hit only one group; you must hit both.
from itertools import combinations
found=None
for r in range(1,len(allk)):
    for A in combinations(allk,r):
        A=set(A); B=set(allk)-A
        ok=True
        for co in el_sets:
            inA=bool(co&A); inB=bool(co&B)
            if inA and inB: ok=False;break  # el must NOT hit both
        if not ok: continue
        for co in you_sets:
            if not (bool(co&A) and bool(co&B)): ok=False;break # you must hit both
        if ok: found=(A,B);break
    if found: break
print('partition found:',found)
