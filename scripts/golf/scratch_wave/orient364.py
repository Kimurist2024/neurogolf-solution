import sys, numpy as np
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/inputs/arc-gen-repo/tasks")
import importlib.util
spec=importlib.util.spec_from_file_location('gen',R+'/inputs/arc-gen-repo/tasks/task_e509e548.py')
gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
from scipy.ndimage import label
def corners_with_orient(s):
    res=[]
    for (r,c) in s:
        nb=tuple(sorted((dr,dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if (r+dr,c+dc) in s))
        if len(nb)==2 and (nb[0][0]+nb[1][0],nb[0][1]+nb[1][1])!=(0,0): res.append(nb)
    return res
def shape_of(s):
    hasT=any(sum(((r+dr,c+dc) in s) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])==3 for (r,c) in s)
    if hasT: return 'aitch'
    return 'el' if len(corners_with_orient(s))==1 else 'you'
you_same=0; you_diff=0; you2=0; youother=0
for it in range(1500):
    ex=gen.generate(); g=np.array(ex['input']); G=(g==3).astype(np.uint8)
    lab,n=label(G,[[0,1,0],[1,1,1],[0,1,0]])
    for k in range(1,n+1):
        ys,xs=np.where(lab==k); s=set(zip(ys.tolist(),xs.tolist()))
        if shape_of(s)=='you':
            co=corners_with_orient(s)
            if len(co)==2:
                you2+=1
                if co[0]==co[1]: you_same+=1
                else: you_diff+=1
            else: youother+=1
print('you objs with exactly2 corners:',you2,'other:',youother)
print('  same-orient:',you_same,' diff-orient:',you_diff)
