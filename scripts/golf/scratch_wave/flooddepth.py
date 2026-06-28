import sys, numpy as np
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
sys.path.insert(0, R+"/inputs/arc-gen-repo/tasks")
import importlib.util
spec=importlib.util.spec_from_file_location('gen',R+'/inputs/arc-gen-repo/tasks/task_e509e548.py')
gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
from scipy.ndimage import label

def nbr4(G):
    p=np.zeros(G.shape,np.int32)
    p[1:,:]+=G[:-1,:]; p[:-1,:]+=G[1:,:]; p[:,1:]+=G[:,:-1]; p[:,:-1]+=G[:,1:]
    return p
def plus_dilate(v,G):
    m=v.copy()
    m[1:,:]=np.maximum(m[1:,:],v[:-1,:]); m[:-1,:]=np.maximum(m[:-1,:],v[1:,:])
    m[:,1:]=np.maximum(m[:,1:],v[:,:-1]); m[:,:-1]=np.maximum(m[:,:-1],v[:,1:])
    return (m*G)
def corner_mask(G):
    # green pixel with exactly 2 perpendicular nbrs
    up=np.zeros_like(G);dn=np.zeros_like(G);lf=np.zeros_like(G);rt=np.zeros_like(G)
    up[1:,:]=G[:-1,:]; dn[:-1,:]=G[1:,:]; lf[:,1:]=G[:,:-1]; rt[:,:-1]=G[:,1:]
    vert=up+dn; horz=lf+rt
    # corner = green & (one vertical nbr exactly) & (one horizontal nbr exactly)? 
    # perpendicular 2-nbr corner: exactly one of {up,dn} and exactly one of {lf,rt}, total 2
    c=(G==1)&((up+dn)==1)&((lf+rt)==1)&((up+dn+lf+rt)==2)
    return c.astype(np.int32)

# seeds: aitch = T (nbr==3); you = corners (will be 2 per you, 1 per el)
# measure depth to flood aitch from T-seed; flood corners within object
worst_a=0; worst_c=0
for it in range(1500):
    ex=gen.generate(); g=np.array(ex['input']); G=(g==3).astype(np.int32)
    lab,n=label(G,[[0,1,0],[1,1,1],[0,1,0]])
    nb=nbr4(G); Tseed=((G==1)&(nb==3)).astype(np.int32)
    cseed=corner_mask(G)
    for src,wname in [(Tseed,'a'),(cseed,'c')]:
        v=src.copy()
        for step in range(1,30):
            nv=plus_dilate(v,G)
            if (nv==v).all(): break
            v=nv
        if wname=='a': worst_a=max(worst_a,step)
        else: worst_c=max(worst_c,step)
print('worst flood depth: T-seed(aitch)=',worst_a,' corner-seed=',worst_c)
