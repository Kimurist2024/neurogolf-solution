"""Fresh-exact reference solver for task366 (e6721834): split-grid box completion.

Rule: input is two equal halves (vertical split if H even, horizontal if W even;
ambiguous when both even -> pick orientation with the most complete matching).
Each half has its own background. The "fore" half has solid forecolor rectangles
("boxes") with interior colored "dots"; the other ("dot") half shows only those
dots (same relative pattern, different absolute position) on its background. Output
= dot half with each box's forecolor rectangle drawn back in, anchored by matching
the box's dot signature. Boxes may run off the visible grid edge.
"""
import json
import numpy as np
from collections import Counter, deque


def exterior_bg(half):
    """Background = color of the largest border-connected region (robust when a box
    fills most of the half so its color outnumbers the true background)."""
    H, W = half.shape
    border = [(0,c) for c in range(W)] + [(H-1,c) for c in range(W)] + \
             [(r,0) for r in range(H)] + [(r,W-1) for r in range(H)]
    best = None
    for col in set(half[r,c] for r,c in border):
        seen = np.zeros((H,W), bool); dq = deque()
        for r,c in border:
            if half[r,c]==col and not seen[r,c]:
                seen[r,c]=True; dq.append((r,c))
        while dq:
            y,x=dq.popleft()
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny,nx=y+dy,x+dx
                if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and half[ny,nx]==col:
                    seen[ny,nx]=True; dq.append((ny,nx))
        sz=int(seen.sum())
        if best is None or sz>best[0]: best=(sz,int(col))
    return best[1]


def cc4(mask):
    H,W=mask.shape; seen=np.zeros_like(mask,bool); comps=[]
    for r in range(H):
        for c in range(W):
            if mask[r,c] and not seen[r,c]:
                st=[(r,c)]; seen[r,c]=True; comp=[]
                while st:
                    y,x=st.pop(); comp.append((y,x))
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not seen[ny,nx]:
                            seen[ny,nx]=True; st.append((ny,nx))
                comps.append(comp)
    return comps


def analyze_fore(fore, bgf, forecolor):
    """Boxes = connected comps of non-bg cells; box = bbox; dots = non-forecolor non-bg inside."""
    boxes=[]
    for comp in cc4(fore!=bgf):
        rs=[p[0] for p in comp]; cs=[p[1] for p in comp]
        r0,r1,c0,c1=min(rs),max(rs),min(cs),max(cs)
        tall,wide=r1-r0+1,c1-c0+1
        # require rectangular (full box visible); skip ragged comps (off-grid clipped boxes
        # are handled by the dot-side anchor, which still carries the relative dot pattern)
        dots={}
        for dr in range(tall):
            for dc in range(wide):
                v=int(fore[r0+dr,c0+dc])
                if v!=bgf and v!=forecolor: dots[(dr,dc)]=v
        area=tall*wide
        filled=sum(1 for dr in range(tall) for dc in range(wide) if fore[r0+dr,c0+dc]!=bgf)
        boxes.append({'tall':tall,'wide':wide,'dots':dots,'rect':filled==area})
    return boxes


def attempt(fore,bgf,dot,bgd):
    cnt=Counter([x for x in fore.flatten().tolist() if x!=bgf])
    if not cnt: return None, 1<<30
    forecolor=cnt.most_common(1)[0][0]
    boxes=analyze_fore(fore,bgf,forecolor)
    oh,ow=dot.shape
    out=np.full(dot.shape,bgd,dtype=dot.dtype)
    dot_cells={(r,c):int(dot[r,c]) for r in range(oh) for c in range(ow) if dot[r,c]!=bgd}
    used=set(); unplaced=0
    for box in sorted(boxes,key=lambda b:-len(b['dots'])):
        dots=box['dots']; tall,wide=box['tall'],box['wide']
        if not dots: unplaced+=1; continue
        placed=False
        # allow the box to run off any edge: br,bc range over [-tall+1, oh-1] etc, clip when stamping
        for br in range(-tall+1, oh):
            for bc in range(-wide+1, ow):
                okm=True
                for (dr,dc),col in dots.items():
                    key=(br+dr,bc+dc)
                    if not (0<=key[0]<oh and 0<=key[1]<ow): okm=False; break
                    if dot_cells.get(key)!=col or key in used: okm=False; break
                if not okm: continue
                region=[(br+dr,bc+dc) for dr in range(tall) for dc in range(wide)
                        if 0<=br+dr<oh and 0<=bc+dc<ow and (br+dr,bc+dc) in dot_cells]
                if len(region)!=len(dots): continue
                for dr in range(tall):
                    for dc in range(wide):
                        if 0<=br+dr<oh and 0<=bc+dc<ow: out[br+dr,bc+dc]=forecolor
                for (dr,dc),col in dots.items():
                    out[br+dr,bc+dc]=col; used.add((br+dr,bc+dc))
                placed=True; break
            if placed: break
        if not placed: unplaced+=1
    leftover=len(dot_cells)-len(used)
    return out, unplaced*100+leftover


def split_attempt(X,Y):
    bgX=exterior_bg(X); bgY=exterior_bg(Y)
    if bgX==bgY: return None,1<<30
    if (X!=bgX).sum()>=(Y!=bgY).sum():
        fore,bgf,dot,bgd=X,bgX,Y,bgY
    else:
        fore,bgf,dot,bgd=Y,bgY,X,bgX
    return attempt(fore,bgf,dot,bgd)


def solve(grid):
    i=np.array(grid); H,W=i.shape; results=[]
    if W%2==0:
        out,b=split_attempt(i[:,:W//2], i[:,W//2:])
        if out is not None: results.append((b,out))
    if H%2==0:
        out,b=split_attempt(i[:H//2,:], i[H//2:,:])
        if out is not None: results.append((b,out))
    if not results: return None
    results.sort(key=lambda t:t[0])
    return results[0][1]


if __name__=="__main__":
    d=json.load(open('/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf/inputs/neurogolf-2026/task366.json'))
    total=0; ok=0; fails=[]
    for split in ['train','test','arc-gen']:
        for k,p in enumerate(d[split]):
            i=np.array(p['input'])
            if max(i.shape)>30: continue
            total+=1
            pred=solve(p['input']); gold=np.array(p['output'])
            if pred is not None and pred.shape==gold.shape and np.array_equal(pred,gold): ok+=1
            else: fails.append((split,k))
    print(f"fixtures: {ok}/{total} exact; fails {fails[:12]}")
