#!/usr/bin/env python3
"""Safe best-of merge: adopt sub24 net only if strictly cheaper AND gold-correct AND margin-stable."""
import sys, tempfile, os, glob, json
sys.path.insert(0, 'scripts')
import onnx
from lib import scoring

STAGE = 'artifacts/wave_opus/stage'
SUB24 = 'inputs/submission24'
OUT = 'artifacts/merge_sub24/stage'
os.makedirs(OUT, exist_ok=True)

adoptions = []
rejects = []
errors = []
kept = 0

for t in range(1, 401):
    name = f'task{t:03d}.onnx'
    sp = f'{STAGE}/{name}'
    cp = f'{SUB24}/{name}'
    # default: keep stage version
    chosen = sp if os.path.exists(sp) else cp
    # score stage incumbent
    scost = None
    if os.path.exists(sp):
        try:
            m = onnx.load(sp)
            with tempfile.TemporaryDirectory() as wd:
                r = scoring.score_and_verify(m, t, wd, label='s', require_correct=True)
            if r: scost = r['cost']
        except Exception:
            scost = None
    # try challenger
    if os.path.exists(cp):
        try:
            mc = onnx.load(cp)
            with tempfile.TemporaryDirectory() as wd:
                rc = scoring.score_and_verify(mc, t, wd, label='c', require_correct=True)
            if rc is not None:
                stable, mn = scoring.model_margin_stable(mc, t)
                ccost = rc['cost']
                if stable and (scost is None or ccost < scost):
                    chosen = cp
                    adoptions.append((t, scost, ccost))
                elif not stable and (scost is None or ccost < scost):
                    rejects.append((t, ccost, scost, 'margin-unstable'))
            else:
                rejects.append((t, None, scost, 'gold-fail'))
        except Exception as e:
            errors.append((t, str(e)[:50]))
    # copy chosen to OUT
    if chosen and os.path.exists(chosen):
        import shutil
        shutil.copy(chosen, f'{OUT}/{name}')
        if chosen == sp: kept += 1

print(f'=== sub24 safe merge ===')
print(f'adoptions: {len(adoptions)}  kept-stage: {kept}  rejects: {len(rejects)}  errors: {len(errors)}')
for t, sc, cc in sorted(adoptions, key=lambda x: (x[1] or 0)-(x[2] or 0))[:30]:
    print(f'  ADOPT task{t:03d}: stage={sc} -> sub24={cc}')
if rejects:
    print('rejects (sub24 cheaper but unsafe):')
    for t, cc, sc, why in rejects[:20]:
        print(f'  REJECT task{t:03d}: sub24={cc} stage={sc} ({why})')
if errors:
    print('errors:', errors[:10])
json.dump({'adoptions': adoptions, 'rejects': rejects, 'errors': errors}, open('artifacts/merge_sub24/report.json','w'), indent=2)
