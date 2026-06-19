#!/usr/bin/env python3
"""Pick the next FOCUS target: the task closest to (but still below) the 15.1
score threshold, i.e. the cheapest task whose cost is still > 19930 (score<15.1).
Closest-to-threshold first = smallest cut needed = highest crossing probability
= most efficient way to convert tasks to >=15.1.

Effective cost = min(campaign_costs, campaign_banked) so a task already pushed
below 19930 (banked but not yet submitted) is not re-picked.

Pass rotation: skips tasks in focus_attempted.json (tried this pass). When all
remaining >19930 tasks are attempted this pass, prints RESET (shell clears the
pass and revisits = "try forever"). Prints "" only when NO task is >19930.

Output: task:hash:cost:template_task   (template='-' if no cheap cluster sibling)
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FS = REPO / "docs" / "golf"
TASKS = REPO / "inputs" / "arc-gen-repo" / "tasks"
THRESH = 19930  # cost for score 15.1 (25 - ln(19930) = 15.10)
JACCARD = 0.6
UBI = {"grids","grid","randint","randints","random_color","random_colors","choice",
       "choices","sample","shuffle","random_el","deepcopy","flatten","black","blue",
       "red","green","yellow","gray","pink","orange","cyan","maroon","set_colors",
       "remove_duplicates","isclose","sqrt","int_sqrt"}
CALL = re.compile(r"common\.([a-z_]+)\s*\(")


def hset(h):
    p = TASKS / f"task_{h}.py"
    return frozenset({m.group(1) for m in CALL.finditer(p.read_text(errors='ignore'))} - UBI) if p.is_file() else frozenset()


def load(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d


def main():
    costs = {int(k): v for k, v in load(FS / "campaign_costs.json", {}).items()}
    banked = {int(k): v for k, v in load(FS / "campaign_banked.json", {}).items()}
    hashes = load(FS / "task_hash_map.json", {})
    done = set(load(FS / "focus_attempted.json", []))
    eff = {t: min(c, banked.get(t, c)) for t, c in costs.items()}
    over = [(c, t) for t, c in eff.items() if c > THRESH]
    if not over:
        print("")  # everything already >=15.1
        return
    avail = [(c, t) for c, t in over if t not in done]
    if not avail:
        print("RESET")  # all attempted this pass; shell resets & revisits
        return
    # Efficient order: achievable (<=1.9x cut) first, by VALUE descending (bigger
    # crossing = more LB points); then the hard >1.9x ones, least-impossible first.
    CAP = THRESH * 1.9
    ach = sorted([(c, t) for c, t in avail if c <= CAP], reverse=True)
    hard = sorted([(c, t) for c, t in avail if c > CAP])
    c, t = (ach + hard)[0]
    h = hashes.get(f"{t:03d}") or hashes.get(str(t))
    # template = cheapest cluster sibling (cost 250-8000) sharing the archetype
    sig = {tt: hset(hashes.get(f"{tt:03d}")) for tt in costs if hashes.get(f"{tt:03d}")}
    sig = {tt: s for tt, s in sig.items() if s}
    tmpl = "-"
    if t in sig and sig[t]:
        best_c = 1e18
        for tt, s in sig.items():
            if tt == t or not (250 <= eff.get(tt, 1e18) <= 8000):
                continue
            j = len(sig[t] & s) / len(sig[t] | s) if (sig[t] | s) else 0
            if j >= JACCARD and eff[tt] < best_c:
                best_c, tmpl = eff[tt], str(tt)
    print(f"{t}:{h}:{int(c)}:{tmpl}")


if __name__ == "__main__":
    main()
