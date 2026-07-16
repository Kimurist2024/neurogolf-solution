#!/usr/bin/env python3
"""Emit the next N mode-routed Kimi campaign targets, prioritized by expected yield.

Modes (by proven effectiveness this session):
  port   -- task shares an archetype cluster with a much-cheaper CORRECT sibling
            (the template). Kimi ports the sibling's cheap technique. Proven:
            task119 12142->6943 (+0.56). Highest priority.
  shrink -- over-built mid-band net (cost 5k-25k) with no cheap sibling. Kimi
            shrinks the existing graph (dtype/crop/fuse). Proven: task125/346.
SKIP: cost < FLOOR (near-optimal) or cost > MONSTER with no cheap template
(pure geometric decoder -> Kimi yields 0; don't waste workers).

Excludes docs/golf/campaign_attempted.json and currently-running tasks (worker
log written within KIMI_RUNNING_SEC). Costs from docs/golf/campaign_costs.json.

Prints lines: task:hash:cost:mode:template_task   (template_task='-' for shrink)
Usage: campaign_targets.py [N=8]
"""
from __future__ import annotations
import json, math, re, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "inputs" / "arc-gen-repo" / "tasks"
FS = REPO / "docs" / "golf"
LOGS = REPO / "artifacts" / "kimi_logs"

FLOOR = 5000        # below this: near-optimal, skip
MONSTER = 25000     # above this with no cheap template: skip (Kimi 0 yield)
PORT_RATIO = 2.2    # template must be this much cheaper than target to be worth porting
JACCARD = 0.6

UBI = {"grids","grid","randint","randints","random_color","random_colors","choice",
       "choices","sample","shuffle","random_el","deepcopy","flatten","black","blue",
       "red","green","yellow","gray","pink","orange","cyan","maroon","set_colors",
       "remove_duplicates","isclose","sqrt","int_sqrt"}
CALL = re.compile(r"common\.([a-z_]+)\s*\(")


def helper_set(h):
    p = TASKS / f"task_{h}.py"
    if not p.is_file():
        return frozenset()
    return frozenset({m.group(1) for m in CALL.finditer(p.read_text(errors="ignore"))} - UBI)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run_sec = float(os.environ.get("KIMI_RUNNING_SEC", "300"))
    costs = {int(k): v for k, v in json.load(open(FS / "campaign_costs.json")).items()}
    hashes = json.load(open(FS / "task_hash_map.json"))
    try:
        attempted = set(json.load(open(FS / "campaign_attempted.json")))
    except Exception:
        attempted = set()

    # currently-running (recent worker log)
    running = set()
    if LOGS.is_dir():
        now = max((p.stat().st_mtime for p in LOGS.glob("task*.log")), default=0.0)
        for p in LOGS.glob("task*.log"):
            try:
                tt = int(p.stem.replace("task", ""))
            except ValueError:
                continue
            if now - p.stat().st_mtime <= run_sec:
                running.add(tt)
    blocked = attempted | running

    sig = {t: helper_set(hashes.get(f"{t:03d}")) for t in costs if hashes.get(f"{t:03d}")}
    sig = {t: s for t, s in sig.items() if s}

    # union-find clusters
    parent = {t: t for t in sig}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    ts = list(sig)
    for i, a in enumerate(ts):
        for b in ts[i+1:]:
            u, v = sig[a], sig[b]
            if u and v and len(u & v) / len(u | v) >= JACCARD:
                parent[find(a)] = find(b)
    clusters = {}
    for t in ts:
        clusters.setdefault(find(t), []).append(t)

    # cheapest correct member per cluster = template
    template = {}  # cluster_root -> (task, cost)
    for root, members in clusters.items():
        cheapest = min(members, key=lambda t: costs.get(t, 1e18))
        template[root] = (cheapest, costs.get(cheapest, 1e18))

    def score(c):
        return max(1.0, 25 - math.log(c))

    ports, shrinks = [], []
    for t, c in costs.items():
        if t in blocked or c < FLOOR:
            continue
        root = find(t) if t in parent else None
        tmpl_t, tmpl_c = template.get(root, (None, 1e18)) if root is not None else (None, 1e18)
        h = hashes.get(f"{t:03d}")
        if not h:
            continue
        # Skip monsters entirely (cost > MONSTER): proven 0-yield for Kimi this
        # session (24 tasks, 0 promotions). Both modes target the achievable band.
        if c > MONSTER:
            continue
        good_template = (
            tmpl_t is not None and tmpl_t != t
            and 250 <= tmpl_c <= 8000          # real implementation, not degenerate (cost 0-10) nor another monster
            and tmpl_c * PORT_RATIO <= c       # template meaningfully cheaper
        )
        if good_template:
            gain = min(2.0, score(max(tmpl_c, 400)) - score(c))
            ports.append((gain, t, h, c, tmpl_t))
        else:
            gain = score(c / math.e) - score(c)  # one full +1 if 2.7x achievable
            shrinks.append((gain, t, h, c))

    ports.sort(key=lambda r: -r[0])
    shrinks.sort(key=lambda r: -r[0])
    out = []
    for g, t, h, c, tt in ports:
        out.append(f"{t}:{h}:{int(c)}:port:{tt}")
        if len(out) >= n:
            break
    for g, t, h, c in shrinks:
        if len(out) >= n:
            break
        out.append(f"{t}:{h}:{int(c)}:shrink:-")
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
