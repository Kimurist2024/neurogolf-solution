#!/usr/bin/env python3
"""Post-wave processor for the Kimi campaign: fresh-gate promotions, bank the
ADOPTs, and AUTO-SUBMIT once the accumulated verified gain crosses a threshold.

Called by kimi_campaign.sh after each wave with the wave's task numbers.

State (docs/golf/):
  campaign_best.txt      -> "<zip_path>\t<public_score>" (current LB best)
  campaign_banked.json   -> {task: cost} fresh-ADOPTed, not yet reflected in best
  campaign_costs.json    -> {task: cost} live cost cache
Staged winners: artifacts/campaign_stage/task<NNN>.onnx

Auto-submit: when sum of banked score-gains >= THRESHOLD, build best+banked,
submit via kaggle, poll, and if the public score improved, archive a new
_BEST_<score>.zip, point campaign_best.txt at it, and clear the banked set.
The Kaggle leaderboard keeps the best submission, so a non-improving submit is
harmless (best is preserved); we still only adopt fresh-gated nets.

Usage: campaign_step.py <task,task,...> [--threshold 0.5] [--k 500] [--no-submit]
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time, zipfile, hashlib, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import verify_fix  # noqa: E402  (reuse the strict gate)

FS = REPO / "docs" / "golf"
HAND = REPO / "artifacts" / "handcrafted"
STAGE = REPO / "artifacts" / "campaign_stage"
BEST_PTR = FS / "campaign_best.txt"
BANKED = FS / "campaign_banked.json"
COSTS = FS / "campaign_costs.json"
COMP = "neurogolf-2026"


def _load(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d


def score(c):
    return max(1.0, 25 - math.log(c))


def best_zip_and_score():
    txt = BEST_PTR.read_text().strip().split("\t")
    return Path(txt[0]), float(txt[1])


def fresh_gate_and_bank(tasknums, k):
    """Fresh-gate each task's handcrafted net; bank ADOPTs cheaper than best."""
    best_zip, _ = best_zip_and_score()
    bz = zipfile.ZipFile(best_zip)
    costs = {int(a): b for a, b in _load(COSTS, {}).items()}
    banked = {int(a): b for a, b in _load(BANKED, {}).items()}
    STAGE.mkdir(parents=True, exist_ok=True)
    newly = []
    for t in tasknums:
        hp = HAND / f"task{t:03d}.onnx"
        if not hp.is_file():
            continue
        best_cost = costs.get(t)
        if best_cost is None:
            continue
        v = verify_fix.verify_one(t, hp, k)
        if v["decision"] != "ADOPT":
            print(f"  task{t}: REJECT ({v.get('fresh_fails')}/{v.get('fresh_total')} fresh)", file=sys.stderr)
            continue
        nc = v["cost"]
        if nc >= best_cost:
            continue  # not cheaper than current best
        shutil.copy(hp, STAGE / f"task{t:03d}.onnx")
        banked[t] = nc
        newly.append((t, best_cost, nc, score(nc) - score(best_cost)))
        print(f"  task{t}: ADOPT {best_cost}->{nc} (+{score(nc)-score(best_cost):.3f})", file=sys.stderr)
    json.dump({str(a): b for a, b in banked.items()}, open(BANKED, "w"))
    return banked, costs, newly


def banked_gain(banked, costs):
    return sum(max(0.0, score(c) - score(costs.get(t, c))) for t, c in banked.items())


def build_and_submit(banked, costs):
    best_zip, best_score = best_zip_and_score()
    out = REPO / "artifacts" / "campaign_submit.zip"
    src = zipfile.ZipFile(best_zip)
    names = sorted(n for n in src.namelist() if n.endswith(".onnx"))
    swaps = {t: (STAGE / f"task{t:03d}.onnx").read_bytes() for t in banked}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            t = int(n[4:7])
            z.writestr(n, swaps.get(t, src.read(n)))
    proj = best_score + banked_gain(banked, costs)
    msg = f"campaign auto: +{len(banked)} fresh-verified tasks, proj ~{proj:.2f}"
    sub = REPO / "submission.zip"
    shutil.copy(out, sub)
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f",
                        str(sub), "-m", msg], capture_output=True, text=True)
    print("  submit:", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:120], file=sys.stderr)
    # poll
    for _ in range(45):
        time.sleep(10)
        q = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                           capture_output=True, text=True).stdout.splitlines()
        if len(q) >= 3 and "PENDING" not in q[2] and "complete" in q[2].lower():
            break
    # read top score
    q = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                       capture_output=True, text=True).stdout.splitlines()
    new_score = None
    if len(q) >= 3:
        import re
        m = re.search(r"(\d{4}\.\d+)", q[2])
        if m:
            new_score = float(m.group(1))
    return out, best_score, new_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--no-submit", action="store_true")
    a = ap.parse_args()
    tasknums = [int(x) for x in a.tasks.split(",") if x.strip()]

    banked, costs, newly = fresh_gate_and_bank(tasknums, a.k)
    gain = banked_gain(banked, costs)
    print(f"BANKED {len(banked)} tasks, pending gain +{gain:.3f} "
          f"(threshold {a.threshold})", file=sys.stderr)

    if a.no_submit or gain < a.threshold or not banked:
        return 0

    out, best_score, new_score = build_and_submit(banked, costs)
    print(f"SUBMIT: best {best_score:.2f} -> new {new_score}", file=sys.stderr)
    if new_score and new_score > best_score + 0.001:
        arch = REPO / "artifacts" / f"_BEST_{new_score:.2f}.zip"
        shutil.copy(out, arch)
        BEST_PTR.write_text(f"{arch}\t{new_score:.2f}")
        # banked tasks are now reflected in best -> update costs, clear banked
        for t, c in banked.items():
            costs[t] = c
        json.dump({str(a_): b for a_, b in costs.items()}, open(COSTS, "w"))
        json.dump({}, open(BANKED, "w"))
        for f in STAGE.glob("*.onnx"):
            f.unlink()
        print(f"NEW BEST {new_score:.2f} -> {arch.name}; banked cleared", file=sys.stderr)
    else:
        print(f"no improvement ({new_score} <= {best_score:.2f}); keeping best, "
              f"banked retained for next attempt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
