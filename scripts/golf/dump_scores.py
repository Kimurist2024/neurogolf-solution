#!/usr/bin/env python3
"""Re-score every task in the current best submission and dump all_scores.csv
sorted by score ascending (lowest first).

Columns: rank,task,hash,cost,score,archetype
  cost  = score_and_verify cost on the net in the best zip (require_correct=False)
  score = max(1, 25 - ln(cost)); 0.0000 when the net fails to score locally
  archetype = generator common.* helper calls (minus ubiquitous helpers)

Usage: dump_scores.py [--best <zip>] [--out all_scores.csv]
       default best = pointer in docs/golf/campaign_best.txt
"""
from __future__ import annotations
import argparse, json, math, re, sys, tempfile, zipfile
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402

TASKS = REPO / "inputs" / "arc-gen-repo" / "tasks"
HMAP = json.load(open(REPO / "docs" / "golf" / "task_hash_map.json"))
UBI = {"grids", "grid", "randint", "randints", "random_color", "random_colors",
       "choice", "choices", "sample", "shuffle", "random_el", "deepcopy",
       "flatten", "black", "blue", "red", "green", "yellow", "gray", "pink",
       "orange", "cyan", "maroon", "set_colors", "remove_duplicates", "isclose",
       "sqrt", "int_sqrt"}
CALL = re.compile(r"common\.([a-z_]+)\s*\(")


def archetype(h: str) -> str:
    p = TASKS / f"task_{h}.py"
    if not p.is_file():
        return ""
    fns = sorted({m.group(1) for m in CALL.finditer(p.read_text(errors="ignore"))} - UBI)
    return "|".join(fns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", type=Path,
                    default=Path((REPO / "docs/golf/campaign_best.txt").read_text().split("\t")[0]))
    ap.add_argument("--out", type=Path, default=REPO / "all_scores.csv")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.best)
    rows = []
    with tempfile.TemporaryDirectory() as wd:
        for t in range(1, 401):
            n = f"task{t:03d}.onnx"
            h = HMAP.get(f"{t:03d}", "")
            cost, sc = 0, 0.0
            if n in z.namelist():
                s = scoring.score_and_verify(onnx.load_model_from_string(z.read(n)),
                                             t, wd, label="x", require_correct=False)
                if s and s.get("cost"):
                    cost = s["cost"]
                    sc = max(1.0, 25 - math.log(cost))
            rows.append((t, h, cost, sc, archetype(h)))

    rows.sort(key=lambda r: r[3])  # score ascending (lowest first)
    lines = ["rank,task,hash,cost,score,archetype"]
    for rank, (t, h, cost, sc, arch) in enumerate(rows, 1):
        lines.append(f"{rank},task{t:03d},{h},{cost},{sc:.4f},{arch}")
    a.out.write_text("\n".join(lines) + "\n")
    total = sum(r[3] for r in rows)
    print(f"wrote {a.out} ({len(rows)} tasks, sum_score {total:.2f}, base {a.best.name})")
    print("lowest 8:")
    for t, h, cost, sc, arch in rows[:8]:
        print(f"  task{t:03d} cost={cost} score={sc:.4f} [{arch}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
