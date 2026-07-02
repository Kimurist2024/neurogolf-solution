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
import argparse, csv, json, math, multiprocessing as mp, re, sys, tempfile, zipfile
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

# Nets with a giant single Einsum (many operands) hang ONNX Runtime LOCALLY even
# though the grader scores them fine. Detect them and score in a child process
# with a hard timeout; on timeout carry over the prior CSV score. The net itself
# is NEVER modified, so submission cost is unchanged.
MAX_EINSUM_OPERANDS = 15   # >= this many inputs on one Einsum node => hang-prone
HANG_TIMEOUT = 30          # seconds before we give up and carry over


def is_hang_prone(model: onnx.ModelProto) -> bool:
    for nd in model.graph.node:
        if nd.op_type == "Einsum" and len(nd.input) >= MAX_EINSUM_OPERANDS:
            return True
    return False


def _score_worker(data: bytes, t: int, q) -> None:
    try:
        with tempfile.TemporaryDirectory() as wd:
            s = scoring.score_and_verify(onnx.load_model_from_string(data), t, wd,
                                         label="x", require_correct=False)
        if s and s.get("score") is not None:
            cost = s.get("cost", 0)
            sc = s["score"] if cost == 0 else max(1.0, 25 - math.log(cost))
            q.put((cost, sc))
        else:
            q.put((0, 0.0))
    except Exception:
        q.put(None)


def score_in_subprocess(data: bytes, t: int, timeout: int):
    """Return (cost, score), or None on timeout/hang (caller carries over)."""
    q = mp.Queue()
    p = mp.Process(target=_score_worker, args=(data, t, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive():
            p.kill()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


def load_carryover(out: Path) -> dict[int, tuple[int, float]]:
    """Prior {task_int: (cost, score)} from an existing all_scores.csv, if any."""
    if not out.is_file():
        return {}
    carry = {}
    for r in csv.DictReader(open(out)):
        try:
            carry[int(r["task"].replace("task", ""))] = (int(r["cost"]), float(r["score"]))
        except (KeyError, ValueError):
            continue
    return carry


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
    carry = load_carryover(a.out)
    rows = []
    hang_tasks, carried = [], []
    with tempfile.TemporaryDirectory() as wd:
        for t in range(1, 401):
            n = f"task{t:03d}.onnx"
            h = HMAP.get(f"{t:03d}", "")
            print(f"scoring task{t:03d} ...", file=sys.stderr, flush=True)
            cost, sc = 0, 0.0
            if n in z.namelist():
                data = z.read(n)
                model = onnx.load_model_from_string(data)
                # タイムアウト無し: giant Einsum も含め全タスクを実 ORT 実行で採点
                # する(carry-over や静的計算はスコアがブレるため使わない)。
                s = scoring.score_and_verify(model, t, wd, label="x",
                                             require_correct=False)
                if s and s.get("score") is not None:
                    # cost may legitimately be 0 (params=0, memory=0 net); the
                    # scorer caps score at 25.0 in that case. Trust its score.
                    cost = s.get("cost", 0)
                    sc = s["score"] if cost == 0 else max(1.0, 25 - math.log(cost))
            rows.append((t, h, cost, sc, archetype(h)))

    rows.sort(key=lambda r: r[3])  # score ascending (lowest first)
    lines = ["rank,task,hash,cost,score,archetype"]
    for rank, (t, h, cost, sc, arch) in enumerate(rows, 1):
        lines.append(f"{rank},task{t:03d},{h},{cost},{sc:.4f},{arch}")
    a.out.write_text("\n".join(lines) + "\n")
    total = sum(r[3] for r in rows)
    print(f"wrote {a.out} ({len(rows)} tasks, sum_score {total:.2f}, base {a.best.name})")
    print(f"hang-prone (giant Einsum, scored in subprocess): {len(hang_tasks)}")
    if carried:
        print(f"  timed out -> carried over prior CSV score: "
              f"{[f'{t:03d}' for t in carried]}")
    print("lowest 8:")
    for t, h, cost, sc, arch in rows[:8]:
        print(f"  task{t:03d} cost={cost} score={sc:.4f} [{arch}]")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
