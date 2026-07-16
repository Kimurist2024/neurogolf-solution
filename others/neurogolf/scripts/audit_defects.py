"""Structural-defect audit: run each current stage net against K FRESH generator
instances. A faithful spec-compiled net matches every fresh instance; an
example-fit net (the task004 landmine pattern) fails many. Also reports cost
(low score = high cost) and margin stability. Output: JSON per task.

Usage: audit_defects.py --tasks 1-400 [--k 200]
"""
from __future__ import annotations
import argparse, json, sys, random, tempfile, importlib
from pathlib import Path
import onnx, onnxruntime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "inputs" / "arc-gen-repo" / "tasks"))  # for 'import common'
from lib import scoring  # noqa: E402

MAP = json.load(open(REPO / "docs" / "golf" / "task_hash_map.json"))
STAGE = REPO / "artifacts" / "wave_opus" / "stage"


def make_session(model):
    san = scoring.sanitize_model(model)
    if not san:
        return None
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(san.SerializeToString(), opts)


def audit_task(t: int, k: int) -> dict:
    h = MAP[f"{t:03d}"]
    try:
        gen = importlib.import_module(f"task_{h}")
    except Exception as e:  # noqa: BLE001
        return {"task": t, "hash": h, "error": f"import:{e}"}
    random.seed(10_000 + t)
    fresh = []
    for _ in range(k):
        try:
            ex = gen.generate()
            if isinstance(ex, dict) and "input" in ex and "output" in ex:
                fresh.append(ex)
        except Exception:  # noqa: BLE001
            continue
    net = STAGE / f"task{t:03d}.onnx"
    try:
        model = onnx.load(str(net))
        sess = make_session(model)
        if sess is None:
            return {"task": t, "hash": h, "error": "session_build_failed"}
        right, wrong, _ = scoring.verify_subset(sess, fresh)
        with tempfile.TemporaryDirectory() as wd:
            r = scoring.score_and_verify(model, t, wd, label="a", require_correct=False)
        stable, mm = scoring.model_margin_stable(model, t)
    except Exception as e:  # noqa: BLE001
        return {"task": t, "hash": h, "error": f"run:{e}"}
    total = right + wrong
    return {
        "task": t, "hash": h,
        "cost": r["cost"] if r else None,
        "fresh_total": total, "fresh_fail": wrong,
        "fresh_fail_rate": round(wrong / total, 4) if total else None,
        "margin_stable": bool(stable), "margin_min": mm,
        "defect": wrong > 0,
    }


def parse_tasks(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1-400")
    ap.add_argument("--k", type=int, default=200)
    a = ap.parse_args()
    results = [audit_task(t, a.k) for t in parse_tasks(a.tasks)]
    print(json.dumps(results))
    defects = [r for r in results if r.get("defect")]
    errs = [r for r in results if r.get("error")]
    print(f"DEFECTS={len(defects)} ERRORS={len(errs)} OF {len(results)}", file=sys.stderr)
    for r in defects:
        print(f"  DEFECT task{r['task']:03d}: {r['fresh_fail']}/{r['fresh_total']} fresh fail, cost={r['cost']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
