"""STRICT merge gate for a defect-fix candidate net.

Adopts a fix ONLY if it independently passes ALL of:
  1. High-K FRESH generator audit (default 5000 instances) -> ZERO failures.
  2. Dual-path visible-gold correctness (scripts/lib AND official neurogolf_utils).
  3. Margin stability.

Never trusts the worker's self-report. Prints a JSON verdict per (task, path).
Usage: verify_fix.py --task N --onnx PATH [--k 5000]
       verify_fix.py --batch task=path,task=path,... [--k 5000]
"""
from __future__ import annotations
import argparse, json, sys, random, tempfile, importlib
from pathlib import Path
import onnx, onnxruntime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "inputs" / "arc-gen-repo" / "tasks"))
# stub notebook-only deps so official neurogolf_utils imports headless
import types  # noqa: E402
for _n in ("IPython", "IPython.display", "matplotlib", "matplotlib.pyplot", "onnx_tool"):
    if _n not in sys.modules:
        m = types.ModuleType(_n)
        if _n == "IPython.display":
            m.display = lambda *a, **k: None
            m.FileLink = lambda *a, **k: None
        sys.modules[_n] = m
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

from lib import scoring  # noqa: E402
sys.path.insert(0, str(REPO / "inputs" / "neurogolf-2026" / "neurogolf_utils"))
import neurogolf_utils as official  # noqa: E402

MAP = json.load(open(REPO / "docs" / "golf" / "task_hash_map.json"))


def _raw_session(model):
    san = scoring.sanitize_model(model)
    if not san:
        return None
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(san.SerializeToString(), opts)


def fresh_audit(model, task: int, k: int) -> tuple[int, int]:
    h = MAP[f"{task:03d}"]
    gen = importlib.import_module(f"task_{h}")
    random.seed(777_000 + task)
    fresh = []
    for _ in range(k):
        try:
            ex = gen.generate()
            if isinstance(ex, dict) and "input" in ex and "output" in ex:
                fresh.append(ex)
        except Exception:  # noqa: BLE001
            continue
    sess = _raw_session(model)
    if sess is None:
        return -1, -1
    right, wrong, _ = scoring.verify_subset(sess, fresh)
    return right + wrong, wrong


def official_gold(path: Path, task: int) -> bool:
    try:
        san = official.sanitize_model(onnx.load(str(path)))
        if not san:
            return False
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory() as od:
            opts.profile_file_prefix = str(Path(od) / f"{task:03d}")
            sess = onnxruntime.InferenceSession(san.SerializeToString(), opts)
            ex = scoring.load_examples(task)
            ar, aw, _ = official.verify_subset(sess, ex["train"] + ex["test"])
            gr, gw, _ = official.verify_subset(sess, ex["arc-gen"])
            sess.end_profiling()
        return (aw + gw) == 0
    except Exception:  # noqa: BLE001
        return False


def verify_one(task: int, path: Path, k: int) -> dict:
    if not path.exists():
        return {"task": task, "decision": "REJECT", "reason": "missing_file"}
    model = onnx.load(str(path))
    with tempfile.TemporaryDirectory() as wd:
        lib = scoring.score_and_verify(model, task, wd, label="v", require_correct=True)
    lib_gold = bool(lib and lib.get("correct"))
    off_gold = official_gold(path, task)
    stable, mm = scoring.model_margin_stable(model, task)
    ftotal, ffail = fresh_audit(model, task, k)
    fresh_ok = ffail == 0 and ftotal > 0
    adopt = bool(lib_gold and off_gold and stable and fresh_ok)
    return {
        "task": task,
        "decision": "ADOPT" if adopt else "REJECT",
        "cost": lib["cost"] if lib else None,
        "lib_gold": lib_gold, "official_gold": off_gold,
        "margin_stable": bool(stable), "margin_min": mm,
        "fresh_total": ftotal, "fresh_fails": ffail, "fresh_ok": fresh_ok,
        "path": str(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--batch", help="task=path,task=path,...")
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    jobs = []
    if a.batch:
        for item in a.batch.split(","):
            t, p = item.split("=", 1)
            jobs.append((int(t), Path(p)))
    elif a.task and a.onnx:
        jobs.append((a.task, a.onnx))
    else:
        print("need --task/--onnx or --batch", file=sys.stderr)
        return 2
    verdicts = [verify_one(t, p, a.k) for t, p in jobs]
    print(json.dumps(verdicts, indent=2))
    ad = [v for v in verdicts if v["decision"] == "ADOPT"]
    print(f"\nADOPT={len(ad)}/{len(verdicts)}", file=sys.stderr)
    for v in verdicts:
        print(f"  task{v['task']:03d}: {v['decision']} "
              f"fresh={v.get('fresh_fails')}/{v.get('fresh_total')} "
              f"lib={v.get('lib_gold')} off={v.get('official_gold')} "
              f"margin={v.get('margin_min')} cost={v.get('cost')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
