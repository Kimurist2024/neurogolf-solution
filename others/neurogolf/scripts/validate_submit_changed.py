"""Dual-path validation of the changed-vs-6471.32 tasks before submission.

For each changed task it verifies the FROZEN-latest candidate is
(a) format-valid + gold-correct via scripts/lib/scoring.score_and_verify,
(b) gold-correct via the OFFICIAL neurogolf_utils.verify_subset (independent path),
(c) margin-stable (generalization proxy), and
(d) strictly cheaper than the grader-confirmed 6471.32 baseline.

Decision per task: ADOPT latest iff all of (a)-(d) hold, else FALLBACK to the
6471.32 baseline file (never regress). Emits a JSON verdict to stdout.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "inputs" / "neurogolf-2026" / "neurogolf_utils"))

# The official neurogolf_utils pulls in notebook-only deps (IPython, matplotlib,
# onnx_tool) used solely by its display/profiling helpers, which we never call.
# Stub them so the pure verification functions import cleanly headless.
import types  # noqa: E402

for _name in ("IPython", "IPython.display", "matplotlib", "matplotlib.pyplot", "onnx_tool"):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        if _name in ("IPython.display",):
            _mod.display = lambda *a, **k: None  # type: ignore[attr-defined]
            _mod.FileLink = lambda *a, **k: None  # type: ignore[attr-defined]
        sys.modules[_name] = _mod
# Wire IPython.display as attribute of IPython, matplotlib.pyplot as plt root.
sys.modules["IPython"].display = sys.modules["IPython.display"]  # type: ignore[attr-defined]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]  # type: ignore[attr-defined]

from lib import scoring  # noqa: E402
import neurogolf_utils as official  # noqa: E402

import os  # noqa: E402

BUILD = Path(os.environ.get("VAL_BUILD", REPO / "artifacts" / "submit_build_stage"))
BASE = Path(os.environ.get("VAL_BASE", REPO / "artifacts" / "baseline_6471"))

CHANGED = [int(x) for x in sys.argv[1:]] or [185, 383, 400]


def official_correct(onnx_path: Path, task_num: int) -> tuple[bool, str]:
    """Independent correctness via the OFFICIAL neurogolf_utils pipeline.

    Uses the official sanitize_model + verify_subset + score_network (independent
    of scripts/lib), but feeds them the LOCAL gold data (the official
    load_examples hardcodes a /kaggle/input path that does not exist locally).
    """
    try:
        sanitized = official.sanitize_model(onnx.load(str(onnx_path)))
        if not sanitized:
            return False, "sanitize_failed"
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory(prefix="off_") as od:
            opts.profile_file_prefix = str(Path(od) / f"{task_num:03d}")
            sess = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
            ex = scoring.load_examples(task_num)  # local gold data
            ar, aw, _ = official.verify_subset(sess, ex["train"] + ex["test"])
            gr, gw, _ = official.verify_subset(sess, ex["arc-gen"])
            trace = sess.end_profiling()
            mem, params = official.score_network(sanitized, trace)
        ok = (aw + gw) == 0 and mem is not None and params is not None and mem >= 0 and params >= 0
        return ok, f"agi {ar}/{ar+aw} gen {gr}/{gr+gw} mem={mem} params={params}"
    except Exception as exc:  # noqa: BLE001
        return False, f"exc:{exc}"


def main() -> int:
    verdicts = []
    with tempfile.TemporaryDirectory(prefix="validate_") as wd:
        for t in CHANGED:
            latest = BUILD / f"task{t:03d}.onnx"
            base = BASE / f"task{t:03d}.onnx"

            # (d) baseline cost (require_correct=False: it is grader-proven)
            base_res = scoring.score_and_verify(
                onnx.load(str(base)), t, wd, label="base", require_correct=False
            )
            # (a) latest via scoring lib (must be correct + valid)
            lat_res = scoring.score_and_verify(
                onnx.load(str(latest)), t, wd, label="lat", require_correct=True
            )
            # (c) margin stability of latest
            stable, margin_min = scoring.model_margin_stable(onnx.load(str(latest)), t)
            # (b) independent official correctness of latest
            off_ok, off_detail = official_correct(latest, t)

            lib_ok = lat_res is not None and lat_res.get("correct") is True
            base_cost = base_res["cost"] if base_res else None
            lat_cost = lat_res["cost"] if lat_res else None
            cheaper = (
                base_cost is not None and lat_cost is not None and lat_cost < base_cost
            )
            adopt = bool(lib_ok and off_ok and stable and cheaper)

            verdicts.append(
                {
                    "task": t,
                    "decision": "ADOPT" if adopt else "FALLBACK",
                    "lib_correct": lib_ok,
                    "official_correct": off_ok,
                    "official_detail": off_detail,
                    "margin_stable": bool(stable),
                    "margin_min": margin_min,
                    "base_cost": base_cost,
                    "latest_cost": lat_cost,
                    "cheaper": cheaper,
                    "score_gain": (
                        round(lat_res["score"] - base_res["score"], 4)
                        if (lat_res and base_res)
                        else None
                    ),
                }
            )

    print(json.dumps({"changed": CHANGED, "verdicts": verdicts}, indent=2))
    n_adopt = sum(1 for v in verdicts if v["decision"] == "ADOPT")
    n_fb = len(verdicts) - n_adopt
    print(f"\nSUMMARY: ADOPT={n_adopt} FALLBACK={n_fb}", file=sys.stderr)
    # Non-zero exit only if a FALLBACK task is somehow worse than baseline
    # (impossible by construction) — adoption decisions are always safe.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
