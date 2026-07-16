#!/usr/bin/env python3
"""Fill the task012 depthwise-Conv geometry gap left by the lane-c4 sweep.

The old evidence only covered windows whose two sides were at least seven.
For this rule each side must be at least five: a single source pixel must be
distinguishable at both relative offsets -2 and +2 along that axis.  This
script therefore exhausts every requested still-missing geometry below the
incumbent 70-weight area and every legal alignment, using the same complete
generator domain and hard linear-separation constraints as lane c4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "lane_c4" / "search_task012_lp.py"
SPEC = importlib.util.spec_from_file_location("task012_old_search", SOURCE)
assert SPEC is not None and SPEC.loader is not None
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)


def weak_affine_solve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    """Require positives >=1 and negatives <=0, exactly matching raw>0."""
    positive = y > 0
    z = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    aub = np.concatenate([-z[positive], z[~positive]], axis=0)
    bub = np.concatenate(
        [-np.ones(int(positive.sum())), np.zeros(int((~positive).sum()))]
    )
    result = linprog(
        np.zeros(z.shape[1]),
        A_ub=aub,
        b_ub=bub,
        bounds=[(None, None)] * z.shape[1],
        method="highs",
    )
    info: dict[str, object] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "constraint_count": int(len(x)),
        "negative_boundary_allowed": True,
    }
    if not result.success:
        return None, info
    scores = z @ result.x
    info.update(
        positive_min=float(scores[positive].min()),
        negative_max=float(scores[~positive].max()),
        maximum_abs_coefficient=float(np.abs(result.x).max()),
    )
    return result.x, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=10)
    parser.add_argument(
        "--long-side-only",
        action="store_true",
        help="scan only shapes with a side above ten (extension of the first run)",
    )
    parser.add_argument(
        "--weak-negatives",
        action="store_true",
        help="allow exact-decoder negative scores to equal zero",
    )
    parser.add_argument(
        "--covered-only",
        action="store_true",
        help="scan the prior kh>=7 and kw>=7 region (for the weak affine boundary)",
    )
    args = parser.parse_args()
    examples = OLD.exhaustive_examples()
    results: list[dict[str, object]] = []
    winners: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    shapes = sorted(
        (
            (kh * kw, kh, kw)
            for kh in range(5, args.max_side + 1)
            for kw in range(5, args.max_side + 1)
            if kh * kw < 70
            and (args.covered_only or kh < 7 or kw < 7)
            and (not args.covered_only or (kh >= 7 and kw >= 7))
            and (not args.long_side_only or max(kh, kw) > 10)
        )
    )
    for area, kh, kw in shapes:
        print("shape", kh, kw, "area", area, flush=True)
        for pt in range(kh):
            for pl in range(kw):
                row: dict[str, object] = {
                    "kh": kh,
                    "kw": kw,
                    "pad_top": pt,
                    "pad_left": pl,
                }
                sols: list[np.ndarray] = []
                for channel, label in ((0, "background"), (1, "foreground")):
                    x, y, error = OLD.unique_constraints(
                        examples, channel, kh, kw, pt, pl
                    )
                    if error:
                        row[label] = {"success": False, "reason": error}
                        break
                    sol, info = (
                        weak_affine_solve(x, y) if args.weak_negatives else OLD.solve(x, y)
                    )
                    row[label] = info
                    if sol is None:
                        break
                    sols.append(sol)
                row["success"] = len(sols) == 2
                results.append(row)
                if len(sols) == 2:
                    winners.append((kh, kw, pt, pl, sols[0], sols[1]))
                    print("WIN", kh, kw, pt, pl, "params", 10 * area + 10, flush=True)
        if winners:
            break

    evidence = {
        "generator_domain_examples": len(examples),
        "lower_bound": {
            "minimum_height": 5,
            "minimum_width": 5,
            "reason": "one source pixel must affect outputs at both signed radius-2 offsets",
        },
        "old_sweep_covered": "kh>=7 and kw>=7",
        "searched_shapes": [list(s) for s in shapes],
        "searched_alignments": len(results),
        "winner_count": len(winners),
        "negative_boundary_allowed": args.weak_negatives,
        "results": results,
    }
    suffix = "_weak" if args.weak_negatives else ""
    if args.covered_only:
        suffix += "_covered"
    if args.long_side_only:
        suffix += "_long_side"
    (HERE / f"missing_biased_search{suffix}.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    if not winners:
        print("no feasible missing geometry")
        return 1

    outdir = HERE / "candidates"
    outdir.mkdir(exist_ok=True)
    for kh, kw, pt, pl, bg, fg in winners:
        model = OLD.make_model(kh, kw, pt, pl, bg, fg)
        path = outdir / f"task012_h{kh}w{kw}_pt{pt}pl{pl}_bias.onnx"
        onnx.save(model, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
