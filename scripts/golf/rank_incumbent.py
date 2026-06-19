#!/usr/bin/env python3
"""Rank all tasks by the *incumbent* cost that try_candidate compares against.

incumbent = min(cost(optimized/taskXXX.onnx), cost(handcrafted/taskXXX.onnx)).
This is the real headroom a golf worker must beat to promote, so it is the
correct target signal (sub12's per-task cost can be much higher when the
submission carries a safe-but-expensive net while a cheaper one sits in
optimized/handcrafted).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO))
from scripts.golf.rank_dir import cost_of  # noqa: E402

OPT = REPO / "artifacts" / "optimized"
HAND = REPO / "artifacts" / "handcrafted"


def _job(task: int) -> tuple[int, int | None, int | None]:
    op = OPT / f"task{task:03d}.onnx"
    hc = HAND / f"task{task:03d}.onnx"
    co = cost_of(str(op))[2] if op.is_file() else None
    ch = cost_of(str(hc))[2] if hc.is_file() else None
    co = co if (co is not None and co >= 0) else None
    ch = ch if (ch is not None and ch >= 0) else None
    return (task, co, ch)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "golf" / "incumbent_costs.json"
    tasks = list(range(1, 401))
    rows: dict[int, dict] = {}
    workers = max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for task, co, ch in ex.map(_job, tasks):
            cand = [x for x in (co, ch) if x is not None]
            inc = min(cand) if cand else None
            src = None
            if inc is not None:
                src = "handcrafted" if (ch is not None and ch == inc) else "optimized"
            rows[task] = {"optimized": co, "handcrafted": ch, "incumbent": inc, "src": src}

    ranked = sorted(
        (t for t in rows if rows[t]["incumbent"] is not None),
        key=lambda t: -rows[t]["incumbent"],
    )
    payload = {
        "ranked": [{"task": t, **rows[t]} for t in ranked],
        "rows": {str(t): rows[t] for t in rows},
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(ranked)} ranked)")
    print("TOP 30 by incumbent cost:")
    for t in ranked[:30]:
        r = rows[t]
        print(f"  task{t:03d} incumbent={r['incumbent']} ({r['src']}) opt={r['optimized']} hand={r['handcrafted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
