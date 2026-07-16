#!/usr/bin/env python3
"""Build the remaining exact/near-exact global factor-reuse candidates.

This is a scratch-only builder.  It reuses the audited in-Einsum rewrite used
for task137 and never mutates the baseline archive or score ledgers.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from build_factor_reuse_small import BASE, ROOT, load, rewrite, save


OUT = Path(__file__).resolve().parent / "lane_global_factor_candidates"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configurations = {
        36: ("F", "K_f0a", "left"),
        51: ("J2", "J1", "left"),
        66: ("WR", "Tcol", "left"),
    }
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        for task, (target, source, side) in configurations.items():
            model = load(archive, task)
            try:
                change = rewrite(model, target, source, side, f"task{task:03d}")
                path = OUT / f"task{task:03d}.onnx"
                save(model, path)
                rows.append(
                    {
                        "task": task,
                        "path": str(path.relative_to(ROOT)),
                        "change": change,
                        "built": True,
                    }
                )
            except Exception as exc:
                rows.append({"task": task, "built": False, "error": repr(exc)})
    (OUT / "build_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
