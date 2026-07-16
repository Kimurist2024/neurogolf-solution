#!/usr/bin/env python3
"""Try conservative initializer, dead operand, bias, Identity, and no-op trims."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8006.61.zip"
TARGETS = (68, 175, 400, 30, 224, 281, 240, 183, 376, 59, 358, 20, 190, 302, 195, 300, 383, 193, 304, 384)

# task302 is private-zero catalogued. task059's exact authority is 0/266 on
# known×4, so behavior-preserving reductions cannot be admitted as sound.
FAIL_CLOSED_EQUIVALENT = {59, 302}

SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20c_87/build_safe_reductions.py"
SPEC = importlib.util.spec_from_file_location("mid20e89_safe_primitives", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load safe reduction primitives")
SAFE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFE)

SCAN_SOURCE = HERE / "scan_authority.py"
SCAN_SPEC = importlib.util.spec_from_file_location("mid20e89_known_quad", SCAN_SOURCE)
if SCAN_SPEC is None or SCAN_SPEC.loader is None:
    raise RuntimeError("cannot load known-quad wrapper")
SCAN = importlib.util.module_from_spec(SCAN_SPEC)
SCAN_SPEC.loader.exec_module(SCAN)

sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from screen_all import static_audit  # noqa: E402
from harvest import actual_screen, known_score  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    out_dir = HERE / "reduced_candidates"
    out_dir.mkdir(exist_ok=True)
    authority = json.loads((HERE / "authority_costs.json").read_text())["costs"]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TARGETS:
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(base_data)
            for kind in ("unused", "dedupe", "zero_bias", "identity", "noops", "combined"):
                model, actions = SAFE.transform(base, kind)
                if not actions["action_count"]:
                    continue
                data = model.SerializeToString()
                digest = sha256(data)
                if digest == sha256(base_data) or (task, digest) in seen:
                    continue
                seen.add((task, digest))
                path = out_dir / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
                path.write_bytes(data)
                row: dict[str, Any] = {
                    "task": task,
                    "kind": kind,
                    "path": rel(path),
                    "sha256": digest,
                    "authority_cost": int(authority[str(task)]),
                    "actions": actions,
                }
                audit = static_audit(data, [rel(BASE_ZIP)], task)
                row["static_audit"] = audit
                if task in FAIL_CLOSED_EQUIVALENT:
                    row.update(stage="policy_reject", reasons=["equivalent_to_private_or_known_incorrect_authority"])
                elif not audit["pass"]:
                    row.update(stage="static_reject", reasons=audit["reasons"])
                else:
                    actual = actual_screen(data, task)
                    row["actual_screen_cost"] = actual
                    if actual is None or actual >= int(authority[str(task)]):
                        row.update(stage="actual_reject", reasons=["actual_cost_not_strictly_lower"])
                    else:
                        profile = known_score(data, task, True, f"mid20e89_mech_{task}_{digest[:8]}")
                        row["official_like_score"] = profile
                        if not profile or not profile.get("correct") or int(profile["cost"]) >= int(authority[str(task)]):
                            row.update(stage="known_reject", reasons=["known_not_complete_or_not_cheaper"])
                        else:
                            row["known_quad"] = SCAN.SCANNER.known_dual(task, data)
                            if any(
                                mode.get("wrong") or mode.get("errors") or mode.get("session_error") or not mode.get("right")
                                for mode in row["known_quad"].values()
                            ):
                                row.update(stage="known_quad_reject", reasons=["known_quad_not_100_percent"])
                            else:
                                row["gain"] = math.log(int(authority[str(task)]) / int(profile["cost"]))
                                row.update(stage="pre_fresh", reasons=[])
                rows.append(row)
    report = {
        "baseline": rel(BASE_ZIP),
        "baseline_sha256": sha256(BASE_ZIP.read_bytes()),
        "targets": list(TARGETS),
        "candidate_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "rows": rows,
    }
    (HERE / "mechanical_reductions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "stage_counts": report["stage_counts"]}, indent=2))
    for row in rows:
        if row["stage"] == "pre_fresh":
            print("PRE_FRESH", row["task"], row["path"], row["official_like_score"]["cost"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
