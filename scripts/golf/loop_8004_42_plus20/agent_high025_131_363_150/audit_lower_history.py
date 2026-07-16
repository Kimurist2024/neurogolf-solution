#!/usr/bin/env python3
"""Re-audit the only known actual-lower task025/task131 history members."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
CANDIDATES = (
    (25, "drop_sigv", ROOT / "scripts/golf/loop_7999_13/lane_a5/candidates/task025_drop_sigv.onnx"),
    (25, "drop_negsigk", ROOT / "scripts/golf/loop_7999_13/lane_a5/candidates/task025_drop_negsigk.onnx"),
    (25, "causal_mask", ROOT / "scripts/golf/loop_7999_13/lane_a9/task025_causal_mask.onnx"),
    (131, "archive_lookup_r02", ROOT / "scripts/golf/loop_7999_13/lane_c25/candidates/task131_archive_r02.onnx"),
    (131, "archive_lookup_r01", ROOT / "scripts/golf/loop_7999_13/lane_c25/candidates/task131_archive_r01.onnx"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high150_history_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high150_history_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    rows: list[dict[str, object]] = []
    base_cache = {
        task: (HERE / f"baseline/task{task:03d}.onnx").read_bytes()
        for task in (25, 131)
    }
    base_profiles = {
        task: SCAN.official_cost(data, f"high150_history_task{task:03d}_base")
        for task, data in base_cache.items()
    }
    for task, label, path in CANDIDATES:
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        profile = SCAN.official_cost(data, f"high150_history_{task:03d}_{label}")
        static = SCAN.structural(copy.deepcopy(model))
        ops = Counter(node.op_type for node in model.graph.node)
        max_einsum_inputs = max(
            (len([name for name in node.input if name]) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        )
        policy = {
            "giant_einsum": max_einsum_inputs > 8,
            "max_einsum_inputs": max_einsum_inputs,
            "lookup": any(node.op_type in {"TfIdfVectorizer", "CategoryMapper"} for node in model.graph.node),
            "op_histogram": dict(sorted(ops.items())),
        }
        configs = {}
        for disable, threads, config_label in CONFIGS:
            configs[config_label] = AUDIT.known_config(
                task, base_cache[task], data, disable, threads
            )
        known_perfect = all(item.get("perfect", False) for item in configs.values())
        # Trace even rejected lower models so the failure ledger distinguishes
        # semantic/raw mismatch from an independent runtime-shape contradiction.
        trace = safe_trace(task, data)
        reasons = []
        if not profile["cost"] < base_profiles[task]["cost"]:
            reasons.append("not_strict_lower")
        if not static.get("pass", False):
            reasons.append("static_gate")
        if policy["giant_einsum"]:
            reasons.append("giant_einsum")
        if policy["lookup"]:
            reasons.append("lookup")
        if not known_perfect:
            reasons.append("known_or_raw_equivalence")
        if not trace.get("truthful", False):
            reasons.append("runtime_shape_truth")
        row = {
            "task": task,
            "label": label,
            "source": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "serialized_bytes": len(data),
            "authority_sha256": digest(base_cache[task]),
            "authority_profile": base_profiles[task],
            "candidate_profile": profile,
            "strict_lower": profile["cost"] < base_profiles[task]["cost"],
            "static": static,
            "policy": policy,
            "known_four_configs": configs,
            "known_perfect_all_configs": known_perfect,
            "runtime_shape_trace": trace,
            "fresh": {"status": "not_run_after_pre_fresh_gate"},
            "accepted": not reasons,
            "reasons": reasons,
        }
        rows.append(row)
        print(
            f"task{task:03d} {label} cost={profile['cost']} known4={known_perfect} "
            f"truthful={trace.get('truthful')} reasons={','.join(reasons)}",
            flush=True,
        )
    payload = {
        "authority": "submission.zip",
        "authority_sha256": digest((ROOT / "submission.zip").read_bytes()),
        "rows": rows,
        "accepted": [row for row in rows if row["accepted"]],
    }
    (HERE / "audit/strict_lower_history.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
