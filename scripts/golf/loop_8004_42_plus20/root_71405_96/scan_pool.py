#!/usr/bin/env python3
"""Read-only official-cost inventory for the newly arrived others/71405 pool."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
POOL = ROOT / "others/71405"
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TASK_RE = re.compile(r"^task(\d{3})", re.IGNORECASE)

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(model: onnx.ModelProto, task: int, workdir: str, label: str) -> tuple[dict | None, str]:
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            result = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label=label, require_correct=False
            )
        return result, capture.getvalue()[-4000:]
    except Exception as exc:  # runtime failures remain evidence, not promotion
        return None, f"{type(exc).__name__}: {exc}\n{capture.getvalue()[-3500:]}"


def static_audit(model: onnx.ModelProto) -> dict:
    full_check = True
    strict_shape = True
    full_error = None
    shape_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        full_check = False
        full_error = f"{type(exc).__name__}: {exc}"
    try:
        shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        strict_shape = False
        shape_error = f"{type(exc).__name__}: {exc}"
    domains = sorted({item.domain for item in model.opset_import})
    return {
        "full_check": full_check,
        "full_check_error": full_error,
        "strict_shape_inference": strict_shape,
        "strict_shape_error": shape_error,
        "standard_domains": all(domain in ("", "ai.onnx") for domain in domains),
        "domains": domains,
        "functions": len(model.functions),
        "conv_bias_ub": [list(item) for item in check_conv_bias(model)],
    }


def main() -> int:
    # The pool intentionally contains many stale or false-declared shapes.
    # Keep their classification in JSON without flooding the terminal with
    # millions of native ORT warning lines.
    ort.set_default_logger_severity(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-task", type=int, default=1)
    parser.add_argument("--skip-task", action="append", type=int, default=[])
    args = parser.parse_args()

    got = sha256(AUTHORITY.read_bytes())
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")

    sources: list[tuple[int, Path]] = []
    for path in sorted(POOL.glob("*.onnx")):
        match = TASK_RE.match(path.name)
        if match:
            task = int(match.group(1))
            if task >= args.min_task and task not in set(args.skip_task):
                sources.append((task, path))

    rows: list[dict] = []
    baseline_cache: dict[int, dict] = {}
    with zipfile.ZipFile(AUTHORITY) as archive, tempfile.TemporaryDirectory(
        prefix="root_71405_96_", dir="/tmp"
    ) as workdir:
        for index, (task, path) in enumerate(sources, 1):
            data = path.read_bytes()
            digest = sha256(data)
            member = f"task{task:03d}.onnx"
            base_data = archive.read(member)
            base_digest = sha256(base_data)
            if task not in baseline_cache:
                base_model = onnx.load_model_from_string(base_data)
                result, log = official(base_model, task, workdir, f"base_{task:03d}")
                baseline_cache[task] = {"sha256": base_digest, "official": result, "log_tail": log}

            try:
                model = onnx.load_model_from_string(data)
                load_error = None
            except Exception as exc:
                model = None
                load_error = f"{type(exc).__name__}: {exc}"
            audit = None if model is None else static_audit(model)
            result, log = (None, "") if model is None else official(
                model, task, workdir, f"cand_{task:03d}_{digest[:8]}"
            )
            base = baseline_cache[task]["official"]
            base_cost = None if base is None else int(base["cost"])
            candidate_cost = None if result is None else int(result["cost"])
            lower = (
                base_cost is not None
                and candidate_cost is not None
                and 0 < candidate_cost < base_cost
            )
            row = {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "bytes": len(data),
                "authority_sha256": base_digest,
                "identical_to_authority": digest == base_digest,
                "load_error": load_error,
                "static": audit,
                "authority_official": base,
                "candidate_official": result,
                "strictly_lower": lower,
                "projected_gain": math.log(base_cost / candidate_cost) if lower else 0.0,
                "candidate_log_tail": log,
            }
            rows.append(row)
            print(
                f"{index:02d}/{len(sources)} task{task:03d} "
                f"cost={candidate_cost}/{base_cost} correct={None if result is None else result['correct']} "
                f"lower={lower}",
                flush=True,
            )

    winners = [
        row
        for row in rows
        if row["strictly_lower"]
        and row["candidate_official"] is not None
        and row["candidate_official"]["correct"]
        and row["static"] is not None
        and row["static"]["full_check"]
        and row["static"]["strict_shape_inference"]
        and row["static"]["standard_domains"]
        and row["static"]["functions"] == 0
        and not row["static"]["conv_bias_ub"]
    ]
    output = {
        "status": "OFFICIAL_COST_AND_KNOWN_ONLY_NOT_FIXED",
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": got,
        "pool": str(POOL.relative_to(ROOT)),
        "onnx_files": len(rows),
        "unique_tasks": len({row["task"] for row in rows}),
        "strict_lower_known_static_clean_count": len(winners),
        "strict_lower_known_static_clean": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["sha256"],
                "authority_cost": row["authority_official"]["cost"],
                "candidate_cost": row["candidate_official"]["cost"],
                "projected_gain": row["projected_gain"],
            }
            for row in winners
        ],
        "baseline_cache": {str(task): value for task, value in sorted(baseline_cache.items())},
        "rows": rows,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.min_task == 1 and not args.skip_task else f"_from_{args.min_task:03d}"
    (HERE / f"official_inventory{suffix}.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["strict_lower_known_static_clean"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
