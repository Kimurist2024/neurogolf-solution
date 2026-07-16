#!/usr/bin/env python3
"""Run the shared structural/cost audit over C14 archive leads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
TARGETS = (69, 71, 79, 91, 99, 105, 109)
ARCHIVES = (
    HERE.parent / "lane_archive_top200",
    HERE.parent / "lane_archive_loose_sweep",
)
PROBES = (
    (
        "probe_task069_spec_padfix",
        69,
        HERE.parents[1] / "scratch_codex" / "task069" / "candidate_compact_padfix.onnx",
    ),
    (
        "probe_task071_truthful_v20",
        71,
        HERE.parents[1] / "scratch" / "task071" / "cand_v20.onnx",
    ),
    (
        "probe_task091_int8_reduce",
        91,
        HERE.parents[1] / "scratch_codex" / "task091_cont" / "int8_reduce.onnx",
    ),
    (
        "probe_task109_global_lppool",
        109,
        HERE.parents[1] / "scratch_codex" / "task109" / "global_lppool.onnx",
    ),
    (
        "probe_task069_v1",
        69,
        HERE.parents[1] / "scratch_codex" / "task069" / "candidate_v1.onnx",
    ),
    (
        "probe_task069_v2",
        69,
        HERE.parents[1] / "scratch_codex" / "task069" / "candidate_v2.onnx",
    ),
    (
        "probe_task069_v3",
        69,
        HERE.parents[1] / "scratch_codex" / "task069" / "candidate_v3.onnx",
    ),
    (
        "probe_task069_padfix",
        69,
        HERE.parents[1] / "scratch_codex" / "task069" / "candidate_padfix.onnx",
    ),
    (
        "probe_task091_v6",
        91,
        HERE.parents[1] / "scratch" / "task091" / "candidate_v6.onnx",
    ),
    (
        "probe_task109_rebuild669",
        109,
        HERE.parents[1] / "scratch_codex" / "task109" / "rebuild_669.onnx",
    ),
    (
        "probe_task109_shared_axes",
        109,
        HERE.parents[1] / "scratch_codex" / "task109" / "shared_axes_rebuild.onnx",
    ),
    (
        "probe_task109_direct_reduce_f32",
        109,
        HERE.parents[1] / "scratch_codex" / "task109" / "direct_reduce_f32.onnx",
    ),
)


def load_auditor():
    path = HERE.parent / "lane_c11" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("c11_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="full-match regex for labels")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    auditor = load_auditor()
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = {}
    if args.resume and output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
    jobs: list[tuple[str, int, Path]] = [
        (f"base_task{task:03d}", task, HERE / "base" / f"task{task:03d}.onnx")
        for task in TARGETS
    ]
    jobs.extend(PROBES)
    pattern = re.compile(r"task(\d{3})_r(\d{2})_static\d+\.onnx$")
    seen: set[tuple[int, str]] = set()
    for archive in ARCHIVES:
        for path in sorted(archive.glob("task*_r*_static*.onnx")):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            task = int(match.group(1))
            key = (task, match.group(2))
            if task not in TARGETS or key in seen:
                continue
            seen.add(key)
            jobs.append((f"task{task:03d}_r{match.group(2)}", task, path))
    for label, task, path in jobs:
        if args.only and re.fullmatch(args.only, label) is None:
            continue
        output[label] = auditor.audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score")
        print(
            label,
            None if score is None else score.get("cost"),
            None if score is None else score.get("correct"),
            flush=True,
        )


if __name__ == "__main__":
    main()
