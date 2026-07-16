#!/usr/bin/env python3
"""Run the C11 structural/cost audit over every retained C12 archive lead."""

from __future__ import annotations

import importlib.util
import argparse
import json
import re
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (102, 124, 132, 163, 175, 178, 228)
ARCHIVE = HERE.parent / "lane_archive_top200"


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
    pattern = re.compile(r"task(\d{3})_r(\d{2})_static\d+\.onnx$")
    for path in sorted(ARCHIVE.glob("task*_r*_static*.onnx")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        task = int(match.group(1))
        if task in TARGETS:
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
