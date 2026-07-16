#!/usr/bin/env python3
"""Freeze and audit task044/117/330 from immutable 8009.46 authority."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
MEMBERS = {
    44: "12b6414193b8716e15c4129f02bf7a8cddf31159f609427824345239de080492",
    117: "042e3ee0976af0c684fb98064800ab0b84e8bf53273a0c4121315ab7a0bfaac2",
    330: "af2a81db8b4b16f913ec05c689cb04e2894e288b6f124c2424c7aa438b9bfd0e",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA:
        raise RuntimeError("authority hash mismatch")
    baseline = HERE / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, expected in MEMBERS.items():
            data = archive.read(f"task{task:03d}.onnx")
            if sha256(data) != expected:
                raise RuntimeError(f"task{task:03d} member hash mismatch")
            (baseline / f"task{task:03d}.onnx").write_bytes(data)

    auditor = load_module(
        "lane153_auditor",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    output: dict[str, object] = {
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": AUTHORITY_SHA,
        "tasks": {},
    }
    for task in MEMBERS:
        path = baseline / f"task{task:03d}.onnx"
        output["tasks"][str(task)] = auditor.audit(f"base_task{task:03d}", task, path)
        (HERE / "baseline_audit.json").write_text(json.dumps(output, indent=2) + "\n")
        print(task, output["tasks"][str(task)].get("official_like_score"), flush=True)


if __name__ == "__main__":
    main()
