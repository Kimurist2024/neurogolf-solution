#!/usr/bin/env python3
"""Freeze and audit task005/297/308 from immutable 8009.46 authority."""

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
    5: "77eb35fdcf2dbbacaa1c63d2dfef4f3b50ecbfbc8178da3bc2e7883ee8275c57",
    297: "cdba3d03bf43853742508f284bf98ca5341fdb2ab50042ec895afb0069296537",
    308: "fc845e9edee06830a880be6f385f2601d1d0ff7f017cb54b64b36cb84da7785d",
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
    base = HERE / "baseline"
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, expected in MEMBERS.items():
            data = archive.read(f"task{task:03d}.onnx")
            if sha256(data) != expected:
                raise RuntimeError(f"task{task:03d} member hash mismatch")
            (base / f"task{task:03d}.onnx").write_bytes(data)
    auditor = load_module(
        "lane140_auditor",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    output = {
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": AUTHORITY_SHA,
        "tasks": {},
    }
    for task in MEMBERS:
        path = base / f"task{task:03d}.onnx"
        output["tasks"][str(task)] = auditor.audit(f"base_task{task:03d}", task, path)
        (HERE / "baseline_audit.json").write_text(json.dumps(output, indent=2) + "\n")
        print(task, output["tasks"][str(task)].get("official_like_score"), flush=True)


if __name__ == "__main__":
    main()
