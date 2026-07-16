#!/usr/bin/env python3
"""Freeze task343 authority, exact control, and known counterexample controls."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
SOURCES = {
    "exact178": (
        ROOT / "scripts/golf/scratch_codex_plus10/wave2_sound/task343_sound_exact.onnx",
        "b47938285ea00b04aebea8709dd448c9983f0e3c8c6284050314097af0525c1b",
    ),
    "bad172_classifier_a": (
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task343_r01_static172.onnx",
        "6ada3c411cf90b4bcb42ff69e47eee35ed1c1b7d8b842c96c5c02c0eb06bec9e",
    ),
    "bad172_classifier_b": (
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task343_r02_static172.onnx",
        "c1047d40b875d37a7a9e28a52a47e2c569f5156924691118082aaca4ed5198e6",
    ),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA:
        raise RuntimeError("authority SHA drift")
    controls = HERE / "controls"
    controls.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        data = archive.read("task343.onnx")
    authority_member_sha = "7d64c3eda1167f322d8981531e433e7195e54d48e16e29c771b52a379af17ab1"
    if digest(data) != authority_member_sha:
        raise RuntimeError("authority task343 member drift")
    path = controls / "authority173.onnx"
    path.write_bytes(data)
    records.append(
        {
            "label": "authority173",
            "path": str(path.relative_to(ROOT)),
            "sha256": authority_member_sha,
            "source": "submission_base_8009.46.zip::task343.onnx",
        }
    )
    for label, (source, expected) in SOURCES.items():
        data = source.read_bytes()
        if digest(data) != expected:
            raise RuntimeError(f"{label} SHA drift")
        path = controls / f"{label}.onnx"
        path.write_bytes(data)
        records.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "sha256": expected,
                "source": str(source.relative_to(ROOT)),
            }
        )
    payload = {
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": AUTHORITY_SHA,
        "controls": records,
    }
    (HERE / "controls_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
