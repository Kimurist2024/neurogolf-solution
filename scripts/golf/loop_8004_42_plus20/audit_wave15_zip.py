#!/usr/bin/env python3
"""Whole-archive audit for the six-task Wave15 promotion."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.golf.check_conv_bias import check_model  # noqa: E402


BASE = ROOT / "submission_base_8005.17.zip"
BUILT = HERE / "submission_8005.17_wave15_task013_158_254_267_323_333_safe_meta.zip"
EXPECTED_CHANGED = [
    "task013.onnx", "task158.onnx", "task254.onnx", "task267.onnx",
    "task323.onnx", "task333.onnx",
]
EXPECTED_SHA256 = {
    "task013.onnx": "ad4eb35978f3e38d1d3e2afdd55e55db871962cc2ea4c989675d9d583434103b",
    "task158.onnx": "3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba",
    "task254.onnx": "814ece451a8f8eda8e9221d58e2f4fb3359fa396dfe971f6ad97693f453b15f8",
    "task267.onnx": "4ca7f921c34f87ef71512a8b680de7c984a2b42cd55b338b57aaabc012321387",
    "task323.onnx": "db773b15ceea8c42fac7543b7b7e93e0fd56a73493c7a8122b587327544c5926",
    "task333.onnx": "0628a573302f0a816d010482ed8b883caac7c307a27f47c9b53df85e2042a6bc",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    with zipfile.ZipFile(BASE) as base, zipfile.ZipFile(BUILT) as built:
        if built.testzip() is not None:
            raise RuntimeError("archive CRC failure")
        if base.namelist() != built.namelist() or base.comment != built.comment:
            raise RuntimeError("archive order/comment mismatch")
        changed: list[str] = []
        metadata_mismatches: list[list[str]] = []
        short: list[list[object]] = []
        long: list[list[object]] = []
        model_load_errors: list[list[str]] = []
        networks = 0
        for old_info, new_info in zip(base.infolist(), built.infolist(), strict=True):
            old_data = base.read(old_info.filename)
            new_data = built.read(new_info.filename)
            if old_data != new_data:
                changed.append(old_info.filename)
            for attribute in (
                "date_time", "compress_type", "external_attr", "internal_attr",
                "flag_bits", "extra", "comment",
            ):
                if getattr(old_info, attribute) != getattr(new_info, attribute):
                    metadata_mismatches.append([old_info.filename, attribute])
            if not new_info.filename.endswith(".onnx"):
                continue
            networks += 1
            try:
                model = onnx.load_model_from_string(new_data)
                for operation, bias, bias_len, out_channels in check_model(model):
                    finding: list[object] = [new_info.filename, operation, bias, bias_len, out_channels]
                    (short if bias_len < out_channels else long).append(finding)
            except Exception as exc:  # noqa: BLE001
                model_load_errors.append([new_info.filename, repr(exc)])
        replacements = {name: sha(built.read(name)) for name in EXPECTED_CHANGED}

    if changed != EXPECTED_CHANGED or replacements != EXPECTED_SHA256:
        raise RuntimeError(f"payload mismatch changed={changed} hashes={replacements}")
    if metadata_mismatches or short or long or model_load_errors:
        raise RuntimeError(
            f"audit failed metadata={metadata_mismatches} short={short} long={long} load={model_load_errors}"
        )
    result = {
        "zip": str(BUILT.relative_to(ROOT)),
        "sha256": sha(BUILT.read_bytes()),
        "networks_checked": networks,
        "model_load_errors": model_load_errors,
        "conv_family_short_bias_findings": len(short),
        "conv_family_long_bias_findings": len(long),
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
        "changed_members_exact": changed,
        "replacement_sha256": replacements,
        "base_already_contains": ["task226.onnx LB-white cost372"],
        "excluded": ["task009 LB-black", "task036 regression", "task192 private-zero approximation"],
    }
    (HERE / "wave15_full_zip_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
