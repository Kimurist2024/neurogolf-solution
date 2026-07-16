#!/usr/bin/env python3
"""Independent dual-runtime fresh-5000 audit of truthful task134 archive r06."""

from __future__ import annotations

import hashlib
import json

import onnxruntime as ort

import audit_task134_r04_fresh as audit


audit.CANDIDATE = (
    audit.ROOT
    / "scripts/golf/loop_7999_13/lane_archive_top200/task134_r06_static322.onnx"
)
audit.SEEDS = {True: 13_406_799_913, False: 13_406_799_914}


def main() -> None:
    ort.set_default_logger_severity(4)
    result = {
        "task": audit.TASK,
        "path": str(audit.CANDIDATE.relative_to(audit.ROOT)),
        "sha256": hashlib.sha256(audit.CANDIDATE.read_bytes()).hexdigest(),
    }
    for disable_all, label in ((True, "disable_all"), (False, "default")):
        result[label] = audit.run(disable_all)
        (audit.HERE / "task134_r06_fresh5000.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        row = result[label]
        print(label, row["correct"], row["wrong"], row["errors"], flush=True)


if __name__ == "__main__":
    main()
