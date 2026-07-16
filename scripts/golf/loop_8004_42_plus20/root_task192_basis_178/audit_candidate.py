#!/usr/bin/env python3
"""Run the established fail-closed task192 audit on the shared-basis model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/audit_task192_exact_poly.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("root178_task192_audit", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)
    audit.HERE = HERE
    audit.ROOT = ROOT
    audit.CANDIDATE = HERE / "candidates/task192_shared_basis_argmax.onnx"
    audit.AUTHORITY = ROOT / "submission_base_8009.46.zip"
    audit.AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
    audit.BASELINE_MEMBER_SHA256 = "e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c"
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
