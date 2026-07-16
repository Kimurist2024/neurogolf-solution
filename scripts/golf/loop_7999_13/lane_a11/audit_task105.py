#!/usr/bin/env python3
"""Independent strict audit of the retained task105 batch-axis fold."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SUPPORT = ROOT / "scripts/golf/loop_7999_13/lane_a10/audit_task048.py"
spec = importlib.util.spec_from_file_location("a10_strict_audit_support", SUPPORT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load strict audit support")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


TASK = 105
BASELINE = HERE / "baseline" / "task105.onnx"
CANDIDATE = HERE / "candidate_task105_static198.onnx"


def main() -> None:
    audit.TASK = TASK
    audit.TASK_HASH = "4612dd53"
    audit.COUNT = 5000
    audit.BASELINE = BASELINE
    audit.CANDIDATE = CANDIDATE
    ort.set_default_logger_severity(4)

    model = onnx.load(CANDIDATE)
    profile = audit.scoring.score_and_verify(
        copy.deepcopy(model),
        TASK,
        str(HERE / "score_work"),
        label="a11_105",
        require_correct=True,
    )
    margin_ok, margin_min = audit.scoring.model_margin_stable(copy.deepcopy(model), TASK)
    baseline_structure = audit.structural(BASELINE)
    candidate_structure = audit.structural(CANDIDATE)
    report = {
        "task": TASK,
        "baseline": baseline_structure,
        "candidate": candidate_structure,
        "comparison": {
            "node_count_delta": candidate_structure["node_count"] - baseline_structure["node_count"],
            "parameter_delta": candidate_structure["parameter_elements"]
            - baseline_structure["parameter_elements"],
            "value_info_delta": candidate_structure["value_info_count"]
            - baseline_structure["value_info_count"],
            "executable_change": "remove one_f[1] and use canonical input batch axis (fixed dimension 1) as the singleton output label in six Einsums",
            "metadata_only": False,
        },
        "profile": profile,
        "projected_gain": math.log(199.0 / 198.0),
        "known_disable_all": audit.known(audit.make_session(CANDIDATE, True)),
        "known_default_ort": audit.known(audit.make_session(CANDIDATE, False)),
        "fresh_disable_all": audit.fresh(True, 8_105_000_001),
        "fresh_default_ort": audit.fresh(False, 8_105_000_002),
        "official_gold": audit.official_gold(CANDIDATE, TASK),
        "margin": {"stable": bool(margin_ok), "minimum": margin_min},
    }
    (HERE / "task105_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
