#!/usr/bin/env python3
"""Replace all graph value_info with clean strict-inference annotations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
from onnx import shape_inference


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    model = onnx.load(source)
    clean = copy.deepcopy(model)
    del clean.graph.value_info[:]
    inferred = shape_inference.infer_shapes(clean, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    before = cost_of(str(source))
    after = cost_of(str(output))
    report = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "value_info_before": len(model.graph.value_info),
        "value_info_after": len(inferred.graph.value_info),
        "cost_before": {"memory": before[0], "params": before[1], "cost": before[2]},
        "cost_after": {"memory": after[0], "params": after[1], "cost": after[2]},
        "checker_full": True,
        "strict_data_prop": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
