#!/usr/bin/env python3
"""Probe whether unused required/variadic outputs may be omitted safely."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


AUTHORITY = ROOT / "submission_base_8000.46.zip"
PROBES = {
    19: (11, 0, "Split_variadic_first"),
    80: (18, 0, "MaxPool_required_value_0"),
    124: (5, 3, "Split_variadic_middle"),
    131: (36, 0, "TopK_required_values"),
    400: (4, 0, "MaxPool_required_value_0"),
}


def main() -> None:
    rejected = HERE / "rejected_multi_output_probes"
    rejected.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, (node_index, output_index, label) in PROBES.items():
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            node = model.graph.node[node_index]
            original_output = node.output[output_index]
            candidate = copy.deepcopy(model)
            candidate.graph.node[node_index].output[output_index] = ""
            retained = [value for value in candidate.graph.value_info if value.name != original_output]
            del candidate.graph.value_info[:]
            candidate.graph.value_info.extend(retained)
            row = {
                "task": task,
                "label": label,
                "node_index": node_index,
                "op": node.op_type,
                "output_index": output_index,
                "original_output": original_output,
                "checker_full": False,
                "strict_shape": False,
            }
            try:
                onnx.checker.check_model(candidate, full_check=True)
                row["checker_full"] = True
            except Exception as exc:  # noqa: BLE001
                row["checker_error"] = f"{type(exc).__name__}: {exc}"
            if row["checker_full"]:
                try:
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    row["strict_shape"] = True
                except Exception as exc:  # noqa: BLE001
                    row["strict_shape_error"] = f"{type(exc).__name__}: {exc}"
            path = rejected / f"task{task:03d}_{label}.onnx"
            onnx.save(candidate, path)
            row["probe"] = str(path.relative_to(ROOT))
            row["probe_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if row["checker_full"] and row["strict_shape"]:
                command = [
                    sys.executable,
                    str(HERE / "probe_model_child.py"),
                    "--task",
                    str(task),
                    "--model",
                    str(path),
                ]
                try:
                    child = subprocess.run(
                        command,
                        cwd=ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                    row["runtime_child_returncode"] = child.returncode
                    row["runtime_child_stdout"] = child.stdout[-4000:]
                    row["runtime_child_stderr"] = child.stderr[-4000:]
                    if child.returncode == 0:
                        row["runtime_and_cost"] = json.loads(child.stdout.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    row["runtime_child_timeout"] = True
            rows.append(row)
            (HERE / "multi_output_omission_probes.partial.json").write_text(
                json.dumps({"probes": rows}, indent=2) + "\n"
            )
    result = {"probes": rows, "passing_checker": [row["task"] for row in rows if row["checker_full"]]}
    (HERE / "multi_output_omission_probes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
