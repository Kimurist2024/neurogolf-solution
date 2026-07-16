#!/usr/bin/env python3
"""Build the four requested dead-node prunes as rejection-only probes."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import onnx

import scan_build_dead_code as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8000.46.zip"
TASKS = (39, 89, 122, 183)


def main() -> None:
    output_dir = HERE / "rejected_named_dead_prunes"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            row = {
                "task": task,
                "source_exclusions": shared.safety.source_exclusions(model),
            }
            try:
                candidate, changes = shared.prune(model)
                path = output_dir / f"task{task:03d}_dead_pruned.onnx"
                onnx.save(candidate, path)
                row.update(
                    {
                        "probe": str(path.relative_to(ROOT)),
                        "probe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "changes": changes,
                    }
                )
                command = [
                    sys.executable,
                    str(HERE / "probe_model_child.py"),
                    "--task",
                    str(task),
                    "--model",
                    str(path),
                ]
                child = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=180,
                )
                row["runtime_child_returncode"] = child.returncode
                row["runtime_child_stdout"] = child.stdout[-8000:]
                row["runtime_child_stderr"] = child.stderr[-8000:]
                if child.returncode == 0:
                    row["runtime_and_cost"] = json.loads(child.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                row["runtime_child_timeout"] = True
            except Exception as exc:  # noqa: BLE001
                row["build_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            (HERE / "named_dead_prune_probes.partial.json").write_text(
                json.dumps({"probes": rows}, indent=2) + "\n"
            )
    result = {"probes": rows, "accepted": []}
    (HERE / "named_dead_prune_probes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
