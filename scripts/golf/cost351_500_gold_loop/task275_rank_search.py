#!/usr/bin/env python3
"""Search smaller shared rank factors in the sound task275 Einsum rule."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LANE = import_path("task275_lane_support", HERE / "worker.py")
TASK = 275
AUTHORITY_COST = 428


def replace_initializer(
    model: onnx.ModelProto, name: str, array: np.ndarray
) -> None:
    for item in model.graph.initializer:
        if item.name == name:
            item.CopyFrom(numpy_helper.from_array(array, name=name))
            return
    raise KeyError(name)


def fresh_exact(data: bytes) -> list[dict[str, Any]]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    rows = []
    for seed in (275_351_500_1, 275_351_500_2):
        cases, generation = LANE.BASE.SUPPORT.fresh_cases(TASK, seed, task_map)
        runtime = LANE.BASE.failfast_known(data, cases)
        rows.append(
            {
                "seed": seed,
                "generation": generation,
                "runtime": runtime,
                "pass": bool(
                    runtime.get("early_reject_reason") is None
                    and LANE.BASE.runtime_pass(runtime)
                ),
            }
        )
    return rows


def main() -> int:
    generated = HERE / "task275_rank_generated"
    accepted = HERE / "candidates"
    generated.mkdir(parents=True, exist_ok=True)
    accepted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(LANE.AUTHORITY) as archive:
        base = onnx.load_model_from_string(archive.read("task275.onnx"))
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in base.graph.initializer
    }

    rows = []
    for rank in (1, 2):
        for subset in itertools.combinations(range(3), rank):
            model = copy.deepcopy(base)
            index = np.asarray(subset, dtype=np.int64)
            replace_initializer(model, "S", arrays["S"][index, :])
            replace_initializer(
                model, "T", arrays["T"][np.ix_(index, index)]
            )
            replace_initializer(
                model, "W", arrays["W"][np.ix_(index, index)]
            )
            data = model.SerializeToString()
            sha = hashlib.sha256(data).hexdigest()
            path = generated / (
                f"task275_rank{rank}_{''.join(map(str, subset))}_{sha[:12]}.onnx"
            )
            path.write_bytes(data)
            gate = LANE.official_gate(path, TASK, AUTHORITY_COST)
            row: dict[str, Any] = {
                "rank": rank,
                "subset": list(subset),
                "sha256": sha,
                "path": str(path.relative_to(ROOT)),
                "official_gate": gate,
                "status": "official_gate_reject",
            }
            if gate["pass"]:
                fresh = fresh_exact(data)
                row["fresh"] = fresh
                if all(item["pass"] for item in fresh):
                    cost = int(gate["candidate_cost"])
                    saved = accepted / f"task275_GOLD_cost{cost}_{sha[:12]}.onnx"
                    shutil.copy2(path, saved)
                    row.update(
                        {
                            "status": "admit",
                            "saved_path": str(saved.relative_to(ROOT)),
                            "score_gain": math.log(AUTHORITY_COST / cost),
                        }
                    )
                else:
                    row["status"] = "fresh_reject"
            rows.append(row)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "subset": subset,
                        "cost": gate["candidate_cost"],
                        "gold": gate["official_gold_exact"],
                        "status": row["status"],
                    }
                ),
                flush=True,
            )

    admissions = [row for row in rows if row["status"] == "admit"]
    payload = {
        "task": TASK,
        "authority_cost": AUTHORITY_COST,
        "method": "shared S/T/W contraction-rank subset search",
        "absolute_gate": "try_candidate official gold exact + margin + structure + fresh-2000x2 exact",
        "rows": rows,
        "admissions": admissions,
    }
    (HERE / "task275_rank_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"admissions": admissions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
