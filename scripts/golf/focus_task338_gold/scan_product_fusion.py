#!/usr/bin/env python3
"""Fuse exact fp16 PRelu mask-product chains in the truthful task338 net."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
AUTHORITY_COST = 403


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = import_path(
    "task338_fusion_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
TRY = import_path("task338_fusion_try", ROOT / "scripts/golf/try_candidate.py")
SUPPORT.THRESHOLD = 1.0
SUPPORT.FRESH_PER_SEED = 2_000
SUPPORT.SUPPORT.POLICY_THRESHOLD = 1.0
SUPPORT.SUPPORT.FRESH_PER_SEED = 2_000


CHAINS: dict[str, dict[str, Any]] = {
    "initial_vertical": {
        "indices": [24, 25],
        "inputs": ["neg_red", "red_up_mb", "red_down_pb"],
        "output": "vert_prelu2",
    },
    "initial_horizontal": {
        "indices": [26, 27],
        "inputs": ["neg_red", "not_up", "not_down"],
        "output": "horiz_prelu4",
    },
    "left7": {
        "indices": [99, 100, 101, 102, 103, 104],
        "inputs": [
            "L_s7_neg0", "L2_s7_not", "L3_s7_not", "L4_s7_not",
            "L5_s7_not", "L6_mb", "L7_mb",
        ],
        "output": "L_s7_prelu7",
    },
    "right7": {
        "indices": [126, 127, 128, 129, 130, 131],
        "inputs": [
            "R_s7_neg0", "R2_pb", "R3_pb", "R4_pb", "R5_pb",
            "R6_pb", "R7_pb",
        ],
        "output": "R_s7_prelu7",
    },
    "up7": {
        "indices": [143, 144, 145, 146, 147, 148],
        "inputs": [
            "U_s7_neg0", "U2_s7_not", "U3_s7_not", "U4_s7_not",
            "U5_s7_not", "U6_s7_not", "U7_mb",
        ],
        "output": "U_s7_prelu7",
    },
    "down7": {
        "indices": [151, 152, 153, 154, 155, 156],
        "inputs": [
            "D_s7_neg0", "D2_pb", "D3_pb", "D4_pb", "D5_pb",
            "D6_pb", "D7_pb",
        ],
        "output": "D_s7_prelu7",
    },
    "support": {
        "indices": [161, 162, 163],
        "inputs": ["sup_neg0", "D1_raw", "D4_raw", "D7_pb"],
        "output": "sup_prelu3",
    },
    "candidate": {
        "indices": [166, 167, 168, 169, 170, 171],
        "inputs": [
            "neg_valid", "not_red", "L7_seen", "R7_seen", "U7_seen",
            "D7_seen", "sup",
        ],
        "output": "cand_prelu6",
    },
}


def fuse(base: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    specs = [CHAINS[name] for name in names]
    removed = {index for spec in specs for index in spec["indices"]}
    insertion = {min(spec["indices"]): spec for spec in specs}
    new_nodes = []
    removed_outputs = set()
    for spec in specs:
        for index in spec["indices"][:-1]:
            removed_outputs.update(base.graph.node[index].output)
    for index, node in enumerate(base.graph.node):
        if index in insertion:
            spec = insertion[index]
            term = "abcd"
            equation = ",".join([term] * len(spec["inputs"])) + "->" + term
            new_nodes.append(
                helper.make_node(
                    "Einsum",
                    list(spec["inputs"]),
                    [str(spec["output"])],
                    equation=equation,
                    name=f"fused_{spec['output']}",
                )
            )
        if index not in removed:
            new_nodes.append(copy.deepcopy(node))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    kept_vi = [item for item in model.graph.value_info if item.name not in removed_outputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    return model


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("early_reject_reason") is None
    )


def official_gate(path: Path) -> dict[str, Any]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        file_ok = TRY._validate_file_size(path)
        model = onnx.load(str(path)) if file_ok else None
        structure_ok = bool(model is not None and TRY._validate_ops_and_shapes(model))
        gold_ok = False
        margin_ok = False
        score = None
        mismatch = None
        minimum_positive = None
        if structure_ok and model is not None:
            gold_ok, mismatch = TRY._verify_gold(model, 338)
        if gold_ok and model is not None:
            margin_ok, minimum_positive = TRY._check_margin(model, 338)
        if margin_ok and model is not None:
            with tempfile.TemporaryDirectory(prefix="task338_fusion_") as workdir:
                score = TRY._score_model(
                    model, 338, workdir, "candidate", require_correct=True
                )
    return {
        "file_ok": file_ok,
        "structure_ok": structure_ok,
        "official_gold_exact": gold_ok,
        "margin_ok": margin_ok,
        "minimum_positive": minimum_positive,
        "candidate_cost": None if score is None else int(score.cost),
        "pass": bool(
            file_ok
            and structure_ok
            and gold_ok
            and margin_ok
            and score is not None
            and score.cost < AUTHORITY_COST
        ),
        "first_mismatch": None
        if mismatch is None
        else {"subset": mismatch.subset, "index": mismatch.index},
        "log": stream.getvalue(),
    }


def audit(
    base: onnx.ModelProto,
    names: list[str],
    known_cases: list[dict[str, Any]],
    fresh_streams: list[tuple[int, list[dict[str, Any]], dict[str, Any]]],
    generated: Path,
    candidates: Path,
) -> dict[str, Any]:
    model = fuse(base, names)
    data = model.SerializeToString()
    sha = hashlib.sha256(data).hexdigest()
    row: dict[str, Any] = {"chains": names, "sha256": sha}
    reasons = [reason for reason in SUPPORT.quick_preflight(model) if reason != "output_io"]
    row["preflight_reasons"] = reasons
    if reasons:
        row["status"] = "preflight_reject"
        return row
    known = SUPPORT.failfast_known(data, known_cases)
    row["known"] = known
    if not exact(known):
        row["status"] = "known_reject"
        return row
    fresh = []
    for seed, cases, generation in fresh_streams:
        runtime = SUPPORT.failfast_known(data, cases)
        fresh.append(
            {
                "seed": seed,
                "generation": generation,
                "runtime": runtime,
                "pass": exact(runtime),
            }
        )
    row["fresh"] = fresh
    if not all(item["pass"] for item in fresh):
        row["status"] = "fresh_reject"
        return row
    path = generated / f"task338_fusion_{'_'.join(names)}_{sha[:12]}.onnx"
    path.write_bytes(data)
    gate = official_gate(path)
    row["official_gate"] = gate
    if not gate["pass"]:
        row["status"] = "official_gate_reject"
        return row
    cost = int(gate["candidate_cost"])
    saved = candidates / f"task338_GOLD_cost{cost}_{sha[:12]}.onnx"
    shutil.copy2(path, saved)
    row.update(
        {
            "status": "admit",
            "saved_path": str(saved.relative_to(ROOT)),
            "score_gain": math.log(AUTHORITY_COST / cost),
        }
    )
    return row


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    generated = HERE / "fusion_generated"
    candidates = HERE / "candidates"
    generated.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        base = onnx.load_model_from_string(archive.read("task338.onnx"))
    known_cases, known_counts = SUPPORT.SUPPORT.known_cases(338)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    fresh_streams = []
    for seed in (338_800_001, 338_800_002):
        cases, generation = SUPPORT.SUPPORT.fresh_cases(338, seed, task_map)
        fresh_streams.append((seed, cases, generation))

    rows = []
    admitted_names = []
    for name in CHAINS:
        row = audit(
            base, [name], known_cases, fresh_streams, generated, candidates
        )
        rows.append(row)
        print(json.dumps({"chains": [name], "status": row["status"], "gate": row.get("official_gate")}), flush=True)
        if row["status"] == "admit":
            admitted_names.append(name)

    # If exact single fusions exist, verify their cumulative composition from
    # the original authority in one independent pass.
    if len(admitted_names) >= 2:
        row = audit(
            base, admitted_names, known_cases, fresh_streams, generated, candidates
        )
        row["cumulative"] = True
        rows.append(row)
        print(json.dumps({"chains": admitted_names, "status": row["status"], "gate": row.get("official_gate")}), flush=True)

    admissions = [row for row in rows if row["status"] == "admit"]
    payload = {
        "task": 338,
        "authority_cost": AUTHORITY_COST,
        "known_counts": known_counts,
        "method": "exact variadic-Einsum fusion of PRelu mask products",
        "absolute_gate": "official gold exact + margin + fresh-2000x2 exact",
        "rows": rows,
        "admissions": admissions,
    }
    (HERE / "product_fusion_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"admission_count": len(admissions), "admissions": admissions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
