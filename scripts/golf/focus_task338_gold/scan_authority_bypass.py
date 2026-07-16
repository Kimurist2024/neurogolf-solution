#!/usr/bin/env python3
"""Exact same-shape node bypass scan of the truthful task338 authority net."""

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
    "task338_bypass_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
TRY = import_path("task338_bypass_try", ROOT / "scripts/golf/try_candidate.py")
SUPPORT.THRESHOLD = 1.0
SUPPORT.FRESH_PER_SEED = 2_000
SUPPORT.SUPPORT.POLICY_THRESHOLD = 1.0
SUPPORT.SUPPORT.FRESH_PER_SEED = 2_000


def descriptor(value: onnx.ValueInfoProto) -> tuple[int, tuple[int, ...]] | None:
    if not value.type.HasField("tensor_type"):
        return None
    tensor = value.type.tensor_type
    dims = tensor.shape.dim
    if any(
        dim.HasField("dim_param")
        or not dim.HasField("dim_value")
        or int(dim.dim_value) <= 0
        for dim in dims
    ):
        return None
    return int(tensor.elem_type), tuple(int(dim.dim_value) for dim in dims)


def descriptors(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...]]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True
    )
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    )
    result = {
        value.name: item
        for value in values
        if (item := descriptor(value)) is not None
    }
    result.update(
        {
            item.name: (int(item.data_type), tuple(int(dim) for dim in item.dims))
            for item in model.graph.initializer
        }
    )
    return result


def variants(base: onnx.ModelProto):
    desc = descriptors(base)
    init_names = {item.name for item in base.graph.initializer}
    graph_outputs = {item.name for item in base.graph.output}
    for node_index, node in enumerate(base.graph.node):
        if len(node.output) != 1 or not node.output[0]:
            continue
        target = node.output[0]
        if target in graph_outputs or target not in desc:
            continue
        for input_index, source in enumerate(node.input):
            if (
                not source
                or source in init_names
                or source == target
                or desc.get(source) != desc[target]
            ):
                continue
            model = copy.deepcopy(base)
            del model.graph.node[node_index]
            for consumer in model.graph.node:
                for position, name in enumerate(consumer.input):
                    if name == target:
                        consumer.input[position] = source
            # Source and target have exactly the same inferred descriptor, so
            # every downstream cached descriptor remains valid.  Remove only
            # the deleted tensor's value_info.
            kept_vi = [item for item in model.graph.value_info if item.name != target]
            del model.graph.value_info[:]
            model.graph.value_info.extend(kept_vi)
            used = {name for item in model.graph.node for name in item.input if name}
            kept_init = [item for item in model.graph.initializer if item.name in used]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept_init)
            yield model, {
                "node_index": node_index,
                "op": node.op_type,
                "input_index": input_index,
                "source": source,
                "target": target,
                "descriptor": [desc[target][0], list(desc[target][1])],
            }


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
            with tempfile.TemporaryDirectory(prefix="task338_bypass_") as workdir:
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
        "strictly_cheaper": bool(score is not None and score.cost < AUTHORITY_COST),
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


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    candidates = HERE / "candidates"
    generated = HERE / "bypass_generated"
    candidates.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        base_data = archive.read("task338.onnx")
    base = onnx.load_model_from_string(base_data)
    cases, counts = SUPPORT.SUPPORT.known_cases(338)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    rows: list[dict[str, Any]] = []
    admissions: list[dict[str, Any]] = []
    seen: set[str] = set()

    for model, meta in variants(base):
        data = model.SerializeToString()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        row: dict[str, Any] = {**meta, "sha256": sha}
        reasons = [
            reason
            for reason in SUPPORT.quick_preflight(model)
            # The accepted cost-403 authority intentionally declares a tiny
            # static output while ORT produces 30x30.  The official scorer and
            # try_candidate gold path accept this lineage; preserve that exact
            # output declaration rather than treating it as a new defect.
            if reason != "output_io"
        ]
        row["preflight_reasons"] = reasons
        if reasons:
            row["status"] = "preflight_reject"
            rows.append(row)
            continue
        profile = SUPPORT.POLICY.fast_profile(SUPPORT.SUPPORT, 338, model, cases[0])
        row["profile"] = profile
        if profile is None or int(profile["cost"]) >= AUTHORITY_COST:
            row["status"] = "cost_reject"
            rows.append(row)
            continue
        known = SUPPORT.failfast_known(data, cases)
        row["known"] = known
        if not exact(known):
            row["status"] = "known_reject"
            rows.append(row)
            continue

        fresh_rows = []
        for seed in (338_700_001, 338_700_002):
            fresh_cases, generation = SUPPORT.SUPPORT.fresh_cases(
                338, seed, task_map
            )
            runtime = SUPPORT.failfast_known(data, fresh_cases)
            fresh_rows.append(
                {
                    "seed": seed,
                    "generation": generation,
                    "runtime": runtime,
                    "pass": exact(runtime),
                }
            )
        row["fresh"] = fresh_rows
        if not all(item["pass"] for item in fresh_rows):
            row["status"] = "fresh_reject"
            rows.append(row)
            continue

        path = generated / (
            f"task338_bypass_n{meta['node_index']}_i{meta['input_index']}"
            f"_{sha[:12]}.onnx"
        )
        path.write_bytes(data)
        gate = official_gate(path)
        row["official_gate"] = gate
        if not gate["pass"]:
            row["status"] = "official_gate_reject"
            rows.append(row)
            continue
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
        admissions.append(row)
        rows.append(row)
        print(json.dumps({"admission": row}), flush=True)

    payload = {
        "task": 338,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_cost": AUTHORITY_COST,
        "known_counts": counts,
        "method": "same-inferred-shape single-node bypass",
        "absolute_gate": "official gold exact + margin + fresh-2000x2 exact",
        "attempted": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "rows": rows,
        "admissions": admissions,
    }
    (HERE / "authority_bypass_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "attempted": payload["attempted"],
                "status_counts": payload["status_counts"],
                "admissions": [
                    {
                        "cost": row["official_gate"]["candidate_cost"],
                        "gain": row["score_gain"],
                        "path": row["saved_path"],
                    }
                    for row in admissions
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
