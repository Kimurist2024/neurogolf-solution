#!/usr/bin/env python3
"""Mine previously accepted NeuroGolf candidates against the 8004.50 payload.

This is deliberately evidence-first.  It only inventories records that an older
lane explicitly accepted, deduplicates them by (task, SHA-256), and performs a
truthful static structural screen.  Runtime/gold/fresh revalidation is a later
gate; this script never edits a submission ZIP.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile
import csv

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8004.50.zip"
OUT = HERE / "history_inventory.json"
TASK_MEMBER = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
PRIVATE_ZERO_MONITORED = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def score(cost: int) -> float:
    return 25.0 if cost == 0 else max(1.0, 25.0 - math.log(cost))


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(d.dim_value) if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]


def structural_audit(data: bytes) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "checker": False,
        "strict": False,
        "static": False,
        "cost": None,
        "memory": None,
        "params": None,
        "ub": [],
        "rejections": [],
    }
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        audit["checker"] = True
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        audit["strict"] = True
    except Exception as exc:
        audit["rejections"].append(f"checker_or_strict:{type(exc).__name__}:{exc}")
        return audit

    if model.functions:
        audit["rejections"].append("functions")
    if model.graph.sparse_initializer:
        audit["rejections"].append("sparse_initializer")
    if any(op.domain not in {"", "ai.onnx"} for op in model.opset_import):
        audit["rejections"].append("nonstandard_domain")
    if any(init.external_data or init.data_location == onnx.TensorProto.EXTERNAL for init in model.graph.initializer):
        audit["rejections"].append("external_data")

    giant = 0
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            audit["rejections"].append(f"banned:{node.op_type}")
        if node.op_type == "TfIdfVectorizer":
            audit["rejections"].append("lookup_tfidf")
        if node.op_type == "Einsum" and len(node.input) >= 15:
            giant = max(giant, len(node.input))
        if any(attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS} for attr in node.attribute):
            audit["rejections"].append("nested_graph")
    if giant:
        audit["rejections"].append(f"giant_einsum:{giant}")

    values = {
        item.name: item
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    init = {item.name: item for item in inferred.graph.initializer}
    params = sum(int(np.prod(item.dims, dtype=np.int64)) for item in inferred.graph.initializer)
    memory = 0
    outputs = {name for node in inferred.graph.node for name in node.output if name}
    for name in outputs:
        if name in {"input", "output"} or name in init:
            continue
        value = values.get(name)
        shape = [] if value is None else dims(value)
        if value is None or not shape or any(d is None or d <= 0 for d in shape):
            audit["rejections"].append(f"nonstatic:{name}")
            continue
        try:
            itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)).itemsize
        except Exception:
            audit["rejections"].append(f"unknown_dtype:{name}")
            continue
        memory += int(np.prod(shape, dtype=np.int64)) * itemsize
    audit["static"] = not any(str(reason).startswith(("nonstatic:", "unknown_dtype:")) for reason in audit["rejections"])

    for node in inferred.graph.node:
        bias_index = 8 if node.op_type == "QLinearConv" else (2 if node.op_type in {"Conv", "ConvTranspose"} else None)
        if bias_index is None or len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = init.get(node.input[bias_index])
        out = values.get(node.output[0])
        out_shape = [] if out is None else dims(out)
        if bias is None or len(out_shape) < 2 or out_shape[1] is None:
            audit["ub"].append({"node": node.name, "reason": "dynamic_or_unknown_bias"})
            continue
        bias_len = int(np.prod(bias.dims, dtype=np.int64))
        if bias_len != out_shape[1]:
            audit["ub"].append({"node": node.name, "bias": bias_len, "channels": out_shape[1]})
    if audit["ub"]:
        audit["rejections"].append("conv_bias_ub")
    if not audit["rejections"]:
        audit.update({"cost": int(memory + params), "memory": int(memory), "params": int(params)})
    return audit


def accepted_verdict(value: Any) -> bool:
    text = str(value or "").upper()
    return text.startswith("ACCEPT") or text in {"WINNER", "PASS", "ADOPT", "ADOPTED"}


def possible_records(payload: Any, source: Path) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, dict):
        return
    candidate_container = payload.get("candidate") if isinstance(payload.get("candidate"), str) else None
    if isinstance(payload.get("candidate"), dict):
        candidate_container = payload["candidate"].get("path") or payload["candidate"].get("zip")
    if isinstance(payload.get("candidate_archive"), dict):
        candidate_container = payload["candidate_archive"].get("zip") or candidate_container

    winners = payload.get("winners")
    if isinstance(winners, list):
        for row in winners:
            if isinstance(row, dict) and isinstance(row.get("task"), int):
                yield {**row, "_record_kind": "winner", "_container": candidate_container}
    elif isinstance(winners, dict):
        for key, row in winners.items():
            if not isinstance(row, dict):
                continue
            task = row.get("task")
            if task is None and str(key).isdigit():
                task = int(key)
            if isinstance(task, int):
                yield {**row, "task": task, "_record_kind": "winner", "_container": candidate_container}

    accepted = payload.get("accepted")
    if isinstance(accepted, dict):
        for key, row in accepted.items():
            if isinstance(row, dict):
                task = row.get("task")
                if task is None and str(key).isdigit():
                    task = int(key)
                if isinstance(task, int):
                    yield {**row, "task": task, "_record_kind": "accepted", "_container": candidate_container}
    elif isinstance(accepted, list):
        for row in accepted:
            if isinstance(row, dict) and isinstance(row.get("task"), int):
                yield {**row, "_record_kind": "accepted", "_container": candidate_container}

    decisions = payload.get("decisions")
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, dict) or not isinstance(row.get("task"), int):
                continue
            if accepted_verdict(row.get("verdict") or row.get("decision")):
                yield {**row, "_record_kind": "decision", "_container": candidate_container}

    candidate_decisions = payload.get("candidate_decisions")
    if isinstance(candidate_decisions, list):
        for row in candidate_decisions:
            if not isinstance(row, dict) or not isinstance(row.get("task"), int):
                continue
            if accepted_verdict(row.get("verdict") or row.get("decision")):
                yield {**row, "_record_kind": "candidate_decision", "_container": candidate_container}


def candidate_fields(row: dict[str, Any]) -> tuple[str | None, str | None, int | None]:
    candidate = row.get("candidate")
    path: str | None = None
    digest: str | None = None
    cost: int | None = None
    if isinstance(candidate, str):
        path = candidate
    elif isinstance(candidate, dict):
        path_value = candidate.get("path")
        path = path_value if isinstance(path_value, str) else None
        digest = candidate.get("sha256")
        cost = candidate.get("cost")
    path = (
        path
        or row.get("path")
        or row.get("candidate_path")
        or row.get("candidate_model")
        or row.get("onnx")
    )
    digest = digest or row.get("sha256") or row.get("candidate_sha256")
    cost = cost or row.get("candidate_cost") or row.get("cost_after") or row.get("truthful_cost")
    return (str(path) if path else None, str(digest) if digest else None, int(cost) if isinstance(cost, (int, float)) else None)


def model_bytes(task: int, path: str | None, container: str | None) -> tuple[bytes | None, str | None]:
    sources = [item for item in (path, container) if item]
    for source in sources:
        if "::" in source:
            archive_text, member = source.split("::", 1)
            archive = ROOT / archive_text
            try:
                with zipfile.ZipFile(archive) as handle:
                    return handle.read(member), source
            except Exception:
                continue
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_file() and candidate.suffix.lower() == ".onnx":
            try:
                return candidate.read_bytes(), str(candidate.relative_to(ROOT))
            except Exception:
                continue
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(candidate) as handle:
                    return handle.read(f"task{task:03d}.onnx"), f"{candidate.relative_to(ROOT)}::task{task:03d}.onnx"
            except Exception:
                continue
    return None, None


def collect_json_paths() -> list[Path]:
    roots = [
        ROOT / "scripts/golf/loop_7999_13",
        ROOT / "scripts/golf/loop_8000_46",
        ROOT / "scripts/golf/loop_8002_63",
        ROOT / "scripts/golf/loop_8003_40",
        ROOT / "scripts/golf/loop_8004_42",
    ]
    selected: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            lower = path.name.lower()
            if "known_winner" in lower or "pre_fresh" in lower:
                continue
            if "winner_manifest" in lower or lower == "final_report.json" or "compare" in lower:
                selected.add(path)
    for root in (ROOT / "scripts/golf").glob("scratch*"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            lower = path.name.lower()
            if "known_winner" in lower or "pre_fresh" in lower:
                continue
            if "winner_manifest" in lower or "final" in lower and "manifest" in lower or "compare" in lower:
                selected.add(path)
    newest = ROOT / "scripts/golf/loop_8004_42"
    if newest.exists():
        for path in newest.rglob("*.json"):
            lower = path.name.lower()
            if "known_winner" in lower or "pre_fresh" in lower:
                continue
            if "winner_manifest" in lower or lower in {"results.json", "final_report.json"}:
                selected.add(path)
    return sorted(selected)


def ledger_costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE) as archive:
        current_data = {task: archive.read(f"task{task:03d}.onnx") for task in range(1, 401)}
    costs = ledger_costs()
    current = {
        task: {"sha256": sha(data), "audit": structural_audit(data), "ledger_cost": costs[task]}
        for task, data in current_data.items()
    }

    records: dict[tuple[int, str], dict[str, Any]] = {}
    parse_errors: list[str] = []
    files = collect_json_paths()
    for json_path in files:
        try:
            payload = json.loads(json_path.read_text())
        except Exception as exc:
            parse_errors.append(f"{json_path.relative_to(ROOT)}:{exc}")
            continue
        for row in possible_records(payload, json_path):
            task = int(row["task"])
            path, claimed_sha, claimed_cost = candidate_fields(row)
            data, resolved = model_bytes(task, path, row.get("_container"))
            if data is None:
                continue
            digest = sha(data)
            audit = structural_audit(data)
            key = (task, digest)
            existing = records.get(key)
            lineage = str(json_path.relative_to(ROOT))
            if existing is None:
                records[key] = {
                    "task": task,
                    "candidate_path": resolved,
                    "sha256": digest,
                    "claimed_sha256": claimed_sha,
                    "claimed_cost": claimed_cost,
                    "audit": audit,
                    "evidence_record": row,
                    "lineage": [lineage],
                }
            elif lineage not in existing["lineage"]:
                existing["lineage"].append(lineage)

    rows: list[dict[str, Any]] = []
    for (task, digest), row in records.items():
        current_audit = current[task]["audit"]
        candidate_audit = row["audit"]
        # Historical final manifests use the official-like runtime profiler.
        # Prefer that measured cost over the static floor, which can be badly
        # understated when a model carries false/small value_info declarations.
        candidate_cost = row.get("claimed_cost")
        if not isinstance(candidate_cost, int):
            candidate_cost = candidate_audit.get("cost")
        current_cost = current[task]["ledger_cost"]
        gain = None
        if isinstance(candidate_cost, int) and isinstance(current_cost, int) and candidate_cost < current_cost:
            gain = score(candidate_cost) - score(current_cost)
        row.update({
            "current_sha256": current[task]["sha256"],
            "current_cost": current_cost,
            "candidate_cost": candidate_cost,
            "projected_gain_vs_8004_50": gain,
            "same_as_current": digest == current[task]["sha256"],
            "private_zero_monitored": task in PRIVATE_ZERO_MONITORED,
            "eligible_structural_lead": bool(
                digest != current[task]["sha256"]
                and gain is not None
                and not candidate_audit["rejections"]
                and not candidate_audit["ub"]
                and task != 153
                and task not in PRIVATE_ZERO_MONITORED
            ),
        })
        rows.append(row)

    rows.sort(key=lambda item: (
        not item["eligible_structural_lead"],
        -(item["projected_gain_vs_8004_50"] or -1e9),
        item["task"],
    ))
    output = {
        "baseline": {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": sha(BASE.read_bytes()),
            "lb_score": 8004.50,
        },
        "json_files_scanned": len(files),
        "parse_errors": parse_errors,
        "accepted_records_unique_task_sha": len(rows),
        "structural_lead_count": sum(bool(row["eligible_structural_lead"]) for row in rows),
        "structural_leads": [row for row in rows if row["eligible_structural_lead"]],
        "all_records": rows,
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n")
    print(json.dumps({key: output[key] for key in ("json_files_scanned", "accepted_records_unique_task_sha", "structural_lead_count")}, indent=2))
    for row in output["structural_leads"]:
        print(f"task{row['task']:03d} {row['current_cost']}->{row['candidate_cost']} gain={row['projected_gain_vs_8004_50']:.9f} {row['candidate_path']}")


if __name__ == "__main__":
    main()
