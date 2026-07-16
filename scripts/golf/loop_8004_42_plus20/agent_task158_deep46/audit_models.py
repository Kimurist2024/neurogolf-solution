#!/usr/bin/env python3
"""Reprofile every distinct task158 history model against the 8005.16 member."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402


TASK = 158
BASE_COST = 7615
BASE_SHA = "4c029ccfc735a52db8eca9db9b3a032e8b512ac68e1e337ab21a090b67cfb208"
shared.BASE_COST = {TASK: BASE_COST}

SEARCH_ROOTS = (
    ROOT / "scripts/golf/loop_7999_13",
    ROOT / "scripts/golf/scratch_codex/task158",
    ROOT / "scripts/golf/scratch_codex_plus10",
    ROOT / "scripts/golf/loop_8000_46",
    ROOT / "scripts/golf/loop_8003_40",
    ROOT / "scripts/golf/loop_8004_42_plus20",
    ROOT / "others",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preference(path: Path) -> tuple[int, int, str]:
    text = str(path.relative_to(ROOT))
    ranking = 5
    if "lane_archive_loose_sweep" in text:
        ranking = 0
    elif "lane_archive_zip_sweep" in text:
        ranking = 1
    elif "lane_a22/sound" in text or "lane_a36" in text:
        ranking = 2
    elif "scratch_codex/task158" in text:
        ranking = 3
    elif text.startswith("scripts/"):
        ranking = 4
    return ranking, len(text), text


def inventory() -> list[dict[str, object]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    seen_paths: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.onnx"):
            if path in seen_paths or not path.name.lower().startswith("task158"):
                continue
            seen_paths.add(path)
            try:
                grouped[sha256(path)].append(path)
            except OSError:
                continue
    rows = []
    for digest, paths in grouped.items():
        ordered = sorted(paths, key=preference)
        rows.append(
            {
                "sha256": digest,
                "representative": str(ordered[0].relative_to(ROOT)),
                "sources": [str(path.relative_to(ROOT)) for path in ordered],
                "source_count": len(ordered),
                "is_current": digest == BASE_SHA,
            }
        )
    rows.sort(key=lambda row: (not row["is_current"], row["sha256"]))
    return rows


def nested_graph_count(model: onnx.ModelProto) -> int:
    return sum(
        1
        for node in model.graph.node
        for attribute in node.attribute
        if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    )


def main() -> int:
    ort.set_default_logger_severity(4)
    entries = inventory()
    manifest = {
        "task": TASK,
        "baseline_cost": BASE_COST,
        "baseline_sha256": BASE_SHA,
        "unique_models": len(entries),
        "entries": entries,
        "complete": True,
    }
    (HERE / "evidence/model_inventory.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    rows: list[dict[str, object]] = []
    for index, entry in enumerate(entries, 1):
        path = ROOT / str(entry["representative"])
        label = "task158_current" if entry["is_current"] else f"task158_hist_{index:02d}"
        print(f"[{index}/{len(entries)}] {label} {entry['sha256'][:12]} {path}", flush=True)
        try:
            row = shared.audit(
                TASK,
                label,
                path,
                None,
                list(entry["sources"]),
                baseline=bool(entry["is_current"]),
            )
            model = onnx.load(path)
            row["nested_graph_count"] = nested_graph_count(model)
            row["source_count"] = entry["source_count"]
            reasons = set(row.get("pre_fresh_reasons", []))
            if row["nested_graph_count"]:
                reasons.add("nested_graph")
            if row.get("shared_structure_gate") != "pass":
                reasons.add("shared_structure_gate")
            profile = row.get("official_like_score")
            row["strictly_cheaper_than_7615"] = bool(
                profile and int(profile["cost"]) < BASE_COST
            )
            row["pre_fresh_reasons"] = sorted(reasons)
            row["pre_fresh_pass"] = not reasons
        except Exception as exc:  # noqa: BLE001
            row = {
                "task": TASK,
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "sha256": entry["sha256"],
                "sources": entry["sources"],
                "audit_error": f"{type(exc).__name__}: {exc}",
                "strictly_cheaper_than_7615": False,
                "pre_fresh_pass": False,
                "pre_fresh_reasons": ["audit_error"],
            }
        rows.append(row)
        (HERE / "evidence/history_audit.json").write_text(
            json.dumps({"rows": rows, "complete": False}, indent=2) + "\n"
        )
        profile = row.get("official_like_score")
        print(
            "  actual=",
            profile.get("cost") if profile else None,
            "known=",
            row.get("known_disable_all"),
            row.get("known_default"),
            "reasons=",
            row["pre_fresh_reasons"],
            flush=True,
        )

    cheaper = [row for row in rows if row.get("strictly_cheaper_than_7615")]
    known100 = [
        row
        for row in cheaper
        if all(
            isinstance(row.get(key), dict)
            and row[key].get("right") == 266
            and row[key].get("wrong") == 0
            and row[key].get("errors") == 0
            for key in ("known_disable_all", "known_default")
        )
    ]
    passed = [row for row in rows if row.get("pre_fresh_pass")]
    result = {
        "task": TASK,
        "baseline_cost": BASE_COST,
        "baseline_sha256": BASE_SHA,
        "unique_models": len(rows),
        "strictly_cheaper_actual": [row["sha256"] for row in cheaper],
        "strictly_cheaper_known100_dual": [row["sha256"] for row in known100],
        "pre_fresh_pass": [row["sha256"] for row in passed],
        "rows": rows,
        "complete": True,
    }
    (HERE / "evidence/history_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        f"DONE unique={len(rows)} cheaper={len(cheaper)} "
        f"known100={len(known100)} pre_fresh_pass={len(passed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
