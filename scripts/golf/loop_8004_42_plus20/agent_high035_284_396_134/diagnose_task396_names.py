#!/usr/bin/env python3
"""Rename duplicate node names in current task396 for a cost-neutral runtime trace."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high134_names_cost",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high134_names_runtime",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def main() -> int:
    model = onnx.load(HERE / "current/task396.onnx")
    original_names = [node.name for node in model.graph.node]
    duplicate_names = sorted(name for name, count in Counter(original_names).items() if name and count > 1)
    seen: dict[str, int] = {}
    renamed = []
    for index, node in enumerate(model.graph.node):
        base = node.name or f"node_{index}"
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        if occurrence:
            old = node.name
            node.name = f"{base}__dup{occurrence}"
            renamed.append({"index": index, "old": old, "new": node.name})
    data = model.SerializeToString()
    outdir = HERE / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "task396_unique_node_names.onnx"
    path.write_bytes(data)
    static = SCAN.structural(copy.deepcopy(model))
    try:
        trace = AUDIT.direct_trace(396, data)
    except Exception as exc:  # noqa: BLE001
        trace = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "source_sha256": hashlib.sha256((HERE / "current/task396.onnx").read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(data).hexdigest(),
        "duplicate_names": duplicate_names,
        "renamed": renamed,
        "semantic_proof": "ONNX node names are non-semantic metadata; inputs, outputs, attributes, and order are unchanged.",
        "cost_change": 0,
        "candidate": False,
        "path": str(path.relative_to(ROOT)),
        "structural": static,
        "runtime_shape_trace": trace,
    }
    (HERE / "task396_name_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"duplicate_names={duplicate_names} renamed={len(renamed)} "
        f"truthful={trace.get('truthful')} mismatches={trace.get('mismatch_count')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
