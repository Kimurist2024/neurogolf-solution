#!/usr/bin/env python3
"""Search every local ZIP lineage for <=half cost51..100 candidates."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import onnx

import history_scan as common


ROOT = common.ROOT
OUT = common.HERE / "zip_history_evidence.json"


def member_task(name: str) -> int | None:
    match = re.search(r"(?:^|/)task[_-]?(\d{3})\.onnx$", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def main() -> int:
    started = time.monotonic()
    if common.sha256(common.AUTHORITY.read_bytes()) != common.AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    costs = common.authority_costs()
    zip_paths = subprocess.check_output(["rg", "--files", "-g", "*.zip"], cwd=ROOT, text=True).splitlines()
    seen = set()
    candidates = []
    malformed = 0
    member_count = 0
    for zindex, relzip in enumerate(zip_paths, 1):
        try:
            archive = zipfile.ZipFile(ROOT / relzip)
        except Exception:
            malformed += 1
            continue
        try:
            for name in archive.namelist():
                task = member_task(name)
                if task not in costs:
                    continue
                member_count += 1
                try:
                    data = archive.read(name)
                    digest = hashlib.sha256(data).hexdigest()
                    key = (task, digest)
                    if key in seen:
                        continue
                    seen.add(key)
                    model = onnx.load_model_from_string(data)
                except Exception:
                    continue
                limit = costs[task] // 2
                pcount = common.params(model)
                lower = common.declared_lower_bound(model)
                if pcount > limit or lower > limit:
                    continue
                safe, reasons = common.structurally_safe(model)
                candidates.append({
                    "task": task, "zip": relzip, "member": name,
                    "sha256": digest, "authority_cost": costs[task],
                    "half_limit": limit, "params": pcount,
                    "declared_lower_bound": lower,
                    "ops": [node.op_type for node in model.graph.node],
                    "catalog_monitored": task in common.PRIVATE_ZERO_OR_UNSOUND,
                    "structurally_safe": safe, "structural_reasons": reasons,
                    "data": data,
                })
        finally:
            archive.close()
        if zindex % 50 == 0:
            print(json.dumps({"zips": zindex, "members": member_count,
                              "unique": len(seen), "candidates": len(candidates)}), flush=True)

    results = []
    for index, row in enumerate(candidates, 1):
        data = row.pop("data")
        item = dict(row)
        task = int(item["task"])
        if item["catalog_monitored"] or not item["structurally_safe"]:
            item.update({"known_exact": False, "checked": 0, "reject": "catalog_or_structure", "profile": None})
        else:
            model = onnx.load_model_from_string(data)
            exact, checked, reject = common.exact_known(model, task)
            item.update({"known_exact": exact, "checked": checked, "reject": reject})
            if exact:
                with tempfile.TemporaryDirectory(prefix=f"zip303_{task:03d}_", dir="/tmp") as tmp:
                    item["profile"] = common.scoring.score_and_verify(
                        model, task, tmp, label="zip_history", require_correct=False
                    )
            else:
                item["profile"] = None
        profile = item["profile"]
        item["winner"] = bool(
            item["known_exact"] and profile and profile["correct"]
            and int(profile["cost"]) <= int(item["half_limit"])
        )
        results.append(item)
        if index % 20 == 0 or item["winner"]:
            print(json.dumps({"i": index, "n": len(candidates), "task": task,
                              "exact": item["known_exact"], "winner": item["winner"]}), flush=True)

    winners = [row for row in results if row["winner"]]
    payload = {
        "authority": common.AUTHORITY.name, "authority_sha256": common.AUTHORITY_SHA256,
        "scope": "all local ZIP members, cost51..100, <=half target",
        "zip_count": len(zip_paths), "malformed_zip_count": malformed,
        "target_member_count": member_count, "unique_task_sha_count": len(seen),
        "candidate_count": len(candidates), "winner_count": len(winners),
        "winners": winners, "results": results,
        "elapsed_seconds": time.monotonic() - started, "protected_writes": "none",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "winners": winners,
                      "evidence": str(OUT.relative_to(ROOT))}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
