#!/usr/bin/env python3
"""Verify every inventory source for the selected A18 candidate bytes."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source(source: str) -> bytes:
    if "::" not in source:
        return (ROOT / source).read_bytes()
    archive_name, member = source.split("::", 1)
    with zipfile.ZipFile(ROOT / archive_name) as archive:
        return archive.read(member)


def main() -> None:
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows: dict[str, object] = {}
    for label, item in manifest["inventory_entries"].items():
        source_rows = []
        for source in item["sources"]:
            row: dict[str, object] = {
                "source": source,
                "quarantine_named": "quarantine" in source.lower(),
                "private_zero_named": "private0" in source.lower(),
            }
            try:
                source_sha = digest(read_source(source))
                row.update(
                    exists=True,
                    sha256=source_sha,
                    byte_identical=source_sha == item["sha256"],
                )
            except Exception as exc:  # noqa: BLE001
                row.update(
                    exists=False,
                    byte_identical=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            source_rows.append(row)
        rows[label] = {
            "candidate_sha256": item["sha256"],
            "inventory_source_count": item["source_count"],
            "listed_source_count": len(item["sources"]),
            "source_list_complete": len(item["sources"]) == item["source_count"],
            "all_listed_sources_resolved": all(row["exists"] for row in source_rows),
            "all_listed_sources_byte_identical": all(
                row["byte_identical"] for row in source_rows
            ),
            "quarantine_or_private_zero_provenance": any(
                row["quarantine_named"] or row["private_zero_named"]
                for row in source_rows
            ),
            "sources": source_rows,
        }
    (HERE / "lineage_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    for label, row in rows.items():
        print(
            label,
            row["listed_source_count"],
            row["all_listed_sources_resolved"],
            row["all_listed_sources_byte_identical"],
            "quarantine=" + str(row["quarantine_or_private_zero_provenance"]),
        )


if __name__ == "__main__":
    main()
