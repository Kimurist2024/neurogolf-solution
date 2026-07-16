#!/usr/bin/env python3
"""Build the second 8012.15 checkpoint without touching root authority files.

The bundle combines separately audited candidates, emits one-task probe ZIPs,
and checks that the four latest leaderboard-black members remain byte-identical
to the pinned 8012.15 authority.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8012.15.zip"
BASE_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
BASE_LB = 8012.15
OUT = ROOT / "others/71407/nonblack_policy90_8012_15_wave2"
BLACK_EXACT_TASKS = {70, 134, 202, 343}

CANDIDATES = (
    {
        "task": 23,
        "source": ROOT / "scripts/golf/restart8012_dedup_main_412/task023_cost1317_exact.onnx",
        "sha256": "c9725627bc4aaa49494c4da4ff6c06849e38cd4fffd0c0fd3c64afdf5ce1472c",
        "authority_cost": 1321,
        "candidate_cost": 1317,
        "classification": "EXACT_AUTHORITY_EQUIVALENT",
        "minimum_accuracy": None,
        "risk": "raw-identical to the verified authority; ordinary POLICY90 does not apply",
    },
    {
        "task": 12,
        "source": ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/candidates/task012_POLICY90_cost650_9aea31a6c01f.onnx",
        "sha256": "9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947",
        "authority_cost": 710,
        "candidate_cost": 650,
        "classification": "NONBLACK_POLICY90",
        "minimum_accuracy": 0.9455,
        "risk": "nonexact; truthful standard Conv; no maintained private-zero hit",
    },
    {
        "task": 161,
        "source": ROOT / "scripts/golf/restart8012_pending_3w_404/candidates/task161_cost186_57487cce1b40_POLICY95_NONEXACT.onnx",
        "sha256": "57487cce1b40cc7df6097cdf1e82e7bfa53b9bcb6f5be954329ea10d132ced81",
        "authority_cost": 190,
        "candidate_cost": 186,
        "classification": "NONBLACK_POLICY90",
        "minimum_accuracy": 0.9935,
        "risk": "nonexact; no maintained private-zero hit",
    },
    {
        "task": 175,
        "source": ROOT / "scripts/golf/restart8012_pending_3w_404/candidates/task175_cost145_40a940588083_POLICY95_NONEXACT.onnx",
        "sha256": "40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c",
        "authority_cost": 166,
        "candidate_cost": 145,
        "classification": "NONBLACK_POLICY90",
        "minimum_accuracy": 0.9849624060150376,
        "risk": "nonexact; no maintained private-zero hit",
    },
    {
        "task": 354,
        "source": ROOT / "scripts/golf/restart8012_task354_main_407/candidates/task354_combined.onnx",
        "sha256": "c45a5760c95e9bd22268f927e2f98cda8195c5d87e9dcca533977574b58b3a75",
        "authority_cost": 497,
        "candidate_cost": 461,
        "classification": "KNOWN_LB_WHITE_LINEAGE_EXACT_FRESH",
        "minimum_accuracy": 1.0,
        "risk": "inherits the authority's legacy declared/runtime output-shape mismatch",
    },
    {
        "task": 355,
        "source": ROOT / "scripts/golf/restart8012_pending_3w_404/candidates/task355_cost249_7ca617858a19_PUBLIC_OVERFIT_RISK_POLICY95.onnx",
        "sha256": "7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8",
        "authority_cost": 250,
        "candidate_cost": 249,
        "classification": "NONBLACK_POLICY90_PUBLIC_OVERFIT_RISK",
        "minimum_accuracy": 0.985,
        "risk": "public-overfit watchlist; not a maintained LB-black hit",
    },
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authority_members() -> tuple[list[str], dict[str, bytes]]:
    with zipfile.ZipFile(BASE) as source:
        names = source.namelist()
        original = {name: source.read(name) for name in names}
    if len(names) != 400 or len(set(names)) != 400:
        raise RuntimeError("authority must contain exactly 400 unique members")
    return names, original


def write_zip(
    path: Path,
    replacements: dict[str, bytes],
    names: list[str],
    original: dict[str, bytes],
) -> dict:
    unknown = set(replacements) - set(names)
    if unknown:
        raise RuntimeError(f"replacement members missing from authority: {sorted(unknown)}")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            out.writestr(info, replacements.get(name, original[name]))

    with zipfile.ZipFile(path) as built:
        built_names = built.namelist()
        built_data = {name: built.read(name) for name in built_names}
    if built_names != names:
        raise RuntimeError("ZIP member order drifted")
    changed = [name for name in names if built_data[name] != original[name]]
    expected_changed = [name for name in names if name in replacements]
    if changed != expected_changed:
        raise RuntimeError(f"unexpected changed members: {changed}")
    black_unchanged = {
        f"task{task:03d}.onnx": built_data[f"task{task:03d}.onnx"]
        == original[f"task{task:03d}.onnx"]
        for task in sorted(BLACK_EXACT_TASKS)
    }
    if not all(black_unchanged.values()):
        raise RuntimeError(f"known-black authority member drift: {black_unchanged}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
        "member_count": len(built_names),
        "changed_members": changed,
        "known_black_members_byte_identical": black_unchanged,
    }


def write_projection(costs: dict[int, int]) -> None:
    rows = []
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            if task in costs:
                cost = costs[task]
                row["cost"] = str(cost)
                row["score"] = f"{max(1.0, 25.0 - math.log(max(1.0, cost))):.4f}"
            rows.append(row)
    rows.sort(key=lambda row: (float(row["score"]), row["task"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = str(rank)
    with (OUT / "all_scores_POLICY90_projection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "task", "hash", "cost", "score", "archetype"],
        )
        writer.writeheader()
        writer.writerows(rows)


def copy_evidence() -> None:
    evidence = OUT / "evidence"
    copies = (
        (ROOT / "scripts/golf/restart8012_dedup_main_412/evidence.json", evidence / "task023_exact.json"),
        (ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker_1.json", evidence / "task012_current_audit.json"),
        (ROOT / "scripts/golf/root_task012_h8w8_policy90_272/evidence.json", evidence / "task012_primary.json"),
        (ROOT / "scripts/golf/agent_review_task012_h8w8_policy90_273/evidence.json", evidence / "task012_independent.json"),
        (ROOT / "scripts/golf/restart8012_pending_3w_404/evidence.json", evidence / "tasks161_175_355.json"),
        (ROOT / "scripts/golf/restart8012_task354_main_407/audit_evidence.json", evidence / "task354.json"),
    )
    for source, destination in copies:
        if not source.is_file():
            raise RuntimeError(f"missing evidence: {source}")
        shutil.copy2(source, destination)


def main() -> int:
    if sha256(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("8012.15 authority SHA drift")
    tasks = {int(row["task"]) for row in CANDIDATES}
    if tasks & BLACK_EXACT_TASKS:
        raise RuntimeError(f"known-black task admitted: {sorted(tasks & BLACK_EXACT_TASKS)}")
    if len(tasks) != len(CANDIDATES):
        raise RuntimeError("duplicate task candidate")
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint: {OUT}")

    (OUT / "candidates").mkdir(parents=True)
    (OUT / "evidence").mkdir()
    (OUT / "probes").mkdir()
    names, original = authority_members()

    payloads: dict[int, bytes] = {}
    rows = []
    for candidate in CANDIDATES:
        task = int(candidate["task"])
        source = Path(candidate["source"])
        data = source.read_bytes()
        if sha256(data) != candidate["sha256"]:
            raise RuntimeError(f"candidate SHA drift task{task:03d}")
        payloads[task] = data
        target = OUT / "candidates" / f"task{task:03d}.onnx"
        target.write_bytes(data)
        rows.append(
            {
                **{key: value for key, value in candidate.items() if key != "source"},
                "source": str(source.relative_to(ROOT)),
                "checkpoint_file": str(target.relative_to(ROOT)),
                "gain": math.log(candidate["authority_cost"] / candidate["candidate_cost"]),
            }
        )

    copy_evidence()
    zips = []
    for task, data in payloads.items():
        zips.append(
            write_zip(
                OUT / "probes" / f"submission_task{task:03d}_ONLY.zip",
                {f"task{task:03d}.onnx": data},
                names,
                original,
            )
        )
    zips.append(
        write_zip(
            OUT / "submission_NONBLACK_POLICY90_ALL6_NOT_LB_GUARANTEED.zip",
            {f"task{task:03d}.onnx": data for task, data in payloads.items()},
            names,
            original,
        )
    )

    projected_gain = sum(float(row["gain"]) for row in rows)
    write_projection({int(row["task"]): int(row["candidate_cost"]) for row in rows})
    manifest = {
        "authority": {"path": str(BASE.relative_to(ROOT)), "sha256": BASE_SHA256, "lb": BASE_LB},
        "policy": (
            "Known LB-black exact candidates excluded; approximate candidates require >=90% "
            "per config/seed and zero hard errors; exact-authority-equivalent rewrites are separate"
        ),
        "known_black_exact_exclusions": sorted(BLACK_EXACT_TASKS),
        "known_black_candidates_present": False,
        "candidates": rows,
        "projected_gain": projected_gain,
        "projected_lb_if_all_hold": BASE_LB + projected_gain,
        "zips": zips,
        "recommended_probe_order": [23, 175, 161, 12, 354, 355],
        "root_authority_modified": False,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Nonblack POLICY90 checkpoint — 8012.15 wave 2",
        "",
        "The latest LB-black candidates 070@52, 134@320, 202@20, and 343@172 are absent.",
        "Every generated ZIP keeps those four authority members byte-identical.",
        "Root `submission.zip`, `all_scores.csv`, and the 8012.15 authority are unchanged.",
        "",
        "| task | cost | minimum audited accuracy | gain | classification |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        accuracy = "authority raw-identical" if row["minimum_accuracy"] is None else f"{100 * row['minimum_accuracy']:.2f}%"
        lines.append(
            f"| {row['task']:03d} | {row['authority_cost']}→{row['candidate_cost']} | "
            f"{accuracy} | +{row['gain']:.6f} | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            f"Conditional gain: **+{projected_gain:.6f}** → **{BASE_LB + projected_gain:.6f}**.",
            "",
            "Probe the six ZIPs in `probes/` one at a time before using the cumulative ALL6 ZIP.",
            "task023 is an exact authority-equivalent rewrite, not an approximate POLICY90 model.",
            "task354 retains its verified authority lineage's legacy declared/runtime shape mismatch.",
            "task355 remains explicitly labeled as public-overfit risk and should be probed last.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT.relative_to(ROOT)),
                "gain": projected_gain,
                "projected": BASE_LB + projected_gain,
                "zips": zips,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
