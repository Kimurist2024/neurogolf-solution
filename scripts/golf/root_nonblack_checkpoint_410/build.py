#!/usr/bin/env python3
"""Build a black-free POLICY90 checkpoint from the pinned 8012.15 authority.

Outputs are written under ``others/71407/nonblack_policy90_8012_15_wave1``.
The root submission, score table, and authority archives are never modified.
Individual probe ZIPs are emitted before the cumulative ZIP so leaderboard
failures can be attributed without another bisect.
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
OUT = ROOT / "others/71407/nonblack_policy90_8012_15_wave1"
BLACK_EXACT_TASKS = {70, 134, 202, 343}

CANDIDATES = (
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
        "risk": "inherits authority's legacy declared/runtime shape mismatch",
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


def write_zip(path: Path, replacements: dict[str, bytes]) -> dict:
    with zipfile.ZipFile(BASE) as source:
        names = source.namelist()
        original = {name: source.read(name) for name in names}
    if len(names) != 400 or len(set(names)) != 400:
        raise RuntimeError("authority must contain exactly 400 unique members")
    unknown = set(replacements) - set(names)
    if unknown:
        raise RuntimeError(f"replacement members missing from authority: {unknown}")

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
    if changed != [name for name in names if name in replacements]:
        raise RuntimeError(f"unexpected changed members: {changed}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
        "member_count": len(built_names),
        "changed_members": changed,
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
        writer = csv.DictWriter(handle, fieldnames=["rank", "task", "hash", "cost", "score", "archetype"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if sha256(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("8012.15 authority SHA drift")
    tasks = {int(row["task"]) for row in CANDIDATES}
    if tasks & BLACK_EXACT_TASKS:
        raise RuntimeError(f"known-black task admitted: {tasks & BLACK_EXACT_TASKS}")
    if len(tasks) != len(CANDIDATES):
        raise RuntimeError("duplicate task candidate")

    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint: {OUT}")
    (OUT / "candidates").mkdir(parents=True)
    (OUT / "evidence").mkdir()
    (OUT / "probes").mkdir()

    payloads: dict[int, bytes] = {}
    rows = []
    for candidate in CANDIDATES:
        task = int(candidate["task"])
        data = Path(candidate["source"]).read_bytes()
        if sha256(data) != candidate["sha256"]:
            raise RuntimeError(f"candidate SHA drift task{task:03d}")
        payloads[task] = data
        target = OUT / "candidates" / f"task{task:03d}.onnx"
        target.write_bytes(data)
        gain = math.log(candidate["authority_cost"] / candidate["candidate_cost"])
        row = {
            **{key: value for key, value in candidate.items() if key != "source"},
            "source": str(Path(candidate["source"]).relative_to(ROOT)),
            "checkpoint_file": str(target.relative_to(ROOT)),
            "gain": gain,
        }
        rows.append(row)

    shutil.copy2(
        ROOT / "scripts/golf/restart8012_pending_3w_404/evidence.json",
        OUT / "evidence/pending_161_175_355.json",
    )
    shutil.copy2(
        ROOT / "scripts/golf/restart8012_task354_main_407/audit_evidence.json",
        OUT / "evidence/task354.json",
    )

    zips = []
    for task, data in payloads.items():
        zips.append(
            write_zip(
                OUT / "probes" / f"submission_task{task:03d}_ONLY.zip",
                {f"task{task:03d}.onnx": data},
            )
        )
    zips.append(
        write_zip(
            OUT / "submission_NONBLACK_POLICY90_ALL4_NOT_LB_GUARANTEED.zip",
            {f"task{task:03d}.onnx": data for task, data in payloads.items()},
        )
    )

    projected_gain = sum(float(row["gain"]) for row in rows)
    write_projection({int(row["task"]): int(row["candidate_cost"]) for row in rows})
    manifest = {
        "authority": {"path": str(BASE.relative_to(ROOT)), "sha256": BASE_SHA256, "lb": BASE_LB},
        "policy": "Known LB-black exact candidates excluded; otherwise >=90% per config/seed and hard errors zero",
        "known_black_exact_exclusions": sorted(BLACK_EXACT_TASKS),
        "known_black_candidates_present": False,
        "candidates": rows,
        "projected_gain": projected_gain,
        "projected_lb_if_all_hold": BASE_LB + projected_gain,
        "zips": zips,
        "recommended_probe_order": [161, 175, 354, 355],
        "root_authority_modified": False,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Nonblack POLICY90 checkpoint — 8012.15 wave 1",
        "",
        "The four latest LB-black candidates (070@52, 134@320, 202@20, 343@172) are not present.",
        "Root `submission.zip`, `all_scores.csv`, and the 8012.15 authority were not modified.",
        "",
        "| task | cost | minimum accuracy | gain | classification |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']:03d} | {row['authority_cost']}→{row['candidate_cost']} | "
            f"{100 * row['minimum_accuracy']:.2f}% | +{row['gain']:.6f} | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            f"Conditional gain: **+{projected_gain:.6f}** → **{BASE_LB + projected_gain:.6f}**.",
            "",
            "Use the four ZIPs in `probes/` individually first. The cumulative ALL4 ZIP is only a convenience and is not LB-guaranteed.",
            "task354 is exact on the audited known/fresh sets and comes from a prior LB-white lineage, but retains its authority's legacy shape mismatch.",
            "task355 is explicitly marked public-overfit risk.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "gain": projected_gain, "projected": BASE_LB + projected_gain, "zips": zips}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
