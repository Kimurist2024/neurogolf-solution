#!/usr/bin/env python3
"""Package the cost<=100 POLICY95 candidates without touching root authority files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "others" / "71407" / "cost_le100_8011_05"
BASE_ZIP = ROOT / "submission_base_8011.05.zip"
BASE_LB = 8011.05
BASE_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"

CANDIDATES = {
    70: {
        "source": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task070_r02_static52.onnx",
        "evidence": ROOT / "scripts/golf/half_cost_51_100_303/task070_policy95_cost52_audit.json",
        "authority_cost": 66,
        "candidate_cost": 52,
        "fresh_rates": [0.99, 0.9845],
        "classification": "POLICY95_PRIVATE_ZERO_RISK",
    },
    202: {
        "source": ROOT / "scripts/golf/half_cost_2_50_304/candidates/task202_policy95_arity14_cost20.onnx",
        "evidence": ROOT / "scripts/golf/half_cost_2_50_304/task202_arity14_evidence.json",
        "authority_cost": 48,
        "candidate_cost": 20,
        "fresh_rates": [0.974, 0.9665],
        "classification": "POLICY95_PRIVATE_ZERO_LINEAGE_NON_GIANT",
    },
}

SEARCH_REPORTS = {
    "cost02_50_REPORT.md": ROOT / "scripts/golf/half_cost_2_50_304/REPORT.md",
    "cost51_100_REPORT.md": ROOT / "scripts/golf/half_cost_51_100_303/REPORT.md",
    "score25_similarity_REPORT.md": ROOT / "scripts/golf/score25_similarity_le100_304/REPORT.md",
    "uniform_output_analysis.json": ROOT / "scripts/golf/score25_similarity_le100_304/uniform_output_analysis.json",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_submission_zip(path: Path) -> None:
    replacements = {
        f"task{task:03d}.onnx": data["source"].read_bytes()
        for task, data in CANDIDATES.items()
    }
    with zipfile.ZipFile(BASE_ZIP, "r") as zin, zipfile.ZipFile(path, "w") as zout:
        for info in zin.infolist():
            data = replacements.get(info.filename, zin.read(info.filename))
            zout.writestr(info, data)


def build_score_csv(path: Path) -> None:
    rows = list(csv.DictReader((ROOT / "all_scores.csv").open()))
    fields = list(rows[0])
    for row in rows:
        task = int(row["task"][4:])
        if task not in CANDIDATES:
            continue
        cost = CANDIDATES[task]["candidate_cost"]
        row["cost"] = str(cost)
        row["score"] = f"{max(1.0, 25.0 - math.log(cost)):.4f}"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if digest(BASE_ZIP) != BASE_SHA256:
        raise SystemExit("authority zip SHA256 mismatch")
    candidates_dir = OUT / "POLICY95_PRIVATE_ZERO_RISK"
    evidence_dir = OUT / "evidence"
    reports_dir = OUT / "search_reports"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name, source in SEARCH_REPORTS.items():
        if not source.is_file():
            raise SystemExit(f"missing search report: {source}")
        shutil.copy2(source, reports_dir / name)

    manifest_candidates = []
    for task, data in sorted(CANDIDATES.items()):
        if not data["source"].is_file() or not data["evidence"].is_file():
            raise SystemExit(f"missing source/evidence for task{task:03d}")
        dst = candidates_dir / f"task{task:03d}.onnx"
        shutil.copy2(data["source"], dst)
        shutil.copy2(data["evidence"], evidence_dir / f"task{task:03d}_audit.json")
        score_gain = math.log(data["authority_cost"] / data["candidate_cost"])
        manifest_candidates.append(
            {
                "task": task,
                "file": str(dst.relative_to(OUT)),
                "sha256": digest(dst),
                "authority_cost": data["authority_cost"],
                "candidate_cost": data["candidate_cost"],
                "half_target": data["candidate_cost"] <= data["authority_cost"] / 2,
                "score_gain": score_gain,
                "fresh_rates_two_seeds": data["fresh_rates"],
                "fresh_threshold": 0.95,
                "known_gold_rate": 1.0,
                "ort_configs": 4,
                "runtime_errors": 0,
                "nonfinite_cases": 0,
                "small_positive_elements": 0,
                "classification": data["classification"],
                "guaranteed_safe": False,
                "evidence": f"evidence/task{task:03d}_audit.json",
            }
        )

    zip_path = OUT / "submission_POLICY95_NOT_LB_GUARANTEED.zip"
    build_submission_zip(zip_path)
    build_score_csv(OUT / "all_scores_POLICY95_projection.csv")

    with zipfile.ZipFile(BASE_ZIP) as base, zipfile.ZipFile(zip_path) as patched:
        base_names = base.namelist()
        patched_names = patched.namelist()
        if base_names != patched_names:
            raise SystemExit("submission member order changed")
        changed = []
        for name in base_names:
            if hashlib.sha256(base.read(name)).digest() != hashlib.sha256(patched.read(name)).digest():
                changed.append(name)
        expected = [f"task{task:03d}.onnx" for task in sorted(CANDIDATES)]
        if changed != expected:
            raise SystemExit(f"unexpected changed members: {changed}")

    total_gain = sum(item["score_gain"] for item in manifest_candidates)
    manifest = {
        "authority": {
            "lb": BASE_LB,
            "zip": BASE_ZIP.name,
            "sha256": BASE_SHA256,
            "md5": digest(BASE_ZIP, "md5"),
        },
        "scope": {
            "authority_tasks_at_cost_le_100": 169,
            "existing_score25_tasks": [67, 129, 179, 241],
            "eligible_non_score25_tasks": 165,
            "output_path_note": "The duplicated absolute path in the request was treated as a paste duplication; this bundle is under the existing others/71407 directory.",
        },
        "acceptance_policy": {
            "fresh_minimum": 0.95,
            "known_private_zero_lineage_allowed": True,
            "runtime_errors_forbidden": True,
            "undefined_behavior_forbidden": True,
            "nonfinite_forbidden": True,
            "shape_cloak_forbidden": True,
            "small_positive_margin_forbidden": True,
        },
        "guaranteed_safe_candidates": [],
        "policy95_candidates": manifest_candidates,
        "search_reports": [f"search_reports/{name}" for name in SEARCH_REPORTS],
        "projected": {
            "score_gain_if_both_receive_credit": total_gain,
            "lb_from_8011_05_if_both_receive_credit": BASE_LB + total_gain,
            "cost_reduction": sum(
                item["authority_cost"] - item["candidate_cost"]
                for item in manifest_candidates
            ),
            "warning": "Leaderboard scoring is all-or-nothing per task. These known private-zero lineages are not guaranteed to receive credit despite >=95% fresh accuracy.",
        },
        "submission_patch": {
            "file": zip_path.name,
            "sha256": digest(zip_path),
            "md5": digest(zip_path, "md5"),
            "member_count": len(zipfile.ZipFile(zip_path).namelist()),
            "changed_members_only": expected,
            "member_order_preserved": True,
            "root_submission_modified": False,
        },
        "key_rejections": [
            {
                "tasks": [322, 372],
                "reason": "Lower histories use short ConvTranspose bias / undefined behavior.",
            },
            {
                "task": 70,
                "candidate_cost": 50,
                "reason": "Fresh small-positive raw outputs in (0, 0.25); cost52 is the stable-margin choice.",
            },
            {
                "task": 135,
                "candidate_cost": 1,
                "reason": "Finite one-parameter ConvTranspose crop attempt failed gold and fresh; output padding does not mask the unwanted rows.",
            },
        ],
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
