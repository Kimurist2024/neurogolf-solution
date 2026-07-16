#!/usr/bin/env python3
"""Independent structural, cost, hash, and archive verification for wave 2."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKPOINT = ROOT / "others/71407/nonblack_policy90_8012_15_wave2"
BASE = ROOT / "submission_base_8012.15.zip"
ROOT_GUARDS = {
    "submission.zip": "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231",
    "submission_base_8012.15.zip": "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231",
    "all_scores.csv": "3f9914a0db88302f9e0424d604f9c0e300dc75115570625d296e21b7fcfaf731",
}
BLACK_TASKS = (70, 134, 202, 343)

sys.path.insert(0, str(ROOT / "scripts/golf"))
from rank_dir import cost_of  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((CHECKPOINT / "MANIFEST.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    rows = []
    guards = {}
    for name, expected in ROOT_GUARDS.items():
        actual = sha256(ROOT / name)
        guards[name] = actual
        if actual != expected:
            errors.append(f"root guard drift: {name}")

    with zipfile.ZipFile(BASE) as authority, tempfile.TemporaryDirectory(prefix="wave2_verify_") as tmp:
        authority_names = authority.namelist()
        cumulative_row = manifest["zips"][-1]
        cumulative_path = ROOT / cumulative_row["path"]
        if sha256(cumulative_path) != cumulative_row["sha256"]:
            errors.append("cumulative ZIP SHA mismatch")
        with zipfile.ZipFile(cumulative_path) as cumulative:
            if cumulative.namelist() != authority_names or len(authority_names) != 400:
                errors.append("cumulative ZIP member set/order mismatch")
            for task in BLACK_TASKS:
                member = f"task{task:03d}.onnx"
                if cumulative.read(member) != authority.read(member):
                    errors.append(f"known-black member drift: {member}")

        for candidate in manifest["candidates"]:
            task = int(candidate["task"])
            path = ROOT / candidate["checkpoint_file"]
            if sha256(path) != candidate["sha256"]:
                errors.append(f"candidate SHA mismatch: task{task:03d}")
                continue
            model = onnx.load(path)
            try:
                onnx.checker.check_model(model, full_check=True)
                onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
                structure_pass = True
                structure_error = None
            except Exception as exc:  # fail closed and preserve diagnostic
                structure_pass = False
                structure_error = f"{type(exc).__name__}: {exc}"
                errors.append(f"structure failure: task{task:03d}: {structure_error}")

            authority_path = Path(tmp) / f"task{task:03d}.onnx"
            authority_path.write_bytes(authority.read(f"task{task:03d}.onnx"))
            authority_profile = cost_of(str(authority_path))
            candidate_profile = cost_of(str(path))
            declared_delta = int(candidate["authority_cost"]) - int(candidate["candidate_cost"])
            diagnostic_delta = authority_profile[2] - candidate_profile[2]
            cost_pass = (
                candidate_profile[2] == int(candidate["candidate_cost"])
                or diagnostic_delta == declared_delta
            )
            if not cost_pass:
                errors.append(
                    f"cost mismatch task{task:03d}: candidate={candidate_profile[2]}, "
                    f"diagnostic_delta={diagnostic_delta}, declared_delta={declared_delta}"
                )
            rows.append(
                {
                    "task": task,
                    "sha256": candidate["sha256"],
                    "structure_pass": structure_pass,
                    "structure_error": structure_error,
                    "authority_profile": {
                        "memory": authority_profile[0],
                        "params": authority_profile[1],
                        "cost": authority_profile[2],
                    },
                    "candidate_profile": {
                        "memory": candidate_profile[0],
                        "params": candidate_profile[1],
                        "cost": candidate_profile[2],
                    },
                    "declared_cost": int(candidate["candidate_cost"]),
                    "declared_delta": declared_delta,
                    "diagnostic_delta": diagnostic_delta,
                    "cost_pass": cost_pass,
                }
            )

    result = {
        "pass": not errors,
        "errors": errors,
        "root_guards": guards,
        "known_black_members_byte_identical": not any("known-black" in e for e in errors),
        "cumulative_member_count": len(authority_names),
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    (HERE / "verification.json").write_text(rendered, encoding="utf-8")
    (CHECKPOINT / "evidence/checkpoint_verification.json").write_text(
        rendered, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
