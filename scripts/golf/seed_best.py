#!/usr/bin/env python3
"""Seed artifacts/handcrafted/task<NNN>.onnx with the current BEST version so
focus workers compete against the true baseline -- UNLESS the existing
handcrafted is already a cheaper, correct net (a pending gain), in which case
keep it. Usage: seed_best.py <task>
"""
from __future__ import annotations
import os, sys, tempfile, zipfile
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402


def cost(model, t):
    with tempfile.TemporaryDirectory() as wd:
        s = scoring.score_and_verify(model, t, wd, label="x", require_correct=True)
    return s["cost"] if s else None


def best_zip_path() -> Path:
    """Highest-scoring zip listed in campaign_best.txt.

    The file is append-only, so its first line is the OLDEST base. Seeding from
    that hands workers a stale, more expensive incumbent and lets try_candidate
    promote nets that are regressions against the real champion.
    """
    override = os.environ.get("SEED_BEST_ZIP")
    if override:
        return Path(override)
    entries: list[tuple[float, str]] = []
    for line in (REPO / "docs/golf/campaign_best.txt").read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            entries.append((float(parts[1]), parts[0]))
    if not entries:
        raise SystemExit("campaign_best.txt has no '<zip>\\t<score>' entries")
    name = max(entries)[1]
    resolved = REPO / name
    return resolved if resolved.is_file() else Path(name)


def main():
    t = int(sys.argv[1])
    best_zip = best_zip_path()
    bmodel = onnx.load_model_from_string(zipfile.ZipFile(best_zip).read(f"task{t:03d}.onnx"))
    bc = cost(bmodel, t)
    hp = REPO / "artifacts" / "handcrafted" / f"task{t:03d}.onnx"
    if hp.is_file() and bc is not None:
        hc = cost(onnx.load(str(hp)), t)
        if hc is not None and hc < bc:
            print(f"keep handcrafted (pending gain {hc} < best {bc})")
            return
    onnx.save(bmodel, str(hp))
    print(f"seeded task{t:03d} handcrafted from {best_zip.name} (cost {bc})")


if __name__ == "__main__":
    main()
