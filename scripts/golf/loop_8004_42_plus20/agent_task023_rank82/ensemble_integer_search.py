#!/usr/bin/env python3
"""Known-hard multi-shard integer search for task023's score kernel."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HELPER_PATH = ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/train_ranker.py"
SPEC = importlib.util.spec_from_file_location("task023_rank_helper_int82", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load task023 helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)

SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all/candidates/task023_9a2b78138891_cost1541.onnx"
ROOT_C1 = ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/task023_ranker_coordinate.onnx"
ROOT_C2 = ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/task023_ranker_coordinate2.onnx"
CANDIDATES = HERE / "candidates"


def get_weights(path: Path) -> np.ndarray:
    model = onnx.load(path)
    return next(
        numpy_helper.to_array(item).astype(np.int16).reshape(-1)
        for item in model.graph.initializer
        if item.name == "score_W_q"
    )


def raw_scores(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("nif,f->ni", x.astype(np.int16), w.astype(np.int16), optimize=True).astype(np.int32)


def margins(raw: np.ndarray, target: np.ndarray) -> np.ndarray:
    score = np.clip(raw, 0, 255)
    key = score * 64 - np.arange(36, dtype=np.int32)[None, :]
    positive = np.where(target, key, np.iinfo(np.int32).max).min(axis=1)
    negative = np.where(target, np.iinfo(np.int32).min, key).max(axis=1)
    return positive - negative


def objective(raw: np.ndarray, target: np.ndarray, shard: np.ndarray) -> tuple[int, int, int, int]:
    value = margins(raw, target)
    rights = [int(np.count_nonzero(value[shard == i] > 0)) for i in range(4)]
    # Quantized low-tail margin is a stable secondary signal after exact case
    # count.  It helps leave flat TopK plateaus without sacrificing a shard.
    low_tail = int(np.clip(value, -4096, 1024).sum())
    return min(rights), sum(rights), low_tail, int(value.sum())


def known_ok(raw: np.ndarray, target: np.ndarray) -> bool:
    return bool(np.all(margins(raw, target) > 0))


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    known_cases = HELPER.known_cases()
    known_x, known_y = HELPER.make_dataset(known_cases)
    # Four independently generated shards are interleaved into one tensor.
    shard_cases = [
        HELPER.generated_cases(5_000, seed)
        for seed in (823_830_001, 823_930_001, 824_030_001, 824_130_001)
    ]
    pairs = [HELPER.make_dataset(cases) for cases in shard_cases]
    train_x = np.concatenate([p[0] for p in pairs]).astype(np.int16)
    train_y = np.concatenate([p[1] for p in pairs])
    shard = np.repeat(np.arange(4, dtype=np.int8), 5_000)

    starts = [("clean1541", SOURCE), ("root_c1", ROOT_C1), ("root_c2", ROOT_C2)]
    all_results: list[dict[str, object]] = []
    deltas = (-16, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 16)
    rng = np.random.default_rng(823_154_182)

    for label, path in starts:
        current = get_weights(path)
        known_raw = raw_scores(known_x, current)
        train_raw = raw_scores(train_x, current)
        if not known_ok(known_raw, known_y):
            continue
        current_obj = objective(train_raw, train_y, shard)
        history: list[dict[str, object]] = [
            {"stage": "start", "objective": list(current_obj), "weights": current.astype(int).tolist()}
        ]
        # Choose the globally best legal coordinate move at every step; this
        # differs from order-dependent greedy coordinate descent.  Twelve full
        # scans are enough to cross the incumbent plateau without tuning on an
        # unbounded stream.
        for move_index in range(12):
            best: tuple[tuple[int, int, int, int], int, int, np.ndarray, np.ndarray] | None = None
            for feature in rng.permutation(36):
                f = int(feature)
                for delta in deltas:
                    new_value = int(current[f]) + delta
                    if not -127 <= new_value <= 127:
                        continue
                    kr = known_raw + delta * known_x[:, :, f]
                    if not known_ok(kr, known_y):
                        continue
                    tr = train_raw + delta * train_x[:, :, f]
                    obj = objective(tr, train_y, shard)
                    if obj <= current_obj:
                        continue
                    if best is None or obj > best[0]:
                        best = (obj, f, delta, kr, tr)
            if best is None:
                break
            current_obj, feature, delta, known_raw, train_raw = best
            current[feature] += delta
            history.append(
                {
                    "stage": "global_coordinate",
                    "move": move_index,
                    "feature": feature,
                    "delta": delta,
                    "objective": list(current_obj),
                }
            )

        model = onnx.load(SOURCE)
        HELPER.replace_weights(model, current.astype(np.int8))
        output = CANDIDATES / f"task023_rank82_integer_{label}.onnx"
        onnx.save(model, output)
        all_results.append(
            {
                "label": label,
                "start": str(path.relative_to(ROOT)),
                "path": str(output.relative_to(ROOT)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "objective": list(current_obj),
                "shard_rates": [
                    float(np.mean(margins(train_raw[shard == i], train_y[shard == i]) > 0))
                    for i in range(4)
                ],
                "known_right": int(np.count_nonzero(margins(known_raw, known_y) > 0)),
                "known_total": len(known_x),
                "weights": current.astype(int).tolist(),
                "history": history,
            }
        )
        print(label, all_results[-1]["shard_rates"], current_obj, flush=True)

    all_results.sort(key=lambda row: tuple(row["objective"]), reverse=True)
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "train_seeds": [823_830_001, 823_930_001, 824_030_001, 824_130_001],
        "train_each": 5_000,
        "known_total": len(known_x),
        "results": all_results,
        "generator_non_injective": True,
    }
    (HERE / "integer_search_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all_results else 2


if __name__ == "__main__":
    raise SystemExit(main())
