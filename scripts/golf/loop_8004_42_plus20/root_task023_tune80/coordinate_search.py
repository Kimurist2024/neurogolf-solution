#!/usr/bin/env python3
"""Known-constrained integer coordinate search for task023's 36-byte kernel."""

from __future__ import annotations

import argparse
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
HELPER_PATH = HERE / "train_ranker.py"
SPEC = importlib.util.spec_from_file_location("task023_ranker_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import train_ranker helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def success_mask(raw: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Emulate uint8 saturation plus ORT TopK's score/index ordering."""
    score = np.clip(raw, 0, 255).astype(np.int32)
    keys = score * 64 - np.arange(36, dtype=np.int32)[None, :]
    positive_floor = np.where(targets, keys, np.iinfo(np.int32).max).min(axis=1)
    negative_peak = np.where(targets, np.iinfo(np.int32).min, keys).max(axis=1)
    return positive_floor > negative_peak


def scores(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum(
        "nif,f->ni", features.astype(np.int16), weights.astype(np.int16), optimize=True
    ).astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--valid", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--mutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=230_023_810)
    parser.add_argument("--source", type=Path, default=HELPER.SOURCE)
    parser.add_argument("--output", type=Path, default=HERE / "task023_ranker_coordinate.onnx")
    args = parser.parse_args()

    known_cases = HELPER.known_cases()
    train_cases = HELPER.generated_cases(args.train, args.seed)
    valid_cases = HELPER.generated_cases(args.valid, args.seed + 10_000_000)
    known_x, known_y = HELPER.make_dataset(known_cases)
    train_x, train_y = HELPER.make_dataset(train_cases)
    valid_x, valid_y = HELPER.make_dataset(valid_cases)
    known_x = known_x.astype(np.int32)
    train_x = train_x.astype(np.int32)
    valid_x = valid_x.astype(np.int32)

    source_path = args.source.resolve()
    source = onnx.load(source_path)
    current = next(
        numpy_helper.to_array(item).astype(np.int16).reshape(-1)
        for item in source.graph.initializer
        if item.name == "score_W_q"
    )
    known_raw = scores(known_x, current)
    train_raw = scores(train_x, current)
    if not bool(success_mask(known_raw, known_y).all()):
        raise RuntimeError("source does not pass the numpy known hard gate")
    current_right = int(success_mask(train_raw, train_y).sum())
    history: list[dict[str, object]] = [
        {"stage": "source", "train_right": current_right, "train_total": args.train}
    ]

    deltas = (-12, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 12)
    for round_index in range(args.rounds):
        improved = False
        order = np.random.default_rng(args.seed + round_index).permutation(36)
        for feature in order:
            best_delta = 0
            best_right = current_right
            for delta in deltas:
                value = int(current[feature]) + delta
                if value < -127 or value > 127:
                    continue
                proposed_known = known_raw + delta * known_x[:, :, feature]
                if not bool(success_mask(proposed_known, known_y).all()):
                    continue
                proposed_train = train_raw + delta * train_x[:, :, feature]
                right = int(success_mask(proposed_train, train_y).sum())
                if right > best_right:
                    best_right = right
                    best_delta = delta
            if best_delta:
                current[feature] += best_delta
                known_raw += best_delta * known_x[:, :, feature]
                train_raw += best_delta * train_x[:, :, feature]
                current_right = best_right
                improved = True
        history.append(
            {
                "stage": "coordinate",
                "round": round_index,
                "train_right": current_right,
                "train_total": args.train,
                "weights": current.astype(int).tolist(),
            }
        )
        if not improved:
            break

    # Random two-coordinate mutations escape plateaus while the complete known
    # fixture remains an inviolable constraint.
    rng = np.random.default_rng(args.seed + 99)
    for mutation in range(args.mutations):
        count = 2 if mutation < (args.mutations * 4 // 5) else 3
        features = rng.choice(36, size=count, replace=False)
        changes = rng.choice(np.asarray([-4, -3, -2, -1, 1, 2, 3, 4]), size=count)
        values = current[features] + changes
        if bool(((values < -127) | (values > 127)).any()):
            continue
        proposed_known = known_raw.copy()
        proposed_train = train_raw.copy()
        for feature, delta in zip(features, changes, strict=True):
            proposed_known += int(delta) * known_x[:, :, int(feature)]
            proposed_train += int(delta) * train_x[:, :, int(feature)]
        if not bool(success_mask(proposed_known, known_y).all()):
            continue
        right = int(success_mask(proposed_train, train_y).sum())
        if right > current_right:
            current[features] = values
            known_raw = proposed_known
            train_raw = proposed_train
            current_right = right
            history.append(
                {
                    "stage": "mutation",
                    "iteration": mutation,
                    "train_right": current_right,
                    "train_total": args.train,
                }
            )

    valid_raw = scores(valid_x, current)
    valid_right_numpy = int(success_mask(valid_raw, valid_y).sum())
    model = copy.deepcopy(source)
    HELPER.replace_weights(model, current.astype(np.int8))
    output = args.output.resolve()
    onnx.save(model, output)
    known_disabled = HELPER.ort_rate(model, known_cases, "disabled")
    known_default = HELPER.ort_rate(model, known_cases, "default")
    valid_disabled = HELPER.ort_rate(model, valid_cases, "disabled")
    valid_default = HELPER.ort_rate(model, valid_cases, "default")
    result = {
        "source": str(source_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "candidate": str(output.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "seed": args.seed,
        "train_right_numpy": current_right,
        "train_total": args.train,
        "valid_right_numpy": valid_right_numpy,
        "valid_total": args.valid,
        "known_disabled": known_disabled,
        "known_default": known_default,
        "valid_disabled": valid_disabled,
        "valid_default": valid_default,
        "weights": current.astype(int).tolist(),
        "history": history,
        "generator_non_injective": True,
    }
    (HERE / "coordinate_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if known_disabled[0] == known_default[0] == len(known_cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
