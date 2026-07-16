#!/usr/bin/env python3
"""Exact integer coordinate refinement for task023 spatial morphology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx

import search


HERE = Path(__file__).resolve().parent


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(search.ROOT))
    except ValueError:
        return str(resolved)


def score_gap(raw: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.clip(raw, 0, 255).astype(np.int32)
    keys = scores * 64 - np.arange(36, dtype=np.int32)[None, :]
    positive = np.where(targets, keys, np.iinfo(np.int32).max).min(axis=1)
    negative = np.where(targets, np.iinfo(np.int32).min, keys).max(axis=1)
    gap = positive - negative
    return gap > 0, gap


def metric(
    known_raw: np.ndarray,
    known_y: np.ndarray,
    train_raw: np.ndarray,
    train_y: np.ndarray,
) -> tuple[int, int, int, int, int]:
    known_ok, known_gap = score_gap(known_raw, known_y)
    train_ok, train_gap = score_gap(train_raw, train_y)
    known_right = int(known_ok.sum())
    train_right = int(train_ok.sum())
    known_deficit = int(np.minimum(known_gap, 0).sum())
    if known_right == len(known_y):
        return (
            known_right,
            train_right,
            int(known_gap.min()),
            int(np.clip(train_gap, -4096, 4096).sum()),
            int(known_gap.sum()),
        )
    return (
        known_right,
        known_deficit,
        train_right,
        int(np.clip(train_gap, -4096, 4096).sum()),
        int(known_gap.sum()),
    )


def make_state(
    patches1: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    layout: search.Layout,
) -> dict[str, np.ndarray]:
    pre = np.einsum(
        "nif,cf->nci", patches1.astype(np.int32), w1.astype(np.int32), optimize=True
    )
    hidden = np.clip(pre, 0, 255).astype(np.int16)
    p2 = search.second_patches(hidden.reshape(len(hidden), 2, 6, 6), layout)
    raw = np.einsum(
        "ncrsij,cij->nrs", p2.astype(np.int32), w2.astype(np.int32), optimize=True
    ).reshape(len(hidden), 36)
    return {"pre": pre, "hidden": hidden, "raw": raw}


def w1_proposal(
    state: dict[str, np.ndarray],
    patches1: np.ndarray,
    w2: np.ndarray,
    layout: search.Layout,
    channel: int,
    feature: int,
    delta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proposed_pre = state["pre"][:, channel, :] + delta * patches1[:, :, feature]
    proposed_hidden = np.clip(proposed_pre, 0, 255).astype(np.int16)
    change = proposed_hidden.astype(np.int32) - state["hidden"][:, channel, :].astype(np.int32)
    change = change.reshape(len(change), 1, 6, 6)
    top, left, bottom, right = layout.pads2
    padded = np.pad(change, ((0, 0), (0, 0), (top, bottom), (left, right)))
    view = np.lib.stride_tricks.sliding_window_view(
        padded, layout.kernel2, axis=(2, 3)
    )
    contribution = np.einsum(
        "ncrsij,ij->nrs", view, w2[channel].astype(np.int32), optimize=True
    ).reshape(len(change), 36)
    return state["raw"] + contribution, proposed_pre, proposed_hidden


def w2_feature(
    state: dict[str, np.ndarray], layout: search.Layout, channel: int, index: int
) -> np.ndarray:
    kh, kw = layout.kernel2
    row, col = divmod(index, kw)
    p2 = search.second_patches(
        state["hidden"].reshape(len(state["hidden"]), 2, 6, 6), layout
    )
    return p2[:, channel, :, :, row, col].reshape(len(p2), 36).astype(np.int32)


def refine(
    candidate: dict,
    layout: search.Layout,
    known_patches: np.ndarray,
    known_y: np.ndarray,
    train_patches: np.ndarray,
    train_y: np.ndarray,
    *,
    rounds: int,
    seed: int,
) -> dict:
    w1 = np.asarray(candidate["w1"], dtype=np.int16).reshape(2, 16)
    w2 = np.asarray(candidate["w2"], dtype=np.int16).reshape(2, *layout.kernel2)
    ks = make_state(known_patches, w1, w2, layout)
    ts = make_state(train_patches, w1, w2, layout)
    current = metric(ks["raw"], known_y, ts["raw"], train_y)
    history = [{"round": -1, "metric": list(current), "w1": w1.astype(int).tolist(), "w2": w2.astype(int).reshape(2, -1).tolist()}]
    rng = np.random.default_rng(seed)
    deltas = (-12, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 12)
    n2 = int(np.prod(layout.kernel2))

    for round_index in range(rounds):
        improved = False
        order = rng.permutation(32 + 2 * n2)
        for encoded in order:
            best = current
            best_update = None
            if encoded < 32:
                channel, feature = divmod(int(encoded), 16)
                for delta in deltas:
                    value = int(w1[channel, feature]) + delta
                    if not -127 <= value <= 127:
                        continue
                    kr, kp, kh = w1_proposal(
                        ks, known_patches, w2, layout, channel, feature, delta
                    )
                    tr, tp, th = w1_proposal(
                        ts, train_patches, w2, layout, channel, feature, delta
                    )
                    proposed = metric(kr, known_y, tr, train_y)
                    if proposed > best:
                        best = proposed
                        best_update = (delta, kr, kp, kh, tr, tp, th)
                if best_update is not None:
                    delta, kr, kp, kh, tr, tp, th = best_update
                    w1[channel, feature] += delta
                    ks["raw"] = kr
                    ks["pre"][:, channel, :] = kp
                    ks["hidden"][:, channel, :] = kh
                    ts["raw"] = tr
                    ts["pre"][:, channel, :] = tp
                    ts["hidden"][:, channel, :] = th
            else:
                offset = int(encoded) - 32
                channel, index = divmod(offset, n2)
                row, col = divmod(index, layout.kernel2[1])
                kfeature = w2_feature(ks, layout, channel, index)
                tfeature = w2_feature(ts, layout, channel, index)
                for delta in deltas:
                    value = int(w2[channel, row, col]) + delta
                    if not -127 <= value <= 127:
                        continue
                    kr = ks["raw"] + delta * kfeature
                    tr = ts["raw"] + delta * tfeature
                    proposed = metric(kr, known_y, tr, train_y)
                    if proposed > best:
                        best = proposed
                        best_update = (delta, kr, tr)
                if best_update is not None:
                    delta, kr, tr = best_update
                    w2[channel, row, col] += delta
                    ks["raw"] = kr
                    ts["raw"] = tr
            if best_update is not None:
                current = best
                improved = True
        history.append(
            {
                "round": round_index,
                "metric": list(current),
                "w1": w1.astype(int).tolist(),
                "w2": w2.astype(int).reshape(2, -1).tolist(),
            }
        )
        print(json.dumps({"layout": layout.label, "round": round_index, "metric": current}), flush=True)
        if not improved:
            break
    return {
        "layout": layout.label,
        "start_known": candidate["known_right"],
        "start_valid": candidate["valid_right"],
        "metric": list(current),
        "w1": w1.astype(int).tolist(),
        "w2": w2.astype(int).reshape(2, -1).tolist(),
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=HERE / "screen.json")
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=246123456)
    parser.add_argument("--output", type=Path, default=HERE / "refine.json")
    parser.add_argument("--save-best", type=Path, default=HERE / "task023_spatial_morphology.onnx")
    args = parser.parse_args()

    screen = json.loads(args.screen.read_text())
    starts = []
    seen = set()
    pool = []
    for record in screen["records"]:
        pool.extend([record["best_lexicographic"], record["best_valid"]])
    pool.sort(key=lambda row: (row["known_right"], row["valid_right"]), reverse=True)
    # Retain the most known-compatible starts plus several highest-fresh starts.
    selected = pool[: args.top]
    selected += sorted(pool, key=lambda row: row["valid_right"], reverse=True)[: max(2, args.top // 2)]
    for row in selected:
        key = row["layout"] + json.dumps(row["w1"]) + json.dumps(row["w2"])
        if key not in seen:
            seen.add(key)
            starts.append(row)

    known_cases = search.known_cases()
    train_cases = search.generated_cases(args.train, args.seed)
    known_x, known_y = search.dataset(known_cases)
    train_x, train_y = search.dataset(train_cases)
    layout_map = {item.label: item for item in search.layouts()}
    results = []
    for index, start in enumerate(starts):
        layout = layout_map[start["layout"]]
        result = refine(
            start,
            layout,
            search.stage1_patches(known_x, layout),
            known_y,
            search.stage1_patches(train_x, layout),
            train_y,
            rounds=args.rounds,
            seed=args.seed + index * 1009,
        )
        results.append(result)

    results.sort(key=lambda row: tuple(row["metric"]), reverse=True)
    best = results[0]
    layout = layout_map[best["layout"]]
    model = search.build_model(
        onnx.load(search.SOURCE),
        layout,
        np.asarray(best["w1"], dtype=np.int8),
        np.asarray(best["w2"], dtype=np.int8),
    )
    onnx.save(model, args.save_best)
    report = {
        "screen": display_path(args.screen),
        "seed": args.seed,
        "train_count": args.train,
        "rounds": args.rounds,
        "start_count": len(starts),
        "results": results,
        "best": best,
        "candidate": display_path(args.save_best),
        "candidate_sha256": hashlib.sha256(args.save_best.read_bytes()).hexdigest(),
        "known_disabled": search.ort_rate(model, known_cases, True),
        "known_default": search.ort_rate(model, known_cases, False),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("best", "candidate", "candidate_sha256", "known_disabled", "known_default")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
