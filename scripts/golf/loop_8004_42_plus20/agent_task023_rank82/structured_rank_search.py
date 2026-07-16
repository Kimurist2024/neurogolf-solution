#!/usr/bin/env python3
"""Multi-shard structured rank search for task023's 36-byte kernel.

Only ``score_W_q`` is changed.  Unlike the existing coordinate lane, training
uses the real uint8 clipping range and the TopK index tie-break, while model
selection maximizes the worst of several independent generator shards.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HELPER_PATH = ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/train_ranker.py"
SPEC = importlib.util.spec_from_file_location("task023_rank_helper_rank82", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load task023 helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)

SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all/candidates/task023_9a2b78138891_cost1541.onnx"
ROOT_LATEST = ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/task023_ranker_coordinate2.onnx"
CANDIDATES = HERE / "candidates"


def weights(path: Path) -> np.ndarray:
    model = onnx.load(path)
    return next(
        numpy_helper.to_array(item).astype(np.float32).reshape(-1)
        for item in model.graph.initializer
        if item.name == "score_W_q"
    )


def exact_mask(features: np.ndarray, targets: np.ndarray, value: np.ndarray) -> np.ndarray:
    raw = np.einsum(
        "nif,f->ni", features.astype(np.int16), value.astype(np.int16), optimize=True
    ).astype(np.int32)
    score = np.clip(raw, 0, 255)
    key = score * 64 - np.arange(36, dtype=np.int32)[None, :]
    positive_floor = np.where(targets, key, np.iinfo(np.int32).max).min(axis=1)
    negative_peak = np.where(targets, np.iinfo(np.int32).min, key).max(axis=1)
    return positive_floor > negative_peak


def evaluate(
    value: np.ndarray,
    known_x: np.ndarray,
    known_y: np.ndarray,
    shards: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    known_mask = exact_mask(known_x, known_y, value)
    rights = [int(exact_mask(x, y, value).sum()) for x, y in shards]
    totals = [len(x) for x, _ in shards]
    return {
        "known_right": int(known_mask.sum()),
        "known_total": len(known_mask),
        "shard_right": rights,
        "shard_total": totals,
        "worst_rate": min(r / t for r, t in zip(rights, totals, strict=True)),
        "aggregate_right": sum(rights),
        "aggregate_total": sum(totals),
    }


def quantized_variants(value: np.ndarray) -> list[np.ndarray]:
    variants: dict[bytes, np.ndarray] = {}
    max_abs = max(float(np.abs(value).max()), 1.0e-6)
    # Include direct integer rounding plus a dense scale range around both the
    # saturation-free and full-int8 normalizations.
    scales = np.concatenate(
        [np.asarray([1.0]), np.geomspace(12.0 / max_abs, 127.0 / max_abs, 72)]
    )
    for scale in scales:
        q = np.clip(np.rint(value * scale), -127, 127).astype(np.int8)
        variants.setdefault(q.tobytes(), q)
    return list(variants.values())


def main() -> int:
    torch.set_num_threads(4)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    known_cases = HELPER.known_cases()
    known_x, known_y = HELPER.make_dataset(known_cases)

    train_cases = HELPER.generated_cases(40_000, 823_820_000)
    train_x, train_y = HELPER.make_dataset(train_cases)
    # Three independent validation shards make selection explicitly robust to
    # generator-seed drift instead of maximizing one pooled stream.
    shard_cases = [
        HELPER.generated_cases(5_000, seed)
        for seed in (823_920_000, 824_020_000, 824_120_000)
    ]
    shards = [HELPER.make_dataset(cases) for cases in shard_cases]

    # Known cases are repeated in training, but remain a strict integer gate.
    fit_x = np.concatenate([train_x, np.tile(known_x, (20, 1, 1))])
    fit_y = np.concatenate([train_y, np.tile(known_y, (20, 1))])
    tx = torch.from_numpy(fit_x.astype(np.float32))
    ty = torch.from_numpy(fit_y)
    kx = torch.from_numpy(known_x.astype(np.float32))
    ky = torch.from_numpy(known_y)
    index_bias = torch.arange(36, dtype=torch.float32) / 64.0

    source_w = weights(SOURCE)
    latest_w = weights(ROOT_LATEST)
    template = np.zeros(36, dtype=np.float32).reshape(6, 6)
    template[2:4, 2:4] = 60.0
    template[1:5, 1:5] -= 12.0
    template[2:4, 2:4] += 12.0
    initials = [source_w, latest_w, template.reshape(-1)]

    rng = np.random.default_rng(823_154_100)
    pool: dict[bytes, tuple[np.ndarray, dict[str, object], str, int]] = {}
    history: list[dict[str, object]] = []
    configs = [
        ("structured_path", 10.0, 0.04),
        ("structured_path", 18.0, 0.03),
        ("pairwise_path", 12.0, 0.03),
        ("pairwise_path", 20.0, 0.022),
    ]

    for run, (objective, temperature, lr) in enumerate(configs):
        # Stay in the known-safe basin found by the integer lane.  Independent
        # noise gives distinct global trajectories without discarding the hard
        # fixture solution.
        base = latest_w.copy()
        if run:
            base += rng.normal(0.0, 0.35 * run, size=36).astype(np.float32)
        initial_tensor = torch.from_numpy(base.copy())
        parameter = torch.nn.Parameter(torch.from_numpy(base))
        optimizer = torch.optim.AdamW([parameter], lr=lr, weight_decay=2.0e-5)
        generator = torch.Generator().manual_seed(823_154_100 + run)
        # Epoch zero is a valid candidate too; this also records the unchanged
        # root control in the same multi-shard comparison.
        for q in quantized_variants(parameter.detach().numpy()):
            report = evaluate(q, known_x, known_y, shards)
            if report["known_right"] == report["known_total"]:
                pool.setdefault(q.tobytes(), (q.copy(), report, objective, -1))
        for epoch in range(14):
            order = torch.randperm(len(tx), generator=generator)
            for start in range(0, len(order), 1024):
                choose = order[start : start + 1024]
                patches = tx[choose]
                target = ty[choose]
                raw = torch.einsum("bif,f->bi", patches, parameter)
                # This is the exact QLinearConv output range before TopK.  The
                # tiny index term reproduces TopK's deterministic tie order.
                score = raw.clamp(0.0, 255.0) - index_bias
                if objective.startswith("structured"):
                    positive = score.masked_fill(~target, float("inf")).amin(dim=1)
                    negative = score.masked_fill(target, float("-inf")).amax(dim=1)
                    loss = F.softplus((negative - positive + 1.0) / temperature).mean()
                else:
                    positive = score[:, :, None]
                    negative = score[:, None, :]
                    pair_mask = target[:, :, None] & ~target[:, None, :]
                    gaps = (negative - positive + 1.0) / temperature
                    loss = F.softplus(gaps[pair_mask]).mean()
                # Enforce the complete known fixture in every update, not only
                # as a final filter. This is a smooth guard plus the exact
                # integer hard gate applied below.
                known_raw = torch.einsum("bif,f->bi", kx, parameter)
                known_score = known_raw.clamp(0.0, 255.0) - index_bias
                known_positive = known_score.masked_fill(~ky, float("inf")).amin(dim=1)
                known_negative = known_score.masked_fill(ky, float("-inf")).amax(dim=1)
                known_loss = F.softplus((known_negative - known_positive + 2.0) / 8.0).mean()
                loss = loss + 0.45 * known_loss
                loss = loss + 2.0e-5 * (parameter - initial_tensor).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Screen every epoch rather than trusting the continuous loss.
            learned = parameter.detach().numpy()
            path_values = [
                latest_w + alpha * (learned - latest_w)
                for alpha in np.linspace(0.04, 1.0, 25)
            ]
            path_quantized = [
                np.clip(np.rint(value * scale), -127, 127).astype(np.int8)
                for value in path_values
                for scale in (0.96, 1.0, 1.04)
            ]
            for q in path_quantized:
                key = q.tobytes()
                if key in pool:
                    continue
                report = evaluate(q, known_x, known_y, shards)
                if report["known_right"] != report["known_total"]:
                    continue
                pool[key] = (q.copy(), report, objective, epoch)
            best = max(
                (item[1] for item in pool.values()),
                key=lambda row: (float(row["worst_rate"]), int(row["aggregate_right"])),
                default=None,
            )
            history.append(
                {"run": run, "objective": objective, "epoch": epoch, "best": best}
            )

    ranked = sorted(
        pool.values(),
        key=lambda item: (float(item[1]["worst_rate"]), int(item[1]["aggregate_right"])),
        reverse=True,
    )
    source_model = onnx.load(SOURCE)
    finalists: list[dict[str, object]] = []
    for rank, (q, report, objective, epoch) in enumerate(ranked[:16]):
        model = copy.deepcopy(source_model)
        HELPER.replace_weights(model, q)
        path = CANDIDATES / f"task023_rank82_{rank:02d}.onnx"
        onnx.save(model, path)
        finalists.append(
            {
                **report,
                "rank": rank,
                "objective": objective,
                "epoch": epoch,
                "weights": q.astype(int).tolist(),
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "root_latest": str(ROOT_LATEST.relative_to(ROOT)),
        "train_count": len(train_x),
        "validation_seeds": [823_920_000, 824_020_000, 824_120_000],
        "validation_each": 5_000,
        "known_total": len(known_x),
        "generator_non_injective": True,
        "pool_size": len(pool),
        "finalists": finalists,
        "history": history,
    }
    (HERE / "search_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0 if finalists else 2


if __name__ == "__main__":
    raise SystemExit(main())
