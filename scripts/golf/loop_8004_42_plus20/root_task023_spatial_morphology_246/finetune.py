#!/usr/bin/env python3
"""Warm-start differentiable known repair for the spatial task023 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

import search


HERE = Path(__file__).resolve().parent


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(search.ROOT))
    except ValueError:
        return str(resolved)


def integer_trial(
    w1f: np.ndarray,
    w2f: np.ndarray,
    patches: np.ndarray,
    targets: np.ndarray,
    layout: search.Layout,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    result = []
    seen = set()
    gauges = (0.55, 0.70, 0.85, 1.0, 1.15, 1.35, 1.65)
    global_scales = (0.70, 0.85, 1.0, 1.20, 1.45)
    for gauge0 in gauges:
        for gauge1 in gauges:
            q1f = w1f.copy()
            q2f = w2f.copy()
            q1f[0] *= gauge0
            q2f[:, 0] /= gauge0
            q1f[1] *= gauge1
            q2f[:, 1] /= gauge1
            for scale in global_scales:
                q1 = np.clip(np.rint(q1f), -127, 127).astype(np.int8)
                q2 = np.clip(np.rint(q2f * scale), -127, 127).astype(np.int8)
                key = q1.tobytes() + q2.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                score = search.integer_scores(patches, q1, q2, layout)
                right = int(search.success_mask(score, targets).sum())
                result.append((right, q1, q2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "task023_spatial_morphology.onnx")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=246623001)
    parser.add_argument("--output", type=Path, default=HERE / "finetune.json")
    parser.add_argument("--save-best", type=Path, default=HERE / "task023_spatial_morphology_finetuned.onnx")
    args = parser.parse_args()
    torch.set_num_threads(4)

    base = onnx.load(args.source)
    arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    start1 = arrays["morph_w1"].astype(np.float32)
    start2 = arrays["morph_w2"].astype(np.float32)
    node1, node2 = [node for node in base.graph.node if node.op_type == "QLinearConv"]
    pads1 = tuple(next(onnx.helper.get_attribute_value(a) for a in node1.attribute if a.name == "pads"))
    pads2 = tuple(next(onnx.helper.get_attribute_value(a) for a in node2.attribute if a.name == "pads"))
    layout = next(item for item in search.layouts() if item.pads1 == pads1 and item.pads2 == pads2 and item.kernel2 == start2.shape[-2:])

    cases = search.known_cases()
    x, y = search.dataset(cases)
    patches = search.stage1_patches(x, layout)
    tx = torch.from_numpy(x)
    ty = torch.from_numpy(y)
    rng = np.random.default_rng(args.seed)
    best = None
    history = []

    for restart in range(args.restarts):
        noise = 0.0 if restart == 0 else 0.08 + 0.025 * (restart % 6)
        init1 = start1 + rng.normal(0.0, noise, start1.shape).astype(np.float32)
        init2 = start2 + rng.normal(0.0, noise, start2.shape).astype(np.float32)
        w1 = torch.nn.Parameter(torch.from_numpy(init1))
        w2 = torch.nn.Parameter(torch.from_numpy(init2))
        lr = (0.003, 0.006, 0.012, 0.020)[restart % 4]
        optimizer = torch.optim.Adam([w1, w2], lr=lr)
        restart_best = 0
        for step in range(args.steps):
            scores = torch.clamp(search.torch_scores(tx, w1, w2, layout), 0.0, 255.0)
            positive = scores.masked_fill(~ty, float("inf")).amin(dim=1)
            negative = scores.masked_fill(ty, float("-inf")).amax(dim=1)
            # Smooth hinge retains margins on already-correct examples while
            # concentrating gradients on the five warm-start failures.
            loss = (F.softplus((negative - positive + 2.0) / 4.0) * 4.0).mean()
            loss = loss + 1.0e-5 * (
                (w1 - torch.from_numpy(start1)).square().mean()
                + (w2 - torch.from_numpy(start2)).square().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step + 1 == args.steps:
                trials = integer_trial(
                    w1.detach().numpy(), w2.detach().numpy(), patches, y, layout
                )
                current = max(trials, key=lambda row: row[0])
                restart_best = max(restart_best, current[0])
                if best is None or current[0] > best[0]:
                    best = current
                    history.append({"restart": restart, "step": step, "known_right": current[0], "loss": float(loss.detach())})
                    print(json.dumps(history[-1]), flush=True)
                if current[0] == len(y):
                    break
        if best is not None and best[0] == len(y):
            break
        print(json.dumps({"restart_done": restart, "best": restart_best}), flush=True)

    if best is None:
        raise RuntimeError("no integer trial")
    right, q1, q2 = best
    model = copy.deepcopy(base)
    for item in model.graph.initializer:
        if item.name == "morph_w1":
            item.CopyFrom(numpy_helper.from_array(q1.reshape(start1.shape), item.name))
        elif item.name == "morph_w2":
            item.CopyFrom(numpy_helper.from_array(q2.reshape(start2.shape), item.name))
    onnx.save(model, args.save_best)
    report = {
        "source": display_path(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "layout": layout.label,
        "seed": args.seed,
        "steps": args.steps,
        "restarts": args.restarts,
        "best_known_numpy": right,
        "w1": q1.astype(int).reshape(2, 16).tolist(),
        "w2": q2.astype(int).reshape(2, -1).tolist(),
        "history": history,
        "candidate": display_path(args.save_best),
        "candidate_sha256": hashlib.sha256(args.save_best.read_bytes()).hexdigest(),
        "known_disabled": search.ort_rate(model, cases, True),
        "known_default": search.ort_rate(model, cases, False),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if right == len(y) else 2


if __name__ == "__main__":
    raise SystemExit(main())
