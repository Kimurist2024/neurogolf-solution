#!/usr/bin/env python3
"""Train a generator-exhaustive rank-3 version of the incumbent task232 tensor net.

The exhaustive atomic rows cover every generator width, legal source column,
and legal source color.  The graph is row-separable, so multi-row examples are
exactly compositions of these rows.  No operand or lookup table is added.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_SHA256 = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def dataset() -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for width in range(7, 15):
        for col in range(width // 2 + 1):
            for color in COLORS:
                x = np.zeros((10, 30), dtype=np.float32)
                y = np.zeros((10, 30), dtype=np.float32)
                x[0, :width] = 1.0
                x[0, col] = 0.0
                x[color, col] = 1.0
                y[0, :width] = 1.0
                for pos in range(col, width):
                    out_color = color if (pos - col) % 2 == 0 else 5
                    y[0, pos] = 0.0
                    y[out_color, pos] = 1.0
                inputs.append(x)
                targets.append(y)
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))


def logits(v: torch.Tensor, f: torch.Tensor, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Algebraically identical to the incumbent 11-operand Einsum, after
    # eliminating indices that occur only in constants.
    a = torch.einsum("uvr,xu,xv->xr", v, f, f)
    s = v.sum(dim=(0, 1))
    t = torch.einsum("pr,bcp,cr->br", a, x, g)
    mask = x.sum(dim=1)
    return mask[:, None, :] * torch.einsum("xr,r,or,br->box", a, s, g, t)


def replace(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))
            return
    raise RuntimeError(name)


def main() -> int:
    torch.set_num_threads(1)
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"baseline SHA mismatch: {digest}")
    with zipfile.ZipFile(BASELINE) as archive:
        base = onnx.load_model_from_string(archive.read("task232.onnx"))
    initial = {item.name: np.asarray(numpy_helper.to_array(item)) for item in base.graph.initializer}
    x, y = dataset()
    sign = y * 2.0 - 1.0
    weight = torch.where(y > 0.5, 9.0, 1.0)
    active = (x.sum(dim=1, keepdim=True) > 0).expand_as(y)
    starts = ((0, 1, 3), (1, 2, 3), (0, 1, 2))
    best: dict[str, object] | None = None
    trials: list[dict[str, object]] = []

    for trial, keep in enumerate(starts):
        torch.manual_seed(232000 + trial)
        v = torch.nn.Parameter(torch.from_numpy(initial["V"][:, :, keep].copy()))
        f = torch.nn.Parameter(torch.from_numpy(initial["F"].copy()))
        g = torch.nn.Parameter(torch.from_numpy(initial["G"][:, keep].copy()))
        optimizer = torch.optim.Adam([v, f, g], lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=400, min_lr=2e-5)
        with torch.no_grad():
            initial_raw = logits(v, f, g, x)
            initial_margin = (sign * initial_raw)[active]
            initial_wrong = int(((initial_raw > 0) != (y > 0)).sum())
        trial_best = {
            "wrong": initial_wrong,
            "min_margin": float(initial_margin.min()),
            "step": 0,
        }
        state = (v.detach().cpu().numpy().copy(), f.detach().cpu().numpy().copy(), g.detach().cpu().numpy().copy())
        for step in range(1, 12001):
            optimizer.zero_grad(set_to_none=True)
            raw = logits(v, f, g, x)
            margin = sign * raw
            # Balanced logistic objective plus an explicit near-boundary term.
            weighted_logistic = weight * torch.nn.functional.softplus(-margin)
            weighted_boundary = weight * torch.relu(0.25 - margin).square()
            loss = weighted_logistic[active].mean()
            loss = loss + 0.2 * weighted_boundary[active].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v, f, g], 100.0)
            optimizer.step()
            scheduler.step(float(loss.detach()))
            if step == 1 or step % 25 == 0:
                with torch.no_grad():
                    check_raw = logits(v, f, g, x)
                    check = (sign * check_raw)[active]
                    wrong = int(((check_raw > 0) != (y > 0)).sum())
                    min_margin = float(check.min())
                    key = (wrong, -min_margin)
                    old_key = (int(trial_best["wrong"]), -float(trial_best["min_margin"]))
                    if key < old_key:
                        trial_best = {"wrong": wrong, "min_margin": min_margin, "step": step, "loss": float(loss.detach())}
                        state = (v.detach().cpu().numpy().copy(), f.detach().cpu().numpy().copy(), g.detach().cpu().numpy().copy())
                    if wrong == 0 and min_margin > 0.02:
                        break
        row = {"trial": trial, "start_components": list(keep), **trial_best}
        trials.append(row)
        print(json.dumps(row), flush=True)
        if state is not None:
            candidate_key = (int(trial_best["wrong"]), -float(trial_best["min_margin"]))
            best_key = (int(best["wrong"]), -float(best["min_margin"])) if best else None
            if best_key is None or candidate_key < best_key:
                best = {**row, "state": state}

    if best is None:
        raise RuntimeError("training produced no state")
    v_np, f_np, g_np = best.pop("state")  # type: ignore[misc]
    model = copy.deepcopy(base)
    replace(model, "V", v_np)
    replace(model, "F", f_np)
    replace(model, "G", g_np)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path = HERE / "task232_rank3_trained.onnx"
    onnx.save(model, path)
    payload = {
        "baseline_sha256": digest,
        "exhaustive_atomic_rows": int(x.shape[0]),
        "trials": trials,
        "best": best,
        "candidate": str(path.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "initializer_params": sum(int(np.prod(item.dims)) for item in model.graph.initializer),
        "max_einsum_inputs": max(len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
    }
    (HERE / "rank3_training.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
