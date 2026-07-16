#!/usr/bin/env python3
"""Probe a cost-188 task344 factorization with the 3x3 S factor removed.

This is an isolated, non-promoting search.  It keeps the verified spatial
rank-4 basis and learns only the constrained color factors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_sound/task344_sound_cost197.onnx"
HELPERS = ROOT / "scripts/golf/scratch_codex/task344"
sys.path.insert(0, str(HELPERS))
import train_rank4_boundary as base  # noqa: E402


def no_s_raw(
    inputs: torch.Tensor,
    h_factor: torch.Tensor,
    v_factor: torch.Tensor,
    basis: torch.Tensor,
    m_factor: torch.Tensor,
) -> torch.Tensor:
    color_neighbor = h_factor @ v_factor
    color_local = torch.einsum("sk,kc,ky,yo->sco", h_factor, v_factor, m_factor, v_factor)
    kernel = (basis.T @ basis).pow(4)
    neighbor = torch.einsum(
        "sd,bdpq,ph,qw->bshw", color_neighbor, inputs, kernel, kernel
    )
    return torch.einsum("bchw,sco,bshw->bohw", inputs, color_local, neighbor)


def solve_m(
    h_factor: np.ndarray,
    v_factor: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    design = np.einsum(
        "sk,kc,yo->scoky", h_factor, v_factor, v_factor, optimize=True
    )
    active_centers = np.asarray((0, 2, 3, 5))
    x = design[:, active_centers].reshape(-1, 16)
    y = target[:, active_centers].reshape(-1)
    solution, *_ = np.linalg.lstsq(x, y, rcond=None)
    return solution.reshape(4, 4).astype(np.float32)


def initializations(arrays: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    target_neighbor = arrays["H"] @ arrays["V"]
    target_local = np.einsum(
        "sx,xk,kc,ky,yo->sco",
        arrays["S"], arrays["H"], arrays["V"], arrays["M"], arrays["V"],
        optimize=True,
    )
    direct_h = arrays["H"].copy()
    direct_v = arrays["V"].copy()
    direct_m = solve_m(direct_h, direct_v, target_local)

    folded_h = (arrays["S"] @ arrays["H"]).astype(np.float32)
    folded_v = np.linalg.lstsq(folded_h, target_neighbor, rcond=None)[0].astype(np.float32)
    folded_m = solve_m(folded_h, folded_v, target_local)
    return [
        {"H": direct_h, "V": direct_v, "M": direct_m},
        {"H": folded_h, "V": folded_v, "M": folded_m},
    ]


def save_candidate(source: Path, output: Path, arrays: dict[str, np.ndarray]) -> None:
    model = onnx.load(source)
    kept = []
    for initializer in model.graph.initializer:
        if initializer.name == "S":
            continue
        if initializer.name in arrays:
            kept.append(numpy_helper.from_array(arrays[initializer.name], initializer.name))
        else:
            kept.append(initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    node = model.graph.node[0]
    inputs = list(node.input)
    if inputs[-6:] != ["input", "V", "S", "H", "M", "V"]:
        raise RuntimeError("unexpected authority task344 operand tail")
    del inputs[-4]
    del node.input[:]
    node.input.extend(inputs)
    equation = next(attr for attr in node.attribute if attr.name == "equation")
    equation.s = (
        b"sl,ld,bdpq,ap,ah,ep,eh,fp,fh,gp,gh,iq,iw,jq,jw,mq,mw,nq,nw,"
        b"bchw,kc,sk,ky,yo->bohw"
    )
    model.graph.name = "task344_rank4_no_s_cost188_probe"
    model.producer_name = "codex-sound93-no-s-probe"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    params = sum(int(np.prod(item.dims)) for item in model.graph.initializer)
    if params != 188:
        raise RuntimeError(f"expected cost188, got {params}")
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "candidates/task344_no_s_cost188.onnx")
    parser.add_argument("--train", type=int, default=3500)
    parser.add_argument("--fixed", type=int, default=1200)
    parser.add_argument("--valid", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=344930093)
    args = parser.parse_args()

    model = onnx.load(SOURCE)
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float32).copy()
        for item in model.graph.initializer
    }
    if {name: value.shape for name, value in arrays.items()} != {
        "H": (3, 4), "V": (4, 10), "B": (4, 30), "S": (3, 3), "M": (4, 4)
    }:
        raise RuntimeError("unexpected authority task344 structure")
    basis = torch.from_numpy(arrays["B"][:, :10])

    train_x, train_y, train_inside = base.make_dataset(args.train, args.seed)
    fixed_x, fixed_y, fixed_inside = base.make_dataset(args.fixed, 777344)
    visible = json.loads((ROOT / "inputs/neurogolf-2026/task344.json").read_text())
    guard_x, guard_y, guard_inside = base.examples_dataset(
        visible["train"] + visible["test"] + visible["arc-gen"]
    )
    train_x = torch.cat((train_x, fixed_x, guard_x.repeat((4, 1, 1, 1))), dim=0)
    train_y = torch.cat((train_y, fixed_y, guard_y.repeat((4, 1, 1, 1))), dim=0)
    train_inside = torch.cat(
        (train_inside, fixed_inside, guard_inside.repeat((4, 1, 1, 1))), dim=0
    )
    valid_x, valid_y, valid_inside = base.make_dataset(args.valid, args.seed + 1)

    starts = initializations(arrays)
    global_best = 10**18
    best_row: dict[str, object] | None = None
    for restart in range(args.restarts):
        torch.manual_seed(args.seed + restart)
        start = starts[restart % len(starts)]
        factors = {
            name: torch.nn.Parameter(torch.from_numpy(value) + torch.randn(value.shape) * 0.002)
            for name, value in start.items()
        }
        optimizer = torch.optim.Adam(factors.values(), lr=1.5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-4
        )
        order = torch.arange(len(train_x))
        for epoch in range(args.epochs):
            order = order[torch.randperm(len(order))]
            for offset in range(0, len(order), args.batch):
                ids = order[offset : offset + args.batch]
                raw = no_s_raw(
                    train_x[ids], factors["H"], factors["V"], basis, factors["M"]
                )
                positive = train_y[ids] > 0
                inside = train_inside[ids].expand(-1, 10, -1, -1)
                negative = (~positive) & inside
                loss = (
                    torch.relu(32.0 - raw[positive]).mean()
                    + torch.relu(raw[negative] + 32.0).mean()
                    + 1e-8 * sum(value.square().mean() for value in factors.values())
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(factors.values()), 1000.0)
                optimizer.step()
            scheduler.step()

            if epoch % 2 and epoch + 1 != args.epochs:
                continue
            with torch.no_grad():
                valid_raw = no_s_raw(
                    valid_x, factors["H"], factors["V"], basis, factors["M"]
                )
                guard_raw = no_s_raw(
                    guard_x, factors["H"], factors["V"], basis, factors["M"]
                )
                valid_mismatch = base.mismatch_count(valid_raw, valid_y)
                guard_mismatch = base.mismatch_count(guard_raw, guard_y)
                mask = valid_inside.expand(-1, 10, -1, -1)
                min_positive = float(valid_raw[valid_y > 0].min().item())
                max_negative = float(valid_raw[(valid_y <= 0) & mask].max().item())
            total = valid_mismatch + guard_mismatch
            print(
                f"restart={restart} epoch={epoch + 1:03d} valid={valid_mismatch} "
                f"guard={guard_mismatch} min_pos={min_positive:.3f} "
                f"max_neg={max_negative:.3f}",
                flush=True,
            )
            if total < global_best:
                global_best = total
                values = {
                    name: value.detach().numpy().astype(np.float32)
                    for name, value in factors.items()
                }
                save_candidate(SOURCE, args.output, values)
                best_row = {
                    "restart": restart,
                    "epoch": epoch + 1,
                    "valid_mismatch": valid_mismatch,
                    "guard_mismatch": guard_mismatch,
                    "min_positive": min_positive,
                    "max_negative": max_negative,
                }
            if total == 0 and min_positive >= 1.0 and max_negative <= 0.0:
                (HERE / "task344_no_s_search.json").write_text(
                    json.dumps({"found": True, "best": best_row}, indent=2) + "\n"
                )
                return 0

    (HERE / "task344_no_s_search.json").write_text(
        json.dumps({"found": False, "best": best_row}, indent=2) + "\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
