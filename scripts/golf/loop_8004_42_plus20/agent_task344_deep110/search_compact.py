#!/usr/bin/env python3
"""Search non-giant, no-S task344 tensor nets below cost 197.

All candidates are one-node static Einsum graphs whose only parameters are
small factor matrices.  Training data comes from task_d90796e8.generate(); the
visible corpus is used only as a fail-closed numerical guard.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_sound/task344_sound_cost197.onnx"
sys.path.insert(0, str(TASKS))


@dataclass(frozen=True)
class Variant:
    name: str
    color_rank: int
    spatial_rank: int
    local: str


VARIANTS = {
    "p2_shared_full": Variant("p2_shared_full", 4, 4, "shared_full"),  # cost 188
    "p2_split_diag": Variant("p2_split_diag", 4, 4, "split_diag"),  # cost 188
    "p2_split_none": Variant("p2_split_none", 4, 4, "split_none"),  # cost 184
    "p2_rank3_split_full": Variant("p2_rank3_split_full", 3, 4, "split_full"),  # cost 177
    "p2_b3_split_full": Variant("p2_b3_split_full", 4, 3, "split_full"),  # cost 170
}


def one_hot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((10, 10, 10), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[color, row, col] = 1.0
    return out


def make_dataset(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = importlib.import_module("task_d90796e8")
    random.seed(seed)
    inputs = np.zeros((count, 10, 10, 10), dtype=np.float32)
    targets = np.zeros_like(inputs)
    inside = np.zeros((count, 1, 10, 10), dtype=bool)
    for index in range(count):
        example = generator.generate()
        inputs[index] = one_hot(example["input"])
        targets[index] = one_hot(example["output"])
        height, width = len(example["input"]), len(example["input"][0])
        inside[index, 0, :height, :width] = True
    return torch.from_numpy(inputs), torch.from_numpy(targets), torch.from_numpy(inside)


def examples_dataset(examples: list[dict[str, list[list[int]]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = np.stack([one_hot(example["input"]) for example in examples])
    targets = np.stack([one_hot(example["output"]) for example in examples])
    inside = np.zeros((len(examples), 1, 10, 10), dtype=bool)
    for index, example in enumerate(examples):
        inside[index, 0, : len(example["input"]), : len(example["input"][0])] = True
    return torch.from_numpy(inputs), torch.from_numpy(targets), torch.from_numpy(inside)


def compact_basis(source: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-r PSD Gram whose elementwise square approximates source^4."""
    gram = source[:, :10].T @ source[:, :10]
    target_gram = gram * gram
    values, vectors = np.linalg.eigh(target_gram)
    order = np.argsort(values)[::-1][:rank]
    return (np.sqrt(np.maximum(values[order], 0.0))[:, None] * vectors[:, order].T).astype(np.float32)


def solve_full_m(h: np.ndarray, v: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.einsum("sk,kc,yo->scoky", h, v, v, optimize=True)
    x = design.reshape(-1, h.shape[1] * h.shape[1])
    y = target.reshape(-1)
    solution, *_ = np.linalg.lstsq(x, y, rcond=None)
    return solution.reshape(h.shape[1], h.shape[1]).astype(np.float32)


def solve_diag(t: np.ndarray, v: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.einsum("sk,kc,ko->scok", t, v, v, optimize=True)
    solution, *_ = np.linalg.lstsq(design.reshape(-1, v.shape[0]), target.reshape(-1), rcond=None)
    return solution.astype(np.float32)


def source_arrays() -> dict[str, np.ndarray]:
    model = onnx.load(SOURCE)
    return {item.name: numpy_helper.to_array(item).astype(np.float32) for item in model.graph.initializer}


def initialize(variant: Variant, restart: int, seed: int) -> dict[str, torch.nn.Parameter]:
    torch.manual_seed(seed + restart)
    arrays = source_arrays()
    h4, v4, s4, m4 = arrays["H"], arrays["V"], arrays["S"], arrays["M"]
    target_local = np.einsum("sx,xk,kc,ky,yo->sco", s4, h4, v4, m4, v4, optimize=True)
    rank = variant.color_rank
    if rank == 4:
        h = h4.copy()
        v = v4.copy()
        t = (s4 @ h4).astype(np.float32)
    else:
        # Rank-3 starts from the exact three neighbor features and learns a
        # separate center/output representation end-to-end.
        neighbor = h4 @ v4
        h = np.eye(3, dtype=np.float32)
        v = neighbor.astype(np.float32)
        t = np.eye(3, dtype=np.float32)
    values: dict[str, np.ndarray] = {
        "H": h,
        "V": v,
        "B": compact_basis(arrays["B"], variant.spatial_rank),
    }
    if variant.local == "shared_full":
        if restart % 2:
            values["H"] = t.copy()
            values["V"] = np.linalg.lstsq(values["H"], h4 @ v4, rcond=None)[0].astype(np.float32)
        values["M"] = solve_full_m(values["H"], values["V"], target_local)
    elif variant.local == "split_full":
        values["T"] = t
        if rank == 4:
            values["M"] = m4.copy()
        else:
            values["M"] = np.eye(rank, dtype=np.float32)
    elif variant.local == "split_diag":
        values["T"] = t
        values["D"] = solve_diag(t, v, target_local)
    elif variant.local == "split_none":
        values["T"] = t
    else:
        raise ValueError(variant.local)
    return {
        name: torch.nn.Parameter(torch.from_numpy(value) + torch.randn(value.shape) * (0.002 if name != "B" else 0.0005))
        for name, value in values.items()
    }


def raw_output(inputs: torch.Tensor, factors: dict[str, torch.Tensor], variant: Variant) -> torch.Tensor:
    h, v, basis = factors["H"], factors["V"], factors["B"]
    kernel = (basis.T @ basis).square()
    neighbor = torch.einsum("sd,bdpq,ph,qw->bshw", h @ v, inputs, kernel, kernel)
    if variant.local == "shared_full":
        local = torch.einsum("kc,sk,ky,yo->sco", v, h, factors["M"], v)
    elif variant.local == "split_full":
        local = torch.einsum("kc,sk,ky,yo->sco", v, factors["T"], factors["M"], v)
    elif variant.local == "split_diag":
        local = torch.einsum("kc,sk,k,ko->sco", v, factors["T"], factors["D"], v)
    elif variant.local == "split_none":
        local = torch.einsum("kc,sk,ko->sco", v, factors["T"], v)
    else:
        raise ValueError(variant.local)
    return torch.einsum("bchw,sco,bshw->bohw", inputs, local, neighbor)


def equation_and_inputs(variant: Variant) -> tuple[str, list[str]]:
    spatial = ["B", "B", "B", "B", "B", "B", "B", "B"]
    prefix_equation = "sl,ld,bdpq,ap,ah,ep,eh,iq,iw,jq,jw,bchw,"
    prefix_inputs = ["H", "V", "input", *spatial, "input"]
    if variant.local == "shared_full":
        return prefix_equation + "kc,sk,ky,yo->bohw", prefix_inputs + ["V", "H", "M", "V"]
    if variant.local == "split_full":
        return prefix_equation + "kc,sk,ky,yo->bohw", prefix_inputs + ["V", "T", "M", "V"]
    if variant.local == "split_diag":
        return prefix_equation + "kc,sk,k,ko->bohw", prefix_inputs + ["V", "T", "D", "V"]
    if variant.local == "split_none":
        return prefix_equation + "kc,sk,ko->bohw", prefix_inputs + ["V", "T", "V"]
    raise ValueError(variant.local)


def save_candidate(path: Path, variant: Variant, factors: dict[str, torch.Tensor]) -> int:
    arrays: dict[str, np.ndarray] = {}
    for name, value in factors.items():
        array = value.detach().cpu().numpy().astype(np.float32)
        if name == "B":
            padded = np.zeros((variant.spatial_rank, 30), dtype=np.float32)
            padded[:, :10] = array
            array = padded
        arrays[name] = array
    equation, inputs = equation_and_inputs(variant)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        [helper.make_node("Einsum", inputs, ["output"], equation=equation)],
        f"task344_{variant.name}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [output_info],
        [numpy_helper.from_array(arrays[name], name) for name in arrays],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    model.producer_name = "codex-task344-deep110"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return sum(int(np.prod(item.dims or [1])) for item in model.graph.initializer)


def mismatch(raw: torch.Tensor, target: torch.Tensor, inside: torch.Tensor) -> tuple[int, float, float]:
    mask = inside.expand(-1, 10, -1, -1)
    positive = target > 0
    negative = (~positive) & mask
    wrong = int(torch.count_nonzero(((raw > 0) != positive) & mask).item())
    return wrong, float(raw[positive].min().item()), float(raw[negative].max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--fixed", type=int, default=1000)
    parser.add_argument("--valid", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=344110001)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    train_x, train_y, train_inside = make_dataset(args.train, args.seed)
    fixed_x, fixed_y, fixed_inside = make_dataset(args.fixed, 777344)
    visible = json.loads((ROOT / "inputs/neurogolf-2026/task344.json").read_text())
    guard_x, guard_y, guard_inside = examples_dataset(visible["train"] + visible["test"] + visible["arc-gen"])
    train_x = torch.cat((train_x, fixed_x, guard_x.repeat((4, 1, 1, 1))), 0)
    train_y = torch.cat((train_y, fixed_y, guard_y.repeat((4, 1, 1, 1))), 0)
    train_inside = torch.cat((train_inside, fixed_inside, guard_inside.repeat((4, 1, 1, 1))), 0)
    valid_x, valid_y, valid_inside = make_dataset(args.valid, args.seed + 1)

    best_total = 10**18
    best_row = None
    out = HERE / f"candidates/task344_{variant.name}.onnx"
    for restart in range(args.restarts):
        factors = initialize(variant, restart, args.seed)
        optimizer = torch.optim.Adam(factors.values(), lr=1.5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
        order = torch.arange(len(train_x))
        for epoch in range(args.epochs):
            order = order[torch.randperm(len(order))]
            for offset in range(0, len(order), args.batch):
                ids = order[offset : offset + args.batch]
                raw = raw_output(train_x[ids], factors, variant)
                positive = train_y[ids] > 0
                negative = (~positive) & train_inside[ids].expand(-1, 10, -1, -1)
                loss = torch.relu(32.0 - raw[positive]).mean() + torch.relu(raw[negative] + 32.0).mean()
                loss = loss + 1e-9 * sum(value.square().mean() for value in factors.values())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(factors.values()), 5000.0)
                optimizer.step()
            scheduler.step()
            if epoch % 2 and epoch + 1 != args.epochs:
                continue
            with torch.no_grad():
                valid = mismatch(raw_output(valid_x, factors, variant), valid_y, valid_inside)
                guard = mismatch(raw_output(guard_x, factors, variant), guard_y, guard_inside)
            total = valid[0] + guard[0]
            print(
                f"variant={variant.name} restart={restart} epoch={epoch+1:03d} "
                f"valid={valid[0]} guard={guard[0]} min_pos={valid[1]:.4f} max_neg={valid[2]:.4f}",
                flush=True,
            )
            if total < best_total:
                best_total = total
                cost = save_candidate(out, variant, factors)
                best_row = {"restart": restart, "epoch": epoch + 1, "valid": valid, "guard": guard, "cost": cost}
            if total == 0 and valid[1] >= 1.0 and valid[2] <= 0.0:
                break
    result = {"variant": variant.__dict__, "found": best_total == 0, "best_total_mismatch": best_total, "best": best_row, "candidate": str(out.relative_to(ROOT))}
    (HERE / f"search_{variant.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
