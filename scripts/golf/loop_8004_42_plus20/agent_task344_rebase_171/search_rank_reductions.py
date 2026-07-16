#!/usr/bin/env python3
"""Search strict-lower rank reductions for the cost-137 task344 tensor net.

The spatial rank-2 kernel is held exactly equal to the LB-white authority.  We
only reduce/factor the color and local-rule contractions, and train against
generator-produced local states plus every visible example.
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
BASELINE = HERE / "baseline/task344.onnx"
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
sys.path.insert(0, str(ROOT))

from scripts.lib import scoring  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    local_rank: int
    color_rank: int
    coupling: str


VARIANTS = {
    "s2_split_full": Variant("s2_split_full", 2, 4, "split_full"),
    "s2_shared_full": Variant("s2_shared_full", 2, 4, "shared_full"),
    "s2_split_diag_m": Variant("s2_split_diag_m", 2, 4, "split_diag_m"),
    "s2_split_no_m": Variant("s2_split_no_m", 2, 4, "split_no_m"),
    "s3_diag_s": Variant("s3_diag_s", 3, 4, "diag_s"),
    "s3_no_s": Variant("s3_no_s", 3, 4, "no_s"),
    "s3_diag_m": Variant("s3_diag_m", 3, 4, "diag_m"),
    "r3s3_shared": Variant("r3s3_shared", 3, 3, "shared_full"),
    "r3s2_split": Variant("r3s2_split", 2, 3, "split_full"),
}


def arrays() -> dict[str, np.ndarray]:
    model = onnx.load(BASELINE)
    return {item.name: numpy_helper.to_array(item).astype(np.float32) for item in model.graph.initializer}


def one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[color, row, col] = 1.0
    return result


def local_samples(examples: list[dict], kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_rows: list[np.ndarray] = []
    centers: list[int] = []
    targets: list[int] = []
    for example in examples:
        x = one_hot(example["input"])
        z = np.einsum("dpq,ph,qw->dhw", x, kernel, kernel, optimize=True)
        height, width = len(example["input"]), len(example["input"][0])
        for row in range(height):
            for col in range(width):
                z_rows.append(z[:, row, col])
                centers.append(example["input"][row][col])
                targets.append(example["output"][row][col])
    return np.asarray(z_rows, dtype=np.float32), np.asarray(centers), np.asarray(targets)


def generator_examples(count: int, seed: int) -> list[dict]:
    generator = importlib.import_module("task_d90796e8")
    random.seed(seed)
    return [generator.generate() for _ in range(count)]


def known_examples() -> list[dict]:
    payload = scoring.load_examples(344)
    return payload["train"] + payload["test"] + payload["arc-gen"]


def authority_tensor(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, v, s, m = source["H"], source["V"], source["S"], source["M"]
    a = h @ v
    q = np.einsum("kc,ky,yo->kco", v, m, v, optimize=True)
    local = np.einsum("sx,xk,kco->sco", s, h, q, optimize=True)
    tensor = np.einsum("sd,sco->dco", a, local, optimize=True)
    return tensor, q, a


def rank2_seed(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    tensor, q, _ = authority_tensor(source)
    u, singular, vh = np.linalg.svd(tensor.reshape(10, 100), full_matrices=False)
    a2 = np.sqrt(singular[:2])[:, None] * u[:, :2].T
    local2 = (np.sqrt(singular[:2])[:, None] * vh[:2]).reshape(2, 10, 10)
    h2 = a2 @ np.linalg.pinv(source["V"])
    q_design = q.reshape(4, -1).T
    t2 = np.stack([np.linalg.lstsq(q_design, item.reshape(-1), rcond=None)[0] for item in local2])
    # Balance the two equivalent H/T component scalings for optimization.
    for index in range(2):
        scale = max(float(np.linalg.norm(h2[index])) / 2.0, 1e-6)
        h2[index] /= scale
        t2[index] *= scale
    return h2.astype(np.float32), t2.astype(np.float32)


def compress_color(source: dict[str, np.ndarray], rank: int) -> tuple[np.ndarray, np.ndarray]:
    v = source["V"]
    u, singular, vh = np.linalg.svd(v, full_matrices=False)
    compressed = (np.sqrt(singular[:rank])[:, None] * vh[:rank]).astype(np.float32)
    left = (u[:, :rank] * np.sqrt(singular[:rank])).astype(np.float32)
    h = source["H"] @ left
    return h.astype(np.float32), compressed


def initialize(variant: Variant, restart: int, seed: int) -> dict[str, torch.nn.Parameter]:
    source = arrays()
    torch.manual_seed(seed + 7919 * restart)
    if variant.color_rank == 4:
        v = source["V"].copy()
    else:
        _, v = compress_color(source, variant.color_rank)

    if variant.local_rank == 2 and variant.color_rank == 4:
        h, t = rank2_seed(source)
    else:
        h_full, _ = compress_color(source, variant.color_rank)
        h = h_full[: variant.local_rank].copy()
        t = h.copy()

    values: dict[str, np.ndarray] = {"H": h, "V": v}
    if variant.coupling.startswith("split"):
        values["T"] = t
    if variant.coupling in {"shared_full", "diag_m"}:
        if variant.local_rank == 2 and variant.color_rank == 4:
            values["S"] = (t @ np.linalg.pinv(h)).astype(np.float32)
        else:
            values["S"] = source["S"][: variant.local_rank, : variant.local_rank].copy()
    elif variant.coupling == "diag_s":
        values["D"] = np.diag(source["S"][: variant.local_rank, : variant.local_rank]).copy()
    if variant.coupling in {"split_full", "shared_full", "diag_s", "no_s"}:
        if variant.color_rank == 4:
            values["M"] = source["M"].copy()
        else:
            values["M"] = np.eye(variant.color_rank, dtype=np.float32)
    elif variant.coupling in {"split_diag_m", "diag_m"}:
        values["D"] = np.diag(source["M"][: variant.color_rank, : variant.color_rank]).copy()

    noise = 0.001 if restart else 0.0
    return {
        name: torch.nn.Parameter(torch.from_numpy(value) + noise * torch.randn(value.shape))
        for name, value in values.items()
    }


def logits(z: torch.Tensor, center: torch.Tensor, factors: dict[str, torch.Tensor], variant: Variant) -> torch.Tensor:
    h, v = factors["H"], factors["V"]
    neighbor = z @ (h @ v).T
    if variant.coupling.startswith("split"):
        q = neighbor @ factors["T"]
    elif variant.coupling == "diag_s":
        q = (neighbor * factors["D"]) @ h
    elif variant.coupling == "no_s":
        q = neighbor @ h
    else:
        q = (neighbor @ factors["S"]) @ h
    q = q * v[:, center].T
    if variant.coupling in {"split_diag_m", "diag_m"}:
        q = q * factors["D"]
    elif variant.coupling not in {"split_no_m"}:
        q = q @ factors["M"]
    return q @ v


def metrics(raw: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    positive = torch.nn.functional.one_hot(target, 10).bool()
    predicted = raw > 0
    wrong_cells = int(torch.count_nonzero(predicted != positive).item())
    wrong_samples = int(torch.count_nonzero(torch.any(predicted != positive, dim=1)).item())
    pos = raw[positive]
    neg = raw[~positive]
    return {
        "wrong_cells": wrong_cells,
        "wrong_samples": wrong_samples,
        "min_positive": float(pos.min().item()),
        "max_negative": float(neg.max().item()),
        "min_abs_nonzero": float(raw.abs().min().item()),
    }


def equation(variant: Variant) -> tuple[str, list[str]]:
    # The authority realizes K(p,h)^16 and K(q,w)^16 with sixteen repeated
    # rank-2 Gram pairs per spatial axis.  Preserve that exact finite-support
    # kernel in every candidate.
    symbols_p = list("abefgijmnrtuvzAB")
    symbols_q = list("CDEFGHIJKLMNOPQR")
    terms = ["sl", "ld", "...dpq"]
    inputs = ["H", "V", "input"]
    for symbol in symbols_p:
        terms += [f"{symbol}p", f"{symbol}h"]
        inputs += ["B", "B"]
    for symbol in symbols_q:
        partner = chr(ord(symbol) + 20) if symbol <= "F" else chr(ord(symbol) + 48)
        # Einsum labels are case-sensitive; choose a distinct companion label.
        terms += [f"{symbol}q", f"{partner}w"]
        inputs += ["B", "B"]
        # The pair must share its summation index.  Correct the second term.
        terms[-1] = f"{symbol}w"
    terms += ["...chw", "kc"]
    inputs += ["input", "V"]
    if variant.coupling.startswith("split"):
        terms.append("sk")
        inputs.append("T")
    elif variant.coupling == "diag_s":
        terms += ["s", "sk"]
        inputs += ["D", "H"]
    elif variant.coupling == "no_s":
        terms.append("sk")
        inputs.append("H")
    else:
        terms += ["sx", "xk"]
        inputs += ["S", "H"]
    if variant.coupling in {"split_diag_m", "diag_m"}:
        terms.append("k")
        inputs.append("D")
        terms.append("ko")
        inputs.append("V")
    elif variant.coupling == "split_no_m":
        terms.append("ko")
        inputs.append("V")
    else:
        terms += ["ky", "yo"]
        inputs += ["M", "V"]
    return ",".join(terms) + "->...ohw", inputs


def save_candidate(path: Path, variant: Variant, factors: dict[str, torch.Tensor]) -> int:
    source = arrays()
    initializers = {"B": source["B"]}
    initializers.update({name: value.detach().cpu().numpy().astype(np.float32) for name, value in factors.items()})
    einsum, inputs = equation(variant)
    graph = helper.make_graph(
        [helper.make_node("Einsum", inputs, ["output"], equation=einsum)],
        "task344_rebase171_" + variant.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(value, name) for name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=7)
    model.producer_name = "codex-task344-rebase171-rank-search"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return sum(int(np.prod(item.dims)) for item in model.graph.initializer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--valid", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=344171001)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    source = arrays()
    gram = source["B"].T @ source["B"]
    kernel = np.power(gram, 16).astype(np.float32)
    visible = known_examples()
    train_examples = generator_examples(args.train, args.seed) + visible * 6
    valid_examples = generator_examples(args.valid, args.seed + 1)
    train = local_samples(train_examples, kernel)
    valid = local_samples(valid_examples, kernel)
    known = local_samples(visible, kernel)
    train_tensors = tuple(torch.from_numpy(item) for item in train)
    valid_tensors = tuple(torch.from_numpy(item) for item in valid)
    known_tensors = tuple(torch.from_numpy(item) for item in known)

    best_key: tuple[int, int, int, float] | None = None
    best_row: dict | None = None
    output = HERE / f"candidates/task344_{variant.name}.onnx"
    history: list[dict] = []
    for restart in range(args.restarts):
        factors = initialize(variant, restart, args.seed)
        optimizer = torch.optim.Adam(factors.values(), lr=4e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)
        order = torch.arange(len(train_tensors[0]))
        for epoch in range(args.epochs):
            order = order[torch.randperm(len(order))]
            for offset in range(0, len(order), args.batch):
                ids = order[offset : offset + args.batch]
                raw = logits(train_tensors[0][ids], train_tensors[1][ids], factors, variant)
                target = train_tensors[2][ids]
                positive = torch.nn.functional.one_hot(target, 10).bool()
                loss = torch.relu(4.0 - raw[positive]).mean() + torch.relu(raw[~positive] + 4.0).mean()
                loss += 1e-10 * sum(value.square().mean() for value in factors.values())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(factors.values()), 1000.0)
                optimizer.step()
            scheduler.step()
            if epoch % 5 != 0 and epoch + 1 != args.epochs:
                continue
            with torch.no_grad():
                valid_metrics = metrics(logits(valid_tensors[0], valid_tensors[1], factors, variant), valid_tensors[2])
                known_metrics = metrics(logits(known_tensors[0], known_tensors[1], factors, variant), known_tensors[2])
            key = (
                int(known_metrics["wrong_cells"]),
                int(valid_metrics["wrong_cells"]),
                int(valid_metrics["wrong_samples"]),
                -min(float(valid_metrics["min_positive"]), -float(valid_metrics["max_negative"])),
            )
            row = {"restart": restart, "epoch": epoch + 1, "known": known_metrics, "valid": valid_metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if best_key is None or key < best_key:
                best_key = key
                cost = save_candidate(output, variant, factors)
                best_row = {**row, "parameter_cost": cost}
            if key[0] == 0 and key[1] == 0 and min(float(valid_metrics["min_positive"]), -float(valid_metrics["max_negative"])) >= 1.0:
                break

    result = {
        "variant": variant.__dict__,
        "authority_cost": 137,
        "candidate": str(output.relative_to(ROOT)),
        "best": best_row,
        "search_cases": {"train": args.train, "valid": args.valid, "known": len(visible)},
        "history": history,
    }
    (HERE / f"audit/search_{variant.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("variant", "candidate", "best")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
