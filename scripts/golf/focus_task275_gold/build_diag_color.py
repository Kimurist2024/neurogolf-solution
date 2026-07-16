#!/usr/bin/env python3
"""Build task275 with the exact rank-3 diagonal color contraction.

The incumbent's spatial router selects, for every output coordinate, the two
input cells at the corresponding local coordinate (one from each half).  Its
three color operands independently choose either cell.  Consequently the
color response for half-colors ``p`` and ``q`` is obtained by replacing every
latent color feature with ``S[:, p] + S[:, q]``.

The original color contraction uses S(3x10), T(3x3), and W(3x3).  This script
keeps the necessary sign-rank three but fits the strictly smaller exact form

    response[k] = sum_l D[l] * S[l,k] * (S[l,p] + S[l,q])**3.

In the ONNX Einsum this identifies T's ``g`` index with ``l`` and replaces W
by a length-three diagonal.  Parameters fall by 15 without changing the
spatial router.  The optimizer is only used to find a robust separating
realization for the ten generator-reachable unordered color pairs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
TASK = 275
AUTHORITY_COST = 428
RESTARTS = 1_024
STEPS = 6_501
SEED = 275_413


def equation(node: onnx.NodeProto) -> str:
    for attribute in node.attribute:
        if attribute.name == "equation":
            value = helper.get_attribute_value(attribute)
            return value.decode("ascii") if isinstance(value, bytes) else str(value)
    raise KeyError("equation")


def set_equation(node: onnx.NodeProto, value: str) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != "equation"]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute("equation", value))


def replace_initializer(
    model: onnx.ModelProto, name: str, array: np.ndarray
) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(
                numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)
            )
            return
    raise KeyError(name)


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [initializer for initializer in model.graph.initializer if initializer.name != name]
    if len(kept) == len(model.graph.initializer):
        raise KeyError(name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def train() -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    torch.manual_seed(SEED)
    pairs = [(p, 0) for p in range(5)] + [(p, 8) for p in range(5)]
    first = torch.tensor([p for p, _q in pairs], dtype=torch.long)
    second = torch.tensor([q for _p, q in pairs], dtype=torch.long)
    target = torch.full((len(pairs), 10), -1.0, dtype=torch.float32)
    for row, (p, q) in enumerate(pairs):
        target[row, p if q == 8 and p > 0 else 0] = 1.0

    s = torch.nn.Parameter(torch.randn(RESTARTS, 3, 10) * 0.7)
    d = torch.nn.Parameter(torch.randn(RESTARTS, 3))
    optimizer = torch.optim.Adam((s, d), lr=0.025)

    best_margin = -math.inf
    best_s: torch.Tensor | None = None
    best_d: torch.Tensor | None = None
    best_step = -1
    best_restart = -1

    for step in range(STEPS):
        z = s[:, :, first] + s[:, :, second]
        response = torch.einsum("rlk,rl,rlp->rpk", s, d, z**3)
        signed = response * target
        regularizer = 1.0e-5 * (
            s.square().mean(dim=(1, 2)) + d.square().mean(dim=1)
        )
        losses = torch.relu(1.0 - signed).square().mean(dim=(1, 2)) + regularizer
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_((s, d), 100.0)
        optimizer.step()

        if step % 100 == 0 or step == STEPS - 1:
            with torch.no_grad():
                exact = (signed > 0.0).all(dim=(1, 2))
                margins = signed.amin(dim=(1, 2))
                ranked = torch.where(exact, margins, torch.full_like(margins, -math.inf))
                restart = int(ranked.argmax().item())
                margin = float(ranked[restart].item())
                if margin > best_margin:
                    best_margin = margin
                    best_s = s[restart].detach().clone()
                    best_d = d[restart].detach().clone()
                    best_step = step
                    best_restart = restart

    if best_s is None or best_d is None or not math.isfinite(best_margin):
        raise RuntimeError("no exact ten-pair separator found")

    with torch.no_grad():
        z = best_s[:, first] + best_s[:, second]
        response = torch.einsum("lk,l,lp->pk", best_s, best_d, z**3)
        signed = response * target
        exact_cells = int((signed > 0.0).sum().item())
        final_margin = float(signed.min().item())
    if exact_cells != 100 or final_margin <= 0.0:
        raise RuntimeError((exact_cells, final_margin))

    metadata: dict[str, float | int] = {
        "seed": SEED,
        "restarts": RESTARTS,
        "steps": STEPS,
        "best_step": best_step,
        "best_restart": best_restart,
        "reachable_pair_signed_cells": exact_cells,
        "reachable_pair_minimum_signed_margin": final_margin,
    }
    return best_s.cpu().numpy(), best_d.cpu().numpy(), metadata


def build(s: np.ndarray, d: np.ndarray) -> onnx.ModelProto:
    with zipfile.ZipFile(AUTHORITY) as archive:
        model = onnx.load_model_from_string(archive.read("task275.onnx"))
    model = copy.deepcopy(model)
    einsum = model.graph.node[-1]
    if einsum.op_type != "Einsum":
        raise RuntimeError(einsum.op_type)

    terms, output = equation(einsum).split("->")
    terms_list = terms.split(",")
    inputs = list(einsum.input)
    if len(terms_list) != 41 or len(inputs) != 41:
        raise RuntimeError((len(terms_list), len(inputs)))
    if terms_list[13] != "gt" or inputs[13] != "S":
        raise RuntimeError((terms_list[13], inputs[13]))
    if terms_list[14] != "gl" or inputs[14] != "T":
        raise RuntimeError((terms_list[14], inputs[14]))
    if terms_list[-2:] != ["ak", "al"] or inputs[-2:] != ["S", "W"]:
        raise RuntimeError((terms_list[-2:], inputs[-2:]))

    # Identify g=l and delete the now-unnecessary dense T matrix.
    terms_list[13] = "lt"
    del terms_list[14]
    del inputs[14]
    # Replace dense W[a,l] with diagonal D[l], identifying a=l.
    terms_list[-2] = "lk"
    terms_list[-1] = "l"

    del einsum.input[:]
    einsum.input.extend(inputs)
    set_equation(einsum, ",".join(terms_list) + "->" + output)
    remove_initializer(model, "T")
    replace_initializer(model, "S", s)
    replace_initializer(model, "W", d)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    s, d, training = train()
    model = build(s, d)
    data = model.SerializeToString()
    digest = hashlib.sha256(data).hexdigest()
    candidate = HERE / f"task275_diag_color_cost413_{digest[:12]}.onnx"
    candidate.write_bytes(data)
    payload = {
        "task": TASK,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_cost": AUTHORITY_COST,
        "candidate": str(candidate.relative_to(ROOT)),
        "candidate_sha256": digest,
        "theoretical_candidate_cost": 413,
        "theoretical_gain": math.log(AUTHORITY_COST / 413),
        "transformation": "rank-3 T/W color maps -> exact learned diagonal D",
        "training": training,
        "S": s.tolist(),
        "D": d.tolist(),
    }
    (HERE / "build_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
