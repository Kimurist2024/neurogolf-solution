#!/usr/bin/env python3
"""Build the gold-exact task275 shared-map color candidate.

For a quotient coordinate, the two halves contribute colors ``p`` and ``b``;
for a remainder coordinate they contribute ``r`` and ``q``.  The incumbent
therefore evaluates

    u = S[:,p] + S[:,b]
    v = S[:,r] + S[:,q]
    response = (S.T @ W) @ (u**2 * (T.T @ v)).

The generator-reachable domain has exactly 100 such quadruples.  This script
fits all 100 with strict sign margin while imposing

    W[a,l] = D[a] * T[a,l].

Thus the same 3x3 initializer is used for both maps and only a length-three
row scale remains: 48 color parameters become 42.  The spatial router and its
proven size/orientation behavior are untouched.
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
RESTARTS = 128
STEPS = 4_101
SEED = 276_771 + sum(map(ord, "row"))


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


def reachable_domain() -> tuple[torch.Tensor, torch.Tensor]:
    quadruples = [
        (p, b, r, q)
        for p in range(5)
        for b in (0, 8)
        for r in range(5)
        for q in (0, 8)
    ]
    indices = torch.tensor(quadruples, dtype=torch.long).T
    target = torch.full((len(quadruples), 10), -1.0, dtype=torch.float32)
    for row, (p, _b, _r, q) in enumerate(quadruples):
        target[row, p if p > 0 and q == 8 else 0] = 1.0
    return indices, target


def responses(
    s: torch.Tensor, matrix: torch.Tensor, row_scale: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    u = s[:, :, indices[0]] + s[:, :, indices[1]]
    v = s[:, :, indices[2]] + s[:, :, indices[3]]
    latent = u.square() * torch.einsum("rgl,rgp->rlp", matrix, v)
    effective_output_map = matrix * row_scale[:, :, None]
    return torch.einsum("rak,ral,rlp->rpk", s, effective_output_map, latent)


def train() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    torch.manual_seed(SEED)
    indices, target = reachable_domain()
    s = torch.nn.Parameter(torch.randn(RESTARTS, 3, 10) * 0.5)
    matrix = torch.nn.Parameter(torch.randn(RESTARTS, 3, 3) * 0.5)
    row_scale = torch.nn.Parameter(torch.randn(RESTARTS, 3))
    parameters = (s, matrix, row_scale)
    optimizer = torch.optim.Adam(parameters, lr=0.012)

    best_margin = -math.inf
    best_s: torch.Tensor | None = None
    best_matrix: torch.Tensor | None = None
    best_scale: torch.Tensor | None = None
    best_step = -1
    best_restart = -1

    for step in range(STEPS):
        response = responses(s, matrix, row_scale, indices)
        signed = response * target
        regularizer = 1.0e-5 * sum(
            item.square().mean(dim=tuple(range(1, item.ndim))) for item in parameters
        )
        losses = torch.relu(1.0 - signed).square().mean(dim=(1, 2)) + regularizer
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
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
                    best_matrix = matrix[restart].detach().clone()
                    best_scale = row_scale[restart].detach().clone()
                    best_step = step
                    best_restart = restart

    if best_s is None or best_matrix is None or best_scale is None:
        raise RuntimeError("no exact 100-quadruple separator found")

    with torch.no_grad():
        final = responses(
            best_s[None, ...],
            best_matrix[None, ...],
            best_scale[None, ...],
            indices,
        )[0]
        signed = final * target
        exact_cells = int((signed > 0.0).sum().item())
        final_margin = float(signed.min().item())
    if exact_cells != 1_000 or final_margin <= 0.0:
        raise RuntimeError((exact_cells, final_margin))

    metadata: dict[str, float | int] = {
        "seed": SEED,
        "restarts": RESTARTS,
        "steps": STEPS,
        "best_step": best_step,
        "best_restart": best_restart,
        "reachable_quadruple_signed_cells": exact_cells,
        "reachable_quadruple_minimum_signed_margin": final_margin,
        "predicted_raw_minimum_margin": final_margin * (14.0**4),
    }
    return (
        best_s.cpu().numpy(),
        best_matrix.cpu().numpy(),
        best_scale.cpu().numpy(),
        metadata,
    )


def build(
    s: np.ndarray, matrix: np.ndarray, row_scale: np.ndarray
) -> onnx.ModelProto:
    with zipfile.ZipFile(AUTHORITY) as archive:
        model = onnx.load_model_from_string(archive.read("task275.onnx"))
    model = copy.deepcopy(model)
    einsum = model.graph.node[-1]
    if einsum.op_type != "Einsum":
        raise RuntimeError(einsum.op_type)
    terms, output = equation(einsum).split("->")
    terms_list = terms.split(",")
    inputs = list(einsum.input)
    if terms_list[-2:] != ["ak", "al"] or inputs[-2:] != ["S", "W"]:
        raise RuntimeError((terms_list[-2:], inputs[-2:]))

    # Reuse T[a,l] in W's old role, then multiply by the learned D[a].
    inputs[-1] = "T"
    inputs.append("W")
    terms_list.append("a")
    del einsum.input[:]
    einsum.input.extend(inputs)
    set_equation(einsum, ",".join(terms_list) + "->" + output)
    replace_initializer(model, "S", s)
    replace_initializer(model, "T", matrix)
    replace_initializer(model, "W", row_scale)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    s, matrix, row_scale, training = train()
    model = build(s, matrix, row_scale)
    data = model.SerializeToString()
    digest = hashlib.sha256(data).hexdigest()
    candidate = HERE / f"task275_shared_row_scale_cost422_{digest[:12]}.onnx"
    candidate.write_bytes(data)
    payload = {
        "task": TASK,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_cost": AUTHORITY_COST,
        "candidate": str(candidate.relative_to(ROOT)),
        "candidate_sha256": digest,
        "theoretical_candidate_cost": 422,
        "theoretical_gain": math.log(AUTHORITY_COST / 422),
        "transformation": "share T/W 3x3 map with learned 3-value W row scale",
        "training": training,
        "S": s.tolist(),
        "shared_matrix": matrix.tolist(),
        "row_scale": row_scale.tolist(),
    }
    (HERE / "shared_row_scale_build_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
