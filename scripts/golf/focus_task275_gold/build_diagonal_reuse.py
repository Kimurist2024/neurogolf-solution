#!/usr/bin/env python3
"""Build task275 by reusing one color map and its diagonal.

The color response uses one shared 3x3 map ``M`` in the old T and W roles,
with the W-role rows scaled by ``diag(M)``.  ONNX Einsum's repeated input
subscript ``aa`` extracts that diagonal without another initializer:

    S[l,c] S[l,d] S[g,t] M[g,l] S[a,k] M[a,l] M[a,a].

The constants below are a float32 gauge transform of a separator trained on
all 100 generator-reachable color quadruples.  Before the gauge transform the
row scale was ``diag(M) * exp(E)``.  Applying

    S'[r,:] = exp(-E[r]/2) S[r,:]
    M'[r,c] = exp(E[r]/2) M[r,c] exp(E[c]/2)

preserves every contraction term and makes the row scale exactly ``diag(M')``.
This removes W's nine parameters entirely while keeping sign-rank three.
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
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
TASK = 275
AUTHORITY_COST = 428

S = np.asarray(
    [
        [
            -0.2669769525527954,
            -0.3303088843822479,
            -0.3172421455383301,
            -0.3317573070526123,
            -0.33202359080314636,
            0.010246257297694683,
            0.005771744064986706,
            0.009609288536012173,
            0.9417421817779541,
            0.015958478674292564,
        ],
        [
            -2.8843271732330322,
            -1.3409913778305054,
            -0.10379832237958908,
            -0.8681402206420898,
            -0.46936851739883423,
            -0.015954937785863876,
            -0.06415707617998123,
            -0.06384441256523132,
            -2.955000638961792,
            -0.03332320228219032,
        ],
        [
            -0.04608163237571716,
            0.04851911589503288,
            0.16439864039421082,
            0.07800837606191635,
            0.11399004608392715,
            0.007513041608035564,
            0.0013371319510042667,
            0.0039466554298996925,
            -0.04572559893131256,
            0.010648796334862709,
        ],
    ],
    dtype=np.float32,
)

M = np.asarray(
    [
        [3.522934913635254, -2.704728126525879, 39.74148941040039],
        [6.042413234710693, -0.6691519021987915, -9.032179832458496],
        [2.3424601554870605, 3.673196315765381, -14.66893482208252],
    ],
    dtype=np.float32,
)


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
            initializer.CopyFrom(numpy_helper.from_array(array, name=name))
            return
    raise KeyError(name)


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [initializer for initializer in model.graph.initializer if initializer.name != name]
    if len(kept) == len(model.graph.initializer):
        raise KeyError(name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def reachable_quadruple_audit() -> dict[str, float | int | bool]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    diagonal = np.diag(M)
    for p in range(5):
        for b in (0, 8):
            for r in range(5):
                for q in (0, 8):
                    u = S[:, p] + S[:, b]
                    v = S[:, r] + S[:, q]
                    latent = np.square(u) * (M.T @ v)
                    rows.append((S.T @ (M * diagonal[:, None])) @ latent)
                    target = -np.ones(10, dtype=np.float32)
                    target[p if p > 0 and q == 8 else 0] = 1.0
                    targets.append(target)
    signed = np.asarray(rows) * np.asarray(targets)
    return {
        "quadruples": len(rows),
        "signed_cells": int(signed.size),
        "strictly_correct_cells": int(np.count_nonzero(signed > 0.0)),
        "minimum_signed_margin": float(signed.min()),
        "pass": bool(np.all(signed > 0.0)),
    }


def build() -> onnx.ModelProto:
    with zipfile.ZipFile(AUTHORITY) as archive:
        model = onnx.load_model_from_string(archive.read("task275.onnx"))
    model = copy.deepcopy(model)
    einsum = model.graph.node[-1]
    terms, output = equation(einsum).split("->")
    terms_list = terms.split(",")
    inputs = list(einsum.input)
    if terms_list[-2:] != ["ak", "al"] or inputs[-2:] != ["S", "W"]:
        raise RuntimeError((terms_list[-2:], inputs[-2:]))

    # M is already T in the third-color role.  Reuse it for W and append one
    # diagonal view; repeated ``aa`` is the standard Einsum diagonal syntax.
    inputs[-1] = "T"
    inputs.append("T")
    terms_list.append("aa")
    del einsum.input[:]
    einsum.input.extend(inputs)
    set_equation(einsum, ",".join(terms_list) + "->" + output)
    replace_initializer(model, "S", S)
    replace_initializer(model, "T", M)
    remove_initializer(model, "W")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> int:
    audit = reachable_quadruple_audit()
    if not audit["pass"]:
        raise RuntimeError(audit)
    model = build()
    data = model.SerializeToString()
    digest = hashlib.sha256(data).hexdigest()
    candidate = HERE / f"task275_diagonal_reuse_cost419_{digest[:12]}.onnx"
    candidate.write_bytes(data)
    payload = {
        "task": TASK,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_cost": AUTHORITY_COST,
        "candidate": str(candidate.relative_to(ROOT)),
        "candidate_sha256": digest,
        "theoretical_candidate_cost": 419,
        "theoretical_gain": math.log(AUTHORITY_COST / 419),
        "transformation": "reuse T as W and reuse diag(T) as W row scale",
        "reachable_quadruple_audit": audit,
        "parameter_counts": {
            "authority_color": 48,
            "candidate_color": 39,
            "saved": 9,
        },
    }
    (HERE / "diagonal_reuse_build_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
