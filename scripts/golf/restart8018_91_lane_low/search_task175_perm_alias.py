#!/usr/bin/env python3
"""Exhaust exact row-permutation gauge aliases between task175 S and Msel.

S and Msel differ only by swapping rows 0/1.  This search removes either one,
aliases all its Einsum uses to the survivor, and exhausts the compatible row
swap on every remaining initializer axis of extent three (R axis 1, selector
axis 0, and TA axis 0).  There are only 16 deterministic variants.  A variant
is retained only if full/static checking and official known gold both pass.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "candidates" / "task175_gauge_remove_w_v.onnx"
OUT = HERE / "perm_alias"
SOURCE_SHA256 = "acead77ce6b60ae5d5dd88e5c2c006cecdac6c9c5fd56bc97b56b37b72df8a1a"

sys.path.insert(0, str(ROOT / "scripts"))
from golf import try_candidate as try_mod  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def swap_axis(array: np.ndarray, axis: int) -> np.ndarray:
    indices = np.arange(array.shape[axis])
    indices[0], indices[1] = indices[1], indices[0]
    return np.take(array, indices, axis=axis).copy()


def build(alias_from: str, alias_to: str, mask: int) -> onnx.ModelProto:
    model = onnx.load_model_from_string(SOURCE.read_bytes())
    arrays = {item.name: np.asarray(numpy_helper.to_array(item))
              for item in model.graph.initializer}
    node = model.graph.node[0]
    replaced = 0
    for index, name in enumerate(node.input):
        if name == alias_from:
            node.input[index] = alias_to
            replaced += 1
    if not replaced:
        raise RuntimeError(f"no uses replaced for {alias_from}")

    selector = alias_to
    specs = (("R", 1), (selector, 0), ("TA", 0))
    replacements: dict[str, np.ndarray] = {}
    for bit, (name, axis) in enumerate(specs):
        if mask & (1 << bit):
            replacements[name] = swap_axis(arrays[name], axis)
    kept = []
    for item in model.graph.initializer:
        if item.name == alias_from:
            continue
        kept.append(numpy_helper.from_array(replacements[item.name], item.name)
                    if item.name in replacements else item)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.producer_name = f"codex-task175-alias-{alias_from}-to-{alias_to}-{mask:03b}"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> int:
    if sha256(SOURCE.read_bytes()) != SOURCE_SHA256:
        raise RuntimeError("cost-134 source drift")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for alias_from, alias_to in (("S", "Msel"), ("Msel", "S")):
        for mask in range(8):
            row: dict[str, object] = {
                "alias_from": alias_from,
                "alias_to": alias_to,
                "permutation_mask": mask,
                "swap_R_axis1": bool(mask & 1),
                "swap_selector_axis0": bool(mask & 2),
                "swap_TA_axis0": bool(mask & 4),
            }
            try:
                model = build(alias_from, alias_to, mask)
                gold, mismatch = try_mod._verify_gold(model, 175)
                row["official_gold_exact"] = bool(gold)
                row["first_mismatch"] = None if mismatch is None else {
                    "subset": mismatch.subset, "index": mismatch.index
                }
                if gold:
                    path = OUT / f"task175_{alias_from}_to_{alias_to}_{mask:03b}.onnx"
                    path.write_bytes(model.SerializeToString())
                    memory, params, cost = cost_of(str(path))
                    row.update({
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path.read_bytes()),
                        "memory": memory, "params": params, "cost": cost,
                    })
            except Exception as exc:
                row.update({"official_gold_exact": False, "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            print(json.dumps(row), flush=True)
    winners = [row for row in rows if row.get("official_gold_exact")]
    result = {"source": str(SOURCE.relative_to(ROOT)), "variants": len(rows),
              "winner_count": len(winners), "winners": winners, "rows": rows}
    (HERE / "task175_perm_alias_search.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"winner_count": len(winners), "winners": winners}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
