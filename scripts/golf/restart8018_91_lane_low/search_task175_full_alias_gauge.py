#!/usr/bin/env python3
"""Exhaust all axis-swap gauges after aliasing task175 S and Msel.

The cost-134 graph has only five remaining tensor families besides S/Msel.
After removing either selector, every initializer axis of extent 2, 3, or 4
can either retain its order or swap values 0/1.  Exhausting those 11 binary
choices for both alias directions covers 4,096 exact finite gauge candidates.
Candidates are screened on two deterministic known cases in parallel; only
survivors receive the complete official-gold gate and are serialized.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "candidates" / "task175_gauge_remove_w_v.onnx"
OUT = HERE / "full_alias_gauge"
SOURCE_SHA256 = "acead77ce6b60ae5d5dd88e5c2c006cecdac6c9c5fd56bc97b56b37b72df8a1a"

sys.path.insert(0, str(ROOT / "scripts"))
from golf import try_candidate as try_mod  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SOURCE_BLOB = SOURCE.read_bytes()
SOURCE_MODEL = onnx.load_model_from_string(SOURCE_BLOB)
BASE_ARRAYS = {item.name: np.asarray(numpy_helper.to_array(item))
               for item in SOURCE_MODEL.graph.initializer}


def swap01(array: np.ndarray, axis: int) -> np.ndarray:
    order = np.arange(array.shape[axis])
    order[0], order[1] = order[1], order[0]
    return np.take(array, order, axis=axis).copy()


def make_model(alias_from: str, alias_to: str, mask: int) -> onnx.ModelProto:
    model = onnx.load_model_from_string(SOURCE_BLOB)
    node = model.graph.node[0]
    replaced = 0
    for index, name in enumerate(node.input):
        if name == alias_from:
            node.input[index] = alias_to
            replaced += 1
    if not replaced:
        raise RuntimeError("alias source has no uses")

    kept_names = [item.name for item in model.graph.initializer if item.name != alias_from]
    axis_specs = [(name, axis) for name in kept_names
                  for axis, extent in enumerate(BASE_ARRAYS[name].shape)
                  if extent in (2, 3, 4)]
    if len(axis_specs) != 11:
        raise RuntimeError(f"expected 11 swappable axes, found {axis_specs}")
    replacements: dict[str, np.ndarray] = {name: BASE_ARRAYS[name].copy() for name in kept_names}
    for bit, (name, axis) in enumerate(axis_specs):
        if mask & (1 << bit):
            replacements[name] = swap01(replacements[name], axis)
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        numpy_helper.from_array(replacements[name], name) for name in kept_names
    )
    model.producer_name = f"codex-task175-full-alias-{alias_from}-{mask:04x}"
    return model


def raw_session(model: onnx.ModelProto) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_cases() -> list[dict[str, np.ndarray]]:
    rows = []
    examples = scoring.load_examples(175)["train"]
    for index in (0, 1):
        benchmark = scoring.convert_to_numpy(examples[index])
        if benchmark is None:
            raise RuntimeError("known example conversion failed")
        rows.append(benchmark)
    return rows


KNOWN = known_cases()


def screen(spec: tuple[str, str, int]) -> dict[str, object]:
    alias_from, alias_to, mask = spec
    row: dict[str, object] = {"alias_from": alias_from, "alias_to": alias_to, "mask": mask}
    try:
        model = make_model(alias_from, alias_to, mask)
        session = raw_session(model)
        for index, benchmark in enumerate(KNOWN):
            raw = session.run(["output"], {"input": benchmark["input"]})[0]
            if not np.all(np.isfinite(raw)):
                row.update({"pass": False, "reject": f"nonfinite:{index}"})
                return row
            if np.any((raw > 0.0) & (raw < 0.25)):
                row.update({"pass": False, "reject": f"small_positive:{index}"})
                return row
            if not np.array_equal(raw > 0.0, benchmark["output"] > 0.0):
                row.update({"pass": False, "reject": f"mismatch:{index}"})
                return row
        row["pass"] = True
        return row
    except Exception as exc:
        row.update({"pass": False, "reject": f"{type(exc).__name__}: {exc}"})
        return row


def main() -> int:
    if sha256(SOURCE_BLOB) != SOURCE_SHA256:
        raise RuntimeError("cost-134 source drift")
    specs = [
        (alias_from, alias_to, mask)
        for alias_from, alias_to in (("S", "Msel"), ("Msel", "S"))
        for mask in range(1 << 11)
    ]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for index, row in enumerate(executor.map(screen, specs), 1):
            rows.append(row)
            if row["pass"] or index % 256 == 0:
                print(json.dumps({"screened": index, "total": len(specs), **row}), flush=True)

    survivors = [row for row in rows if row["pass"]]
    OUT.mkdir(parents=True, exist_ok=True)
    winners = []
    for row in survivors:
        model = make_model(str(row["alias_from"]), str(row["alias_to"]), int(row["mask"]))
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        gold, mismatch = try_mod._verify_gold(model, 175)
        row["official_gold_exact"] = bool(gold)
        row["official_mismatch"] = None if mismatch is None else {
            "subset": mismatch.subset, "index": mismatch.index
        }
        if not gold:
            continue
        path = OUT / f"task175_{row['alias_from']}_to_{row['alias_to']}_{int(row['mask']):04x}.onnx"
        path.write_bytes(model.SerializeToString())
        memory, params, cost = cost_of(str(path))
        winner = {**row, "path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()),
                  "memory": memory, "params": params, "cost": cost}
        winners.append(winner)
        print(json.dumps({"winner": winner}), flush=True)

    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "variant_count": len(specs),
        "known_screen_survivors": len(survivors),
        "winner_count": len(winners),
        "winners": winners,
        "rows": rows,
    }
    (HERE / "task175_full_alias_gauge_search.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("variant_count", "known_screen_survivors", "winner_count", "winners")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
