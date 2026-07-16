#!/usr/bin/env python3
"""Finite prefix/suffix scan for tiny ConvTranspose authorities with NaN/Inf taps."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
COSTS = {164: 16, 172: 16, 210: 16, 311: 16, 322: 20, 116: 23, 372: 13}


def load_support():
    spec = importlib.util.spec_from_file_location("finite_conv_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_support()


def attr_ints(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    item = next((attr for attr in node.attribute if attr.name == name), None)
    return list(item.ints) if item is not None else default


def set_attr_ints(node: onnx.NodeProto, name: str, values: list[int]) -> None:
    item = next(attr for attr in node.attribute if attr.name == name)
    del item.ints[:]
    item.ints.extend(values)


def exact_row(row: dict[str, object]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and not row.get("session_error")
    )


def main() -> int:
    rows = []
    best = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, authority_cost in COSTS.items():
            base = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
            if len(base.graph.node) != 1 or base.graph.node[0].op_type != "ConvTranspose":
                raise RuntimeError(f"task{task:03d}: unexpected authority graph")
            node = base.graph.node[0]
            init = next(item for item in base.graph.initializer if item.name in {"x", "X"})
            array = numpy_helper.to_array(init)
            long_axes = [axis for axis in (2, 3) if array.shape[axis] > 1]
            if len(long_axes) != 1:
                raise RuntimeError(f"task{task:03d}: expected one long spatial axis")
            axis = long_axes[0]
            spatial = axis - 2
            length = array.shape[axis]
            pads = attr_ints(node, "pads", [0, 0, 0, 0])
            strides = attr_ints(node, "strides", [1, 1])
            cases, known_counts = SUPPORT.known_cases(task)
            quick_cases = cases[: min(12, len(cases))]
            task_best = {"right": -1, "label": None, "cost": None}
            for start in range(length):
                for stop in range(start + 1, length + 1):
                    if stop - start >= length:
                        continue
                    new_pads = list(pads)
                    new_pads[spatial] -= start * strides[spatial]
                    new_pads[spatial + 2] -= (length - stop) * strides[spatial]
                    if min(new_pads) < 0:
                        continue
                    sliced = np.take(array, range(start, stop), axis=axis)
                    for nan_value, inf_value in itertools.product(
                        (0.0, -1.0, -100.0, -10_000.0),
                        (1.0, 100.0, 10_000.0),
                    ):
                        finite = np.nan_to_num(
                            sliced, nan=nan_value, posinf=inf_value, neginf=-inf_value
                        ).astype(np.float32)
                        model = copy.deepcopy(base)
                        target = next(
                            item for item in model.graph.initializer if item.name == init.name
                        )
                        target.CopyFrom(numpy_helper.from_array(finite, init.name))
                        set_attr_ints(model.graph.node[0], "pads", new_pads)
                        data = model.SerializeToString()
                        structure = SUPPORT.structural_audit(task, model, data)
                        if not structure["pass"]:
                            continue
                        trace = structure["runtime_intermediate_trace"]
                        cost = structure["initializer_elements"] + trace["single_example_intermediate_bytes"]
                        if cost >= authority_cost:
                            continue
                        try:
                            run = SUPPORT.make_session(data, True, 1)
                            quick, _ = SUPPORT.evaluate_config(run, quick_cases, None)
                            if exact_row(quick):
                                known, _ = SUPPORT.evaluate_config(run, cases, None)
                            else:
                                known = {
                                    **quick,
                                    "screen_only": True,
                                    "screen_total": len(quick_cases),
                                }
                        except Exception as exc:  # noqa: BLE001
                            known = {
                                "total": len(cases), "right": 0, "wrong": 0,
                                "errors": len(cases), "session_error": f"{type(exc).__name__}: {exc}",
                                "nonfinite_cases": 0, "runtime_shape_mismatches": 0,
                                "small_positive_elements_0_to_0_25": 0,
                            }
                        label = (
                            f"crop{start}_{stop}_nan{nan_value:g}_inf{inf_value:g}"
                        )
                        row = {
                            "task": task, "label": label, "cost": cost,
                            "authority_cost": authority_cost,
                            "known_right": known.get("right"),
                            "known_wrong": known.get("wrong"),
                            "known_errors": known.get("errors"),
                            "known_nonfinite_cases": known.get("nonfinite_cases"),
                            "known_shape_mismatches": known.get("runtime_shape_mismatches"),
                            "known_small_positive": known.get("small_positive_elements_0_to_0_25"),
                            "known_exact": exact_row(known),
                            "screen_only": bool(known.get("screen_only")),
                            "known_counts": known_counts,
                        }
                        rows.append(row)
                        if int(known.get("right", 0)) > task_best["right"]:
                            task_best = {"right": int(known.get("right", 0)), "label": label, "cost": cost}
                        if row["known_exact"]:
                            four = SUPPORT.evaluate_four(data, cases)
                            row["four_exact"] = all(exact_row(value) for value in four.values())
                            row["four_summary"] = {
                                name: {
                                    key: value.get(key) for key in (
                                        "total", "right", "wrong", "errors", "nonfinite_cases",
                                        "runtime_shape_mismatches", "small_positive_elements_0_to_0_25",
                                    )
                                }
                                for name, value in four.items()
                            }
                            if row["four_exact"]:
                                out = HERE / "candidates" / f"task{task:03d}_{label}_cost{cost}.onnx"
                                out.parent.mkdir(parents=True, exist_ok=True)
                                out.write_bytes(data)
                                row["candidate_path"] = str(out.relative_to(ROOT))
            best[str(task)] = task_best
            print(json.dumps({"task": task, "best": task_best}), flush=True)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "method": "finite cropped ConvTranspose data initializer scan",
        "best_by_task": best,
        "exact_candidates": [row for row in rows if row.get("four_exact")],
        "attempts": rows,
    }
    (HERE / "finite_conv_scan.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
