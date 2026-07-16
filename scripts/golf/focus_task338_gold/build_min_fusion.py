#!/usr/bin/env python3
"""Algebraically fuse binary PRelu chains in the task338 authority graph."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = import_path(
    "task338_min_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
SUPPORT.THRESHOLD = 1.0
SUPPORT.FRESH_PER_SEED = 2_000
SUPPORT.SUPPORT.POLICY_THRESHOLD = 1.0
SUPPORT.SUPPORT.FRESH_PER_SEED = 2_000


def hard_not(source: str, target: str, name: str) -> onnx.NodeProto:
    return helper.make_node(
        "HardSigmoid", [source], [target], name=name, alpha=-1.0, beta=1.0
    )


def build(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    replacements: dict[int, list[onnx.NodeProto]] = {
        24: [
            helper.make_node(
                "Min", ["red", "red_up_mb", "red_down_pb"], ["vert_prelu1"],
                name="vertical_all",
            ),
            helper.make_node(
                "Selu", ["vert_prelu1"], ["vert_prelu2"],
                name="vertical_negative", alpha=1.0, gamma=-1.0,
            ),
        ],
        26: [
            helper.make_node(
                "Min", ["red", "not_up", "not_down"], ["horiz"],
                name="horizontal_all",
            )
        ],
        98: [
            helper.make_node(
                "Min",
                ["L1_s7_not", "L2_s7_not", "L3_s7_not", "L4_s7_not",
                 "L5_s7_not", "L6_mb", "L7_mb"],
                ["L_s7_prelu7"], name="left_all_absent",
            ),
            hard_not("L_s7_prelu7", "L7_seen", "left_seen"),
        ],
        125: [
            helper.make_node(
                "Min",
                ["R1_s7_not", "R2_pb", "R3_pb", "R4_pb", "R5_pb",
                 "R6_pb", "R7_pb"],
                ["R_s7_prelu7"], name="right_all_absent",
            ),
            hard_not("R_s7_prelu7", "R7_seen", "right_seen"),
        ],
        142: [
            helper.make_node(
                "Min",
                ["U1_s7_not", "U2_s7_not", "U3_s7_not", "U4_s7_not",
                 "U5_s7_not", "U6_s7_not", "U7_mb"],
                ["U_s7_prelu7"], name="up_all_absent",
            ),
            hard_not("U_s7_prelu7", "U7_seen", "up_seen"),
        ],
        150: [
            helper.make_node(
                "Min",
                ["D1_pb", "D2_pb", "D3_pb", "D4_pb", "D5_pb", "D6_pb",
                 "D7_pb"],
                ["D_s7_prelu7"], name="down_all_absent",
            ),
            hard_not("D_s7_prelu7", "D7_seen", "down_seen"),
        ],
        160: [
            helper.make_node(
                "Min", ["U6_mb", "D1_raw", "D4_raw", "D7_pb"],
                ["sup_prelu3"], name="support_all",
            ),
            hard_not("sup_prelu3", "sup", "support_seen"),
        ],
        166: [
            helper.make_node(
                "Min",
                ["valid", "not_red", "L7_seen", "R7_seen", "U7_seen",
                 "D7_seen", "sup"],
                ["cand"], name="candidate_all",
            )
        ],
    }
    removed = {
        15,
        24, 25,
        26, 27, 28,
        *range(98, 106),
        *range(125, 133),
        *range(142, 150),
        *range(150, 158),
        *range(160, 165),
        *range(166, 173),
    }
    new_nodes = []
    for index, node in enumerate(base.graph.node):
        if index in replacements:
            new_nodes.extend(replacements[index])
        if index not in removed:
            new_nodes.append(copy.deepcopy(node))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    produced = {name for node in model.graph.node for name in node.output if name}
    kept_vi = [item for item in model.graph.value_info if item.name in produced]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    return model


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("early_reject_reason") is None
    )


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        base = onnx.load_model_from_string(archive.read("task338.onnx"))
    model = build(base)
    data = model.SerializeToString()
    sha = hashlib.sha256(data).hexdigest()
    path = HERE / f"task338_min_fused_{sha[:12]}.onnx"
    path.write_bytes(data)
    cases, counts = SUPPORT.SUPPORT.known_cases(338)
    profile = SUPPORT.POLICY.fast_profile(SUPPORT.SUPPORT, 338, model, cases[0])
    known = SUPPORT.failfast_known(data, cases)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    fresh = []
    for seed in (338_900_001, 338_900_002):
        fresh_cases, generation = SUPPORT.SUPPORT.fresh_cases(338, seed, task_map)
        runtime = SUPPORT.failfast_known(data, fresh_cases)
        fresh.append(
            {
                "seed": seed,
                "generation": generation,
                "runtime": runtime,
                "pass": exact(runtime),
            }
        )
    payload = {
        "task": 338,
        "authority_cost": 403,
        "candidate_path": str(path.relative_to(ROOT)),
        "candidate_sha256": sha,
        "node_count": len(model.graph.node),
        "profile": profile,
        "known_counts": counts,
        "known": known,
        "known_pass": exact(known),
        "fresh": fresh,
        "fresh_pass": all(item["pass"] for item in fresh),
        "method": "binary-mask algebra: PRelu products -> variadic Min",
    }
    (HERE / "min_fusion_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
