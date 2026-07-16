#!/usr/bin/env python3
"""Build exact CastLike-to-Cast parameter shaves against LB 8009.46.

When every use of an initializer is the type-witness input of CastLike, its
values and shape are semantically irrelevant.  Replacing those CastLike nodes
with Cast(to=<initializer dtype>) makes the fixed target type an attribute and
allows the witness initializer to be removed.  The transformation is exact
for every input; runtime and raw-output audits are performed separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank")
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def structural(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
    try:
        findings = check_conv_bias(model)
        row["conv_bias_ub0"] = not findings
        row["conv_bias_findings"] = findings
    except Exception as exc:  # noqa: BLE001
        row.update(conv_bias_ub0=False, conv_bias_error=f"{type(exc).__name__}: {exc}")
    row["pass"] = bool(row.get("full_check") and row.get("strict_data_prop") and row.get("conv_bias_ub0"))
    return row


def profile(path: Path) -> dict[str, int] | dict[str, str]:
    try:
        memory, params, cost = cost_of(str(path))
        return {"memory": int(memory), "params": int(params), "cost": int(cost)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def exclusive_witnesses(model: onnx.ModelProto) -> dict[str, list[int]]:
    initializers = {item.name: item for item in model.graph.initializer}
    uses: dict[str, list[tuple[int, int]]] = {}
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            uses.setdefault(name, []).append((node_index, input_index))
    result: dict[str, list[int]] = {}
    for name in initializers:
        current = uses.get(name, [])
        if not current:
            continue
        if all(
            model.graph.node[node_index].op_type == "CastLike"
            and input_index == 1
            and len(model.graph.node[node_index].input) == 2
            and not model.graph.node[node_index].attribute
            for node_index, input_index in current
        ):
            result[name] = [node_index for node_index, _ in current]
    return result


def rewrite(model: onnx.ModelProto, witnesses: dict[str, list[int]]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    initializers = {item.name: item for item in result.graph.initializer}
    replacement: dict[int, onnx.NodeProto] = {}
    for name, indexes in witnesses.items():
        target_type = int(initializers[name].data_type)
        for index in indexes:
            node = result.graph.node[index]
            replacement[index] = helper.make_node(
                "Cast",
                [node.input[0]],
                list(node.output),
                name=node.name,
                to=target_type,
            )
    nodes = [replacement.get(index, node) for index, node in enumerate(result.graph.node)]
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    remove = set(witnesses)
    kept = [item for item in result.graph.initializer if item.name not in remove]
    if len(kept) + len(remove) != len(result.graph.initializer):
        raise ValueError("witness initializer removal count mismatch")
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(kept)
    return result


def main() -> int:
    if digest(AUTHORITY_ZIP.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 authority ZIP changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    for base_path in sorted(AUTHORITY.glob("task*.onnx")):
        model = onnx.load(base_path)
        witnesses = exclusive_witnesses(model)
        if not witnesses:
            continue
        task = int(base_path.stem[4:])
        candidate = rewrite(model, witnesses)
        candidate_path = CANDIDATES / base_path.name
        onnx.save(candidate, candidate_path)
        base_profile = profile(base_path)
        candidate_profile = profile(candidate_path)
        row = {
            "task": task,
            "authority_sha256": digest(base_path.read_bytes()),
            "candidate_sha256": digest(candidate_path.read_bytes()),
            "witnesses": [
                {
                    "name": name,
                    "dtype": onnx.TensorProto.DataType.Name(
                        next(item.data_type for item in model.graph.initializer if item.name == name)
                    ),
                    "castlike_nodes": len(indexes),
                }
                for name, indexes in witnesses.items()
            ],
            "authority_structure": structural(model),
            "candidate_structure": structural(candidate),
            "authority_profile": base_profile,
            "candidate_profile": candidate_profile,
        }
        if "cost" in base_profile and "cost" in candidate_profile:
            row["strict_lower"] = candidate_profile["cost"] < base_profile["cost"]
            row["cost_reduction"] = base_profile["cost"] - candidate_profile["cost"]
        else:
            row["strict_lower"] = False
        row["admit_for_runtime_audit"] = bool(row["strict_lower"] and row["candidate_structure"]["pass"])
        rows.append(row)
        print(
            f"task{task:03d}: witnesses={len(witnesses)} "
            f"cost={base_profile.get('cost')}->{candidate_profile.get('cost')} "
            f"structure={row['candidate_structure']['pass']}"
        )
    output = {
        "authority_zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "semantic_identity": "CastLike(x, fixed_initializer) == Cast(x, to=fixed_initializer.dtype)",
        "scanned_tasks": 400,
        "candidate_tasks": len(rows),
        "runtime_audit_tasks": [row["task"] for row in rows if row["admit_for_runtime_audit"]],
        "rows": rows,
    }
    (HERE / "build.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"runtime audit shortlist: {len(output['runtime_audit_tasks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
