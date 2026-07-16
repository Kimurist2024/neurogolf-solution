#!/usr/bin/env python3
"""Build exact nonnegative-Mul -> Selu memshave candidates from 8009.46.

For finite x >= 0, ONNX Selu(alpha=1, gamma=g) is exactly g*x in real
arithmetic, including x=0.  Moving a positive scalar from an initializer to
the Selu gamma attribute can therefore remove one parameter when every use of
the initializer is such a Mul.  This lane only encodes cases whose source
nonnegativity follows from the valid one-hot input domain and standard ONNX
operator semantics; numerical/raw equivalence is audited separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_DIR = Path("/private/tmp/ng800946_rank")
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


SPECS = {
    13: {
        "mode": "mul",
        "initializer": "half_h",
        "sources": ("p0x2_h",),
        "proof": (
            "valid task013 inputs contain two markers at coordinates a,b on the "
            "long axis; P=a+b and d=sqrt(2(a^2+b^2)-(a+b)^2)=|a-b|, "
            "so p0x2_h=P-d=2*min(a,b)>=0"
        ),
    },
    90: {
        "mode": "div",
        "initializer": "ln2",
        "sources": ("ln_lowbit", "ln_sel_run"),
        "proof": (
            "valid task090 inputs have a nonempty selected maximum rectangle; "
            "selected_run is a positive integer bitset and lowbit=run&-run is "
            "a positive power of two, so both logarithms are finite nonnegative"
        ),
    },
    134: {
        "mode": "mul",
        "initializer": "hInv29",
        "sources": ("rowq", "col", "m2"),
        "proof": (
            "rowq=floor(mega_local/30) and col=mega_local mod 30 with "
            "mega_local in [0,899]; m2 is a sum of nonnegative decoded values"
        ),
    },
    209: {
        "mode": "mul",
        "initializer": "fhalf16",
        "sources": ("ysr16", "ysc16"),
        "proof": (
            "ysr/ysc are Einsum sums of the nonnegative one-hot input, a 0/1 "
            "selector, and coordinate vector [0..29], followed by Cast; "
            "pclog logs pball&-pball, a positive power of two because each valid "
            "continuous creature has at least four pixels"
        ),
    },
    233: {
        "mode": "mul",
        "initializer": "qseq_256_f16",
        "sources": ("qseq_high_f16",),
        "proof": (
            "qseq_high_f16 is fmod=1 of nonnegative float16 values emitted by "
            "a uint8 QLinearConv rank pipeline, so its reachable values are "
            "nonnegative (in practice 0 or 1) and never negative zero"
        ),
    },
    366: {
        "mode": "mul",
        "initializer": "safe_name_19",
        "sources": ("safe_name_40",),
        "proof": (
            "safe_name_40 is Cast(Einsum(nonnegative one-hot input, coordinate "
            "vector [0..29])); safe_name_752/753 cast sums of nonnegative "
            "coordinate gathers and nonnegative bounded uint8/int8 offsets"
        ),
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scalar(model: onnx.ModelProto, name: str) -> float:
    for item in model.graph.initializer:
        if item.name == name:
            array = np.asarray(numpy_helper.to_array(item))
            if array.size != 1 or array.dtype.kind != "f":
                raise ValueError(f"{name} is not a scalar float")
            value = float(array.reshape(-1)[0])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} is not finite positive")
            return value
    raise ValueError(f"missing initializer {name}")


def rewrite(model: onnx.ModelProto, initializer: str, sources: tuple[str, ...]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    gamma = scalar(result, initializer)
    sources_left = set(sources)
    uses = []
    for index, node in enumerate(result.graph.node):
        for input_index, name in enumerate(node.input):
            if name == initializer:
                uses.append((index, input_index, node))
    if len(uses) != len(sources):
        raise ValueError(f"{initializer}: expected {len(sources)} uses, got {len(uses)}")
    replacements: dict[int, onnx.NodeProto] = {}
    for index, _input_index, node in uses:
        if node.op_type != "Mul" or len(node.input) != 2 or len(node.output) != 1:
            raise ValueError(f"{initializer}: non-Mul use {node.op_type}")
        source = node.input[0] if node.input[1] == initializer else node.input[1]
        if source not in sources_left:
            raise ValueError(f"{initializer}: unexpected source {source}")
        sources_left.remove(source)
        replacements[index] = helper.make_node(
            "Selu",
            [source],
            list(node.output),
            name=f"exact_nonnegative_{source}_scale",
            alpha=1.0,
            gamma=gamma,
        )
    if sources_left:
        raise ValueError(f"unused expected sources: {sorted(sources_left)}")
    nodes = [replacements.get(index, node) for index, node in enumerate(result.graph.node)]
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    kept = [item for item in result.graph.initializer if item.name != initializer]
    if len(kept) + 1 != len(result.graph.initializer):
        raise ValueError(f"initializer removal failed: {initializer}")
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(kept)
    return result


def rewrite_div(model: onnx.ModelProto, initializer: str, sources: tuple[str, ...]) -> onnx.ModelProto:
    """Replace Div(x,c) by Selu(x,gamma=1/c) for proven nonnegative x."""
    result = copy.deepcopy(model)
    divisor = scalar(result, initializer)
    gamma = 1.0 / divisor
    sources_left = set(sources)
    uses = []
    for index, node in enumerate(result.graph.node):
        for input_index, name in enumerate(node.input):
            if name == initializer:
                uses.append((index, input_index, node))
    if len(uses) != len(sources):
        raise ValueError(f"{initializer}: expected {len(sources)} uses, got {len(uses)}")
    replacements: dict[int, onnx.NodeProto] = {}
    for index, input_index, node in uses:
        if node.op_type != "Div" or input_index != 1 or len(node.input) != 2 or len(node.output) != 1:
            raise ValueError(f"{initializer}: non-denominator-Div use {node.op_type}:{input_index}")
        source = node.input[0]
        if source not in sources_left:
            raise ValueError(f"{initializer}: unexpected source {source}")
        sources_left.remove(source)
        replacements[index] = helper.make_node(
            "Selu",
            [source],
            list(node.output),
            name=f"exact_nonnegative_{source}_div_{initializer}",
            alpha=1.0,
            gamma=gamma,
        )
    if sources_left:
        raise ValueError(f"unused expected sources: {sorted(sources_left)}")
    nodes = [replacements.get(index, node) for index, node in enumerate(result.graph.node)]
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    kept = [item for item in result.graph.initializer if item.name != initializer]
    if len(kept) + 1 != len(result.graph.initializer):
        raise ValueError(f"initializer removal failed: {initializer}")
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(kept)
    return result


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


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> int:
    if digest(AUTHORITY_ZIP.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 authority ZIP changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, spec in SPECS.items():
        base_path = AUTHORITY_DIR / f"task{task:03d}.onnx"
        base = onnx.load(base_path)
        if spec["mode"] == "mul":
            candidate = rewrite(base, spec["initializer"], spec["sources"])
        elif spec["mode"] == "div":
            candidate = rewrite_div(base, spec["initializer"], spec["sources"])
        else:
            raise ValueError(spec["mode"])
        if task == 209:
            candidate = rewrite_div(candidate, "ln2", ("pclog",))
        if task == 366:
            candidate = rewrite(candidate, "safe_name_31", ("safe_name_752", "safe_name_753"))
        candidate_path = CANDIDATES / f"task{task:03d}.onnx"
        onnx.save(candidate, candidate_path)
        base_structure = structural(base)
        candidate_structure = structural(candidate)
        base_profile = profile(base_path)
        candidate_profile = profile(candidate_path)
        row = {
            "task": task,
            "authority_sha256": digest(base_path.read_bytes()),
            "candidate_sha256": digest(candidate_path.read_bytes()),
            "initializer": spec["initializer"],
            "sources": list(spec["sources"]),
            "nonnegative_proof": spec["proof"],
            "authority_structure": base_structure,
            "candidate_structure": candidate_structure,
            "authority_profile": base_profile,
            "candidate_profile": candidate_profile,
            "cost_reduction": base_profile["cost"] - candidate_profile["cost"],
            "strict_lower": candidate_profile["cost"] < base_profile["cost"],
        }
        rows.append(row)
        print(
            f"task{task:03d}: {base_profile['cost']} -> {candidate_profile['cost']} "
            f"structural={candidate_structure['pass']}"
        )
    payload = {
        "authority_zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_zip_sha256": AUTHORITY_SHA256,
        "rows": rows,
    }
    (HERE / "build.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
