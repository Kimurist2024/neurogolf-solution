#!/usr/bin/env python3
"""Replace initializer-anchored CastLike nodes by equivalent Cast nodes.

CastLike uses its second input only to select a dtype.  When that input is a
dense initializer, the dtype is statically known and can be encoded as Cast's
``to`` attribute.  Any initializer used solely as such a dtype anchor is then
removed.  No tensor values, shapes, or runtime operations other than the dtype
selection mechanism change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def rewrite(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    result = copy.deepcopy(model)
    initializers = {tensor.name: tensor for tensor in result.graph.initializer}
    conversions: list[dict[str, object]] = []
    for index, node in enumerate(result.graph.node):
        if node.op_type != "CastLike" or len(node.input) != 2:
            continue
        reference = initializers.get(node.input[1])
        if reference is None:
            continue
        if node.attribute:
            # Current competition models use no CastLike attributes.  Refuse
            # future float8/saturate variants instead of guessing semantics.
            continue
        reference_name = reference.name
        node.op_type = "Cast"
        del node.input[1:]
        node.attribute.append(onnx.helper.make_attribute("to", int(reference.data_type)))
        conversions.append(
            {
                "node_index": index,
                "reference": reference_name,
                "to": int(reference.data_type),
            }
        )

    if not conversions:
        return result, {"conversions": [], "removed_initializers": [], "saved_params": 0}

    uses = Counter(name for node in result.graph.node for name in node.input if name)
    protected = {value.name for value in result.graph.input} | {
        value.name for value in result.graph.output
    }
    removed: list[dict[str, object]] = []
    kept: list[onnx.TensorProto] = []
    for tensor in result.graph.initializer:
        if uses[tensor.name] == 0 and tensor.name not in protected:
            elements = math.prod(tensor.dims) if tensor.dims else 1
            removed.append({"name": tensor.name, "elements": int(elements)})
        else:
            kept.append(tensor)
    del result.graph.initializer[:]
    result.graph.initializer.extend(kept)

    onnx.checker.check_model(result, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(result, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    return result, {
        "conversions": conversions,
        "removed_initializers": removed,
        "saved_params": sum(int(item["elements"]) for item in removed),
    }


def repository_cost(path: Path) -> tuple[int, int, int]:
    from scripts.golf.rank_dir import cost_of

    memory, params, cost = cost_of(str(path))
    return int(memory), int(params), int(cost)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with zipfile.ZipFile(baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            try:
                source = onnx.load_model_from_string(archive.read(member))
                candidate, change = rewrite(source)
                if not change["removed_initializers"]:
                    continue
                with tempfile.TemporaryDirectory(prefix=f"castlike_{task:03d}_") as tmp:
                    source_path = Path(tmp) / "source.onnx"
                    candidate_path = Path(tmp) / "candidate.onnx"
                    onnx.save(source, source_path)
                    onnx.save(candidate, candidate_path)
                    base_memory, base_params, base_cost = repository_cost(source_path)
                    memory, params, cost = repository_cost(candidate_path)
                if cost >= base_cost:
                    continue
                path = out_dir / member
                onnx.save(candidate, path)
                rows.append(
                    {
                        "task": task,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "base_memory": base_memory,
                        "base_params": base_params,
                        "base_cost": base_cost,
                        "candidate_memory": memory,
                        "candidate_params": params,
                        "candidate_cost": cost,
                        "cost_reduction": base_cost - cost,
                        "projected_gain": math.log(base_cost / cost),
                        "change": change,
                    }
                )
            except Exception as error:
                failures.append({"task": task, "error": repr(error)})

    document = {
        "baseline": str(baseline.relative_to(ROOT)),
        "rows": rows,
        "winner_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "failures": failures,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
