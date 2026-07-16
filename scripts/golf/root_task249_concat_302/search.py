#!/usr/bin/env python3
"""Search parameter-free reuse of task249's existing predicate scalars."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.lib import scoring  # noqa: E402


def runtime(model: onnx.ModelProto) -> ort.InferenceSession | None:
    try:
        onnx.checker.check_model(model, full_check=True)
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        return ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return None


def examples() -> list[dict[str, np.ndarray]]:
    loaded = scoring.load_examples(249)
    return [
        converted
        for item in loaded["train"] + loaded["test"] + loaded["arc-gen"]
        if (converted := scoring.convert_to_numpy(item)) is not None
    ]


def exact(session: ort.InferenceSession, items: list[dict[str, np.ndarray]], limit: int | None) -> bool:
    for item in items if limit is None else items[:limit]:
        try:
            raw = session.run(["output"], {"input": item["input"]})[0]
        except Exception:
            return False
        if raw.shape != item["output"].shape or not np.array_equal(raw > 0, item["output"] > 0):
            return False
        if not np.isfinite(raw).all():
            return False
    return True


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "submission_base_8011.05.zip") as archive:
        base = onnx.load_model_from_string(archive.read("task249.onnx"))
    concat = next(node for node in base.graph.node if node.op_type == "Concat")
    assert list(concat.input) == ["tru", "fal", "fal", "b3", "b4", "b5"]
    items = examples()
    rows = []

    # Keep the three dynamic class slots as a permutation; replace the fixed
    # prefix with predicates where possible. This covers every zero-parameter
    # direct reuse without adding a counted intermediate.
    for prefix in itertools.product(("tru", "fal", "b3", "b4", "b5"), repeat=3):
        for suffix in itertools.permutations(("b3", "b4", "b5")):
            names = prefix + suffix
            if names == tuple(concat.input):
                continue
            model = copy.deepcopy(base)
            target = next(node for node in model.graph.node if node.op_type == "Concat")
            del target.input[:]
            target.input.extend(names)
            used = {name for node in model.graph.node for name in node.input if name}
            kept = [tensor for tensor in model.graph.initializer if tensor.name in used]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept)
            params = scoring.calculate_params(model)
            if params is None or params >= 4:
                continue
            sess = runtime(model)
            if sess is None or not exact(sess, items, 12):
                continue
            row = {"concat": names, "params": params, "quick": True, "full": exact(sess, items, None)}
            if row["full"]:
                path = HERE / f"task249_p{params}_{'_'.join(names)}.onnx"
                onnx.save(model, path)
                row["path"] = str(path.relative_to(ROOT))
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                row["profile"] = scoring.score_and_verify(
                    model, 249, str(HERE / "profiles"), path.stem, require_correct=True
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"tested": 125 * 6 - 1, "survivors": rows}
    (HERE / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
