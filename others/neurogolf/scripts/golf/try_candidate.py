"""Validate, score, and optionally promote one handcrafted ONNX candidate."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib import scoring  # noqa: E402

OPTIMIZED_DIR = REPO_ROOT / "artifacts" / "optimized"
HANDCRAFTED_DIR = REPO_ROOT / "artifacts" / "handcrafted"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "run-012.json"
MARGIN = 0.25
BANNED_OPS = {
    "LOOP",
    "SCAN",
    "NONZERO",
    "UNIQUE",
    "SCRIPT",
    "FUNCTION",
    "COMPRESS",
}
ALLOWED_OPSET_DOMAINS = {"", "ai.onnx"}


@dataclass(frozen=True)
class ScoreInfo:
    cost: int
    score: float
    memory: int
    params: int


@dataclass(frozen=True)
class FirstMismatch:
    subset: str
    index: int
    expected_grid: list[list[int]]
    actual_mask: np.ndarray


def _score_for_cost(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def _mask_to_text(mask: np.ndarray, height: int, width: int) -> str:
    if mask.ndim != 4 or mask.shape[0] != 1:
        return f"<unexpected output shape {tuple(mask.shape)}>"

    rows: list[str] = []
    cropped = mask[0, :, :height, :width]
    for r in range(height):
        chars: list[str] = []
        for c in range(width):
            active = np.flatnonzero(cropped[:, r, c] > 0)
            if len(active) == 1:
                chars.append(str(int(active[0])))
            elif len(active) == 0:
                chars.append(".")
            else:
                chars.append("*")
        rows.append("".join(chars))
    return "\n".join(rows)


def _all_examples(task_num: int) -> list[tuple[str, int, dict[str, Any]]]:
    examples = scoring.load_examples(task_num)
    out: list[tuple[str, int, dict[str, Any]]] = []
    for subset in ("train", "test", "arc-gen"):
        for idx, example in enumerate(examples.get(subset, []), start=1):
            out.append((subset, idx, example))
    return out


def _validate_file_size(path: Path) -> bool:
    if not path.is_file():
        print(f"FAIL file: missing {_display_path(path)}")
        return False
    size = path.stat().st_size
    limit = int(scoring.FILESIZE_LIMIT_IN_BYTES)
    if size > scoring.FILESIZE_LIMIT_IN_BYTES:
        print(f"FAIL file-size: {size} bytes > {limit} bytes")
        return False
    print(f"PASS file-size: {size} bytes <= {limit} bytes")
    return True


def _validate_ops_and_shapes(model: onnx.ModelProto) -> bool:
    ok = True
    for opset in model.opset_import:
        if opset.domain not in ALLOWED_OPSET_DOMAINS:
            print(f"FAIL opset-domain: {opset.domain!r} is not allowed")
            ok = False

    if model.functions:
        print(f"FAIL functions: model contains {len(model.functions)} functions")
        ok = False

    for node in model.graph.node:
        op_upper = node.op_type.upper()
        if op_upper in BANNED_OPS or "Sequence" in node.op_type:
            print(f"FAIL banned-op: {node.op_type}")
            ok = False
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                print(f"FAIL nested-graph: node {node.name or node.op_type}")
                ok = False

    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL checker: {exc}")
        return False

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL shape-inference: {exc}")
        return False

    tensors = (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    )
    for tensor in tensors:
        ttype = tensor.type
        if ttype.HasField("sequence_type"):
            print(f"FAIL static-shape: sequence tensor {tensor.name}")
            ok = False
            continue
        if not ttype.HasField("tensor_type"):
            continue
        tensor_type = ttype.tensor_type
        if not tensor_type.HasField("shape"):
            print(f"FAIL static-shape: missing shape for {tensor.name}")
            ok = False
            continue
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                print(f"FAIL static-shape: dynamic dimension in {tensor.name}")
                ok = False
                break
            if dim.dim_value <= 0:
                print(f"FAIL static-shape: non-positive dimension in {tensor.name}")
                ok = False
                break

    if ok:
        print("PASS validator: ops, opsets, checker, and static shapes")
    return ok


def _verify_gold(model: onnx.ModelProto, task_num: int) -> tuple[bool, FirstMismatch | None]:
    session = scoring._make_raw_session(model)
    if session is None:
        print("FAIL gold: unable to create sanitized raw session")
        return False, None

    for subset, idx, example in _all_examples(task_num):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            # The official scorer (scoring.verify_subset) SKIPS examples whose
            # grids exceed 30x30 instead of failing them. Mirror that here so
            # tasks with oversized arc-gen fixtures (366/184/202) stay
            # promotable; the private set only scores grids within 30x30.
            print(f"SKIP gold: {subset}[{idx}] exceeds scorer grid limits (official scorer skips it)")
            continue
        try:
            raw = scoring._raw_output(session, benchmark["input"])
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL gold: inference error at {subset}[{idx}]: {exc}")
            return False, None
        actual = (raw > 0.0).astype(np.float32)
        if not np.array_equal(actual, benchmark["output"]):
            return (
                False,
                FirstMismatch(
                    subset=subset,
                    index=idx,
                    expected_grid=example["output"],
                    actual_mask=actual,
                ),
            )
    print("PASS gold: all train/test/arc-gen examples match")
    return True, None


def _print_mismatch(mismatch: FirstMismatch) -> None:
    height = len(mismatch.expected_grid)
    width = len(mismatch.expected_grid[0]) if mismatch.expected_grid else 0
    print(f"FAIL gold: first mismatch at {mismatch.subset}[{mismatch.index}]")
    print("expected:")
    print(_grid_to_text(mismatch.expected_grid))
    print("actual:")
    print(_mask_to_text(mismatch.actual_mask, height, width))


def _check_margin(model: onnx.ModelProto, task_num: int) -> tuple[bool, float | None]:
    session = scoring._make_raw_session(model)
    if session is None:
        print("FAIL margin: unable to create sanitized raw session")
        return False, None

    min_positive: float | None = None
    for subset, idx, example in _all_examples(task_num):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            # Mirror the official scorer (scoring.model_margin_stable), which
            # SKIPS grids over 30x30 instead of failing them. The private set
            # only scores grids within 30x30, so oversized arc-gen fixtures
            # must not block the margin gate.
            print(f"SKIP margin: {subset}[{idx}] exceeds scorer grid limits (official scorer skips it)")
            continue
        try:
            raw = scoring._raw_output(session, benchmark["input"])
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL margin: inference error at {subset}[{idx}]: {exc}")
            return False, min_positive
        positive = raw[raw > 0.0]
        if positive.size:
            ex_min = float(positive.min())
            min_positive = ex_min if min_positive is None else min(min_positive, ex_min)
        near = positive[positive < MARGIN]
        if near.size:
            print(
                "FAIL margin: raw output in open interval "
                f"(0, {MARGIN}) at {subset}[{idx}], min={float(near.min()):.8g}"
            )
            return False, min_positive

    observed = "none" if min_positive is None else f"{min_positive:.8g}"
    print(f"PASS margin: no raw output cell in (0, {MARGIN}); min positive={observed}")
    return True, min_positive


def _score_model(
    model: onnx.ModelProto,
    task_num: int,
    workdir: str,
    label: str,
    *,
    require_correct: bool,
) -> ScoreInfo | None:
    scored = scoring.score_and_verify(
        model,
        task_num,
        workdir,
        label=label,
        require_correct=require_correct,
    )
    if scored is None:
        return None
    return ScoreInfo(
        cost=int(scored["cost"]),
        score=float(scored["score"]),
        memory=int(scored["memory"]),
        params=int(scored["params"]),
    )


def _score_path(
    path: Path,
    task_num: int,
    workdir: str,
    label: str,
    *,
    require_correct: bool,
) -> ScoreInfo | None:
    if not path.is_file():
        return None
    return _score_model(
        onnx.load(str(path)),
        task_num,
        workdir,
        label,
        require_correct=require_correct,
    )


def _report_incumbent_cost(task_num: int) -> ScoreInfo | None:
    if not REPORT_PATH.is_file():
        return None
    with open(REPORT_PATH) as f:
        report = json.load(f)
    for item in report.get("tasks", []):
        if item.get("task") == task_num:
            cost = int(item["cost_post_002"])
            return ScoreInfo(
                cost=cost,
                score=float(item.get("score_post_002", _score_for_cost(cost))),
                memory=-1,
                params=-1,
            )
    return None


def _current_best(
    task_num: int,
    optimized: ScoreInfo,
    handcrafted: ScoreInfo | None,
) -> tuple[str, ScoreInfo]:
    if handcrafted is not None and handcrafted.cost < optimized.cost:
        return "handcrafted", handcrafted
    return "optimized", optimized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True, help="Task number 1..400")
    parser.add_argument("--onnx", type=Path, required=True, help="Candidate ONNX path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_num = args.task
    if not 1 <= task_num <= 400:
        raise SystemExit("--task must be in 1..400")

    candidate_path = args.onnx
    if not candidate_path.is_absolute():
        candidate_path = REPO_ROOT / candidate_path
    candidate_path = candidate_path.resolve()

    if not _validate_file_size(candidate_path):
        return 1

    try:
        candidate = onnx.load(str(candidate_path))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL load: {exc}")
        return 1

    if not _validate_ops_and_shapes(candidate):
        return 1

    gold_ok, mismatch = _verify_gold(candidate, task_num)
    if not gold_ok:
        if mismatch is not None:
            _print_mismatch(mismatch)
        return 1

    margin_ok, _ = _check_margin(candidate, task_num)
    if not margin_ok:
        return 1

    with tempfile.TemporaryDirectory(prefix="neurogolf_try_") as workdir:
        candidate_score = _score_model(
            candidate,
            task_num,
            workdir,
            "candidate",
            require_correct=True,
        )
        if candidate_score is None:
            print("FAIL score: scoring.score_and_verify rejected candidate")
            return 1

        optimized_path = OPTIMIZED_DIR / f"task{task_num:03d}.onnx"
        optimized_score = _score_path(
            optimized_path,
            task_num,
            workdir,
            "optimized",
            require_correct=False,
        )
        if optimized_score is None:
            optimized_score = _report_incumbent_cost(task_num)
        if optimized_score is None:
            print(f"FAIL incumbent: unable to score {_display_path(optimized_path)}")
            return 1

        hand_path = HANDCRAFTED_DIR / f"task{task_num:03d}.onnx"
        handcrafted_score = _score_path(
            hand_path,
            task_num,
            workdir,
            "handcrafted",
            require_correct=True,
        )

    hand_text = "missing"
    if handcrafted_score is not None:
        hand_text = str(handcrafted_score.cost)

    print(
        "PASS score: "
        f"cost={candidate_score.cost} "
        f"memory={candidate_score.memory} "
        f"params={candidate_score.params} "
        f"score={candidate_score.score:.6f}"
    )
    print(f"COMPARE optimized: cost={optimized_score.cost}")
    print(f"COMPARE handcrafted: cost={hand_text}")

    best_label, best_score = _current_best(
        task_num, optimized_score, handcrafted_score
    )
    if candidate_score.cost >= best_score.cost:
        print(
            "NOT PROMOTED: "
            f"candidate cost {candidate_score.cost} is not cheaper than "
            f"current best {best_score.cost} ({best_label})"
        )
        return 0

    HANDCRAFTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = HANDCRAFTED_DIR / f"task{task_num:03d}.onnx"
    shutil.copy2(candidate_path, dest)
    print(
        "PROMOTED: "
        f"{_display_path(candidate_path)} -> {_display_path(dest)} "
        f"({best_score.cost} -> {candidate_score.cost})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
