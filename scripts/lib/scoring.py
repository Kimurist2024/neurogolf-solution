"""Faithful port of the official NeuroGolf 2026 scorer.

Ported from ``inputs/neurogolf-2026/neurogolf_utils/neurogolf_utils.py`` with
the scoring/verification logic preserved byte-for-byte in behaviour. The only
deliberate differences from the official module are:

* No IPython / matplotlib / onnx_tool imports (display-only helpers dropped).
* The data directory is repointed from the Kaggle path to the local
  ``inputs/neurogolf-2026/`` tree.

Nothing here changes scoring behaviour. The arithmetic, rejection conditions
and tensor-selection rules are identical to the official implementation.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import tempfile
import traceback
import uuid
from typing import Any

import numpy as np
import onnx
import onnxruntime
from onnxruntime.capi import onnxruntime_pybind11_state as _ort_state

# ONNX Runtime raises a family of exceptions that all subclass ``Exception``
# directly (there is no shared ORT base). The official module references
# ``onnxruntime.ONNXRuntimeError`` which is not exported in onnxruntime 1.24.x,
# so we catch the concrete pybind exception classes here to preserve the
# intended "swallow ORT runtime/load errors" behaviour.
_ORT_ERRORS: tuple[type[BaseException], ...] = tuple(
    obj
    for name in dir(_ort_state)
    if isinstance(obj := getattr(_ort_state, name), type)
    and issubclass(obj, Exception)
)

# --- Constants (mirrored from neurogolf_utils.py) ----------------------------

_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH = 1, 10, 30, 30
_EXCLUDED_OP_TYPES = [
    "LOOP",
    "SCAN",
    "NONZERO",
    "UNIQUE",
    "SCRIPT",
    "FUNCTION",
    "COMPRESS",
]
FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
_GRID_SHAPE = [_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH]
_MAX_GRID_DIMENSION = 30

# Local data directory holding task001.json .. task400.json.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_NEUROGOLF_DIR = str(_REPO_ROOT / "inputs" / "neurogolf-2026") + os.sep


# --- Core scoring (ported verbatim in behaviour) -----------------------------


def calculate_memory(model: onnx.ModelProto, trace_path: str) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None
    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [
                onnx.AttributeProto.GRAPH,
                onnx.AttributeProto.GRAPHS,
            ]:
                return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)
    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, Any] = {}
    tensor_map = {
        t.name: t
        for t in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in ["input", "output"]:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize
        tensor_dtypes[tensor_name] = np_dtype

    # Defensive check to verify uniqueness.
    seen: set[str] = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    # Retrieve actual tensor shapes via the ONNX Runtime Profiler's JSON Trace.
    with open(trace_path, "r") as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], mem)
    return sum(tensor_memory.values())


def check_network(filename: str) -> bool:
    file_path = pathlib.Path(filename)
    if not file_path.is_file():
        print(f"Error: File {filename} does not exist.")
        return False
    if (filesize := file_path.stat().st_size) > FILESIZE_LIMIT_IN_BYTES:
        print(f"Error: Filesize {filesize} exceeds {FILESIZE_LIMIT_IN_BYTES}.")
        return False
    return True


def convert_to_numpy(example: dict[str, Any]) -> dict[str, np.ndarray] | None:
    benchmark: dict[str, np.ndarray] = {}
    example_shape = (1, _CHANNELS, _HEIGHT, _WIDTH)
    for mode in ["input", "output"]:
        benchmark[mode] = np.zeros(example_shape, dtype=np.float32)
        grid = example[mode]
        if max(len(grid), len(grid[0])) > _MAX_GRID_DIMENSION:
            return None
        for r, _ in enumerate(grid):
            for c, color in enumerate(grid[r]):
                benchmark[mode][0][color][r][c] = 1.0
    return benchmark


def calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(d <= 0 for d in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d <= 0 for d in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                if any(d <= 0 for d in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def score_network(
    sanitized: onnx.ModelProto, trace_path: str
) -> tuple[int | None, int | None]:
    for node in sanitized.graph.node:
        if node.op_type.upper() in _EXCLUDED_OP_TYPES:
            print(f"Error: Op type {node.op_type} is not permitted.")
            return None, None
        if "Sequence" in node.op_type:
            print(f"Error: Op type {node.op_type} is not permitted.")
            return None, None
    return calculate_memory(sanitized, trace_path), calculate_params(sanitized)


def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto | None:
    for node in model.graph.node:
        node.name = node.output[0]
        if "kernel_time" in node.output[0]:
            return None

    name_map: dict[str, str] = {}
    counter = 0

    def get_safe_name(old_name: str) -> str:
        nonlocal counter
        if not old_name or old_name in ["input", "output"]:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for inp in model.graph.input:
        inp.name = get_safe_name(inp.name)
    for init in model.graph.initializer:
        init.name = get_safe_name(init.name)

    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = get_safe_name(node.input[i])
        for i in range(len(node.output)):
            node.output[i] = get_safe_name(node.output[i])
        if len(node.output) > 0 and node.output[0]:
            node.name = node.output[0]

    for out in model.graph.output:
        out.name = get_safe_name(out.name)
    for vi in model.graph.value_info:
        vi.name = get_safe_name(vi.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def load_examples(task_num: int) -> dict[str, Any]:
    """Loads relevant data from ARC-AGI and ARC-GEN."""
    with open(_NEUROGOLF_DIR + f"task{task_num:03d}.json") as f:
        examples = json.load(f)
    return examples


def run_network(
    session: onnxruntime.InferenceSession, benchmark_input: np.ndarray
) -> np.ndarray:
    result = session.run(["output"], {"input": benchmark_input})
    return (result[0] > 0.0).astype(float)


def verify_subset(
    session: onnxruntime.InferenceSession, example_subset: list[dict[str, Any]]
) -> tuple[int, int, dict[str, Any] | None]:
    right, wrong, expected, error = 0, 0, None, ""
    for example in example_subset:
        benchmark = convert_to_numpy(example)
        if not benchmark:
            continue
        try:
            user_output = run_network(session, benchmark["input"])
            if np.array_equal(user_output, benchmark["output"]):
                right += 1
            else:
                expected = example
                wrong += 1
        except _ORT_ERRORS:
            error = traceback.format_exc()
            wrong += 1
    if error:
        print(f"Error: {error}")
    return right, wrong, expected


# --- Convenience wrapper ------------------------------------------------------


def score_and_verify(
    model: onnx.ModelProto,
    task_num: int,
    workdir: str,
    label: str = "cand",
    require_correct: bool = True,
) -> dict[str, Any] | None:
    """Save, validate, verify and score a candidate model.

    Mirrors the official ``verify_network`` pipeline: writes a temp ONNX file,
    enforces the file-size limit, sanitizes names, creates an ORT session with
    profiling disabled-optimizations, verifies every example subset for exact
    correctness, then scores memory/params. Uses a unique profile prefix per
    call so concurrent/repeated runs never clobber each other's traces.

    Returns ``{"memory", "params", "cost", "score", "correct"}`` on success, or
    ``None`` on any rejection (size, load failure, unscorable, or incorrect).
    Cleans up the temp ONNX file and the profiling JSON trace before returning.

    When ``require_correct`` is ``False`` the gold-match correctness requirement
    is dropped: the model is still rejected on file size, load failure,
    sanitize failure, ``None``/negative memory or params, but a gold mismatch no
    longer forces ``None``. The returned ``correct`` field reflects the actual
    gold-match outcome regardless of this flag. This is used for locally
    divergent tasks where the macOS arm64 ``> 0.0`` boundary disagrees with the
    official Linux grader; safety is then guaranteed by a separate
    bit-identity check against the original model rather than by gold match.
    The default ``True`` preserves all existing behaviour unchanged.
    """
    os.makedirs(workdir, exist_ok=True)
    unique = f"{task_num:03d}_{label}_{uuid.uuid4().hex[:8]}"
    onnx_path = os.path.join(workdir, f"task{unique}.onnx")
    trace_path: str | None = None

    try:
        onnx.save(model, onnx_path)
        if not check_network(onnx_path):
            return None

        try:
            sanitized = sanitize_model(onnx.load(onnx_path))
            if not sanitized:
                return None
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            # H1 (proposal 002): force single-threaded execution so local
            # verification is deterministic across runs (multi-provider ORT
            # threading otherwise flips the `> 0.0` boundary on some tasks).
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            # Profiler writes <prefix>_<timestamp>.json into the CWD, so anchor
            # the prefix inside workdir with a unique label per call.
            options.profile_file_prefix = os.path.join(workdir, unique)
            session = onnxruntime.InferenceSession(
                sanitized.SerializeToString(), options
            )
        except _ORT_ERRORS as exc:
            print(f"Error: Unable to load ONNX model: {exc}")
            return None

        examples = load_examples(task_num)
        agi_right, agi_wrong, _ = verify_subset(
            session, examples["train"] + examples["test"]
        )
        gen_right, gen_wrong, _ = verify_subset(session, examples["arc-gen"])
        correct = (agi_wrong + gen_wrong) == 0

        trace_path = session.end_profiling()
        memory, params = score_network(sanitized, trace_path)
        if memory is None or params is None:
            return None
        if memory < 0 or params < 0:
            return None
        if require_correct and not correct:
            return None

        cost = memory + params
        score = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return {
            "memory": int(memory),
            "params": int(params),
            "cost": int(cost),
            "score": float(score),
            "correct": correct,
        }
    finally:
        _safe_remove(onnx_path)
        if trace_path is not None:
            _safe_remove(trace_path)


def _make_raw_session(
    model: onnx.ModelProto,
) -> onnxruntime.InferenceSession | None:
    """Create a sanitized ORT session with all graph optimizations disabled.

    Mirrors the session setup in ``score_and_verify`` (sanitize names, then an
    ``ORT_DISABLE_ALL`` session) but without profiling — only the raw forward
    outputs are needed. Returns ``None`` if the model cannot be sanitized or
    loaded.
    """
    sanitized = sanitize_model(copy.deepcopy(model))
    if not sanitized:
        return None
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    # H1 (proposal 002): single-threaded for deterministic raw outputs.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        return onnxruntime.InferenceSession(
            sanitized.SerializeToString(), options
        )
    except _ORT_ERRORS:
        return None


def _raw_output(
    session: onnxruntime.InferenceSession, benchmark_input: np.ndarray
) -> np.ndarray:
    """Run the network and return the RAW float output before the > 0 threshold."""
    return session.run(["output"], {"input": benchmark_input})[0]


def outputs_bit_identical(
    model_a: onnx.ModelProto,
    model_b: onnx.ModelProto,
    task_num: int,
) -> bool:
    """Return True iff two models produce bit-identical raw outputs everywhere.

    Both models are run through sanitized ``ORT_DISABLE_ALL`` sessions (no
    profiling) on the inputs of every example for the task — train, test and
    arc-gen — skipping pairs whose grids exceed 30x30 exactly as
    ``convert_to_numpy`` does. The comparison is on the RAW float outputs
    (``session.run`` result, BEFORE the ``> 0.0`` threshold), required to be
    ``np.array_equal`` for every example.

    This is platform-difference-proof: it does not depend on the gold table or
    on the sign of the threshold, so a model that is numerically identical to
    the original is accepted even on tasks where the local ``> 0.0`` boundary
    diverges from the official Linux grader. Returns ``False`` if either session
    cannot be built, if any inference errors, or if any output differs.
    """
    sess_a = _make_raw_session(model_a)
    if sess_a is None:
        return False
    sess_b = _make_raw_session(model_b)
    if sess_b is None:
        return False

    examples = load_examples(task_num)
    all_examples = (
        list(examples.get("train", []))
        + list(examples.get("test", []))
        + list(examples.get("arc-gen", []))
    )

    compared_any = False
    for example in all_examples:
        benchmark = convert_to_numpy(example)
        if not benchmark:
            continue
        try:
            out_a = _raw_output(sess_a, benchmark["input"])
            out_b = _raw_output(sess_b, benchmark["input"])
        except _ORT_ERRORS:
            return False
        if not np.array_equal(out_a, out_b):
            return False
        compared_any = True

    return compared_any


def masks_equal_with_margin(
    model_a: onnx.ModelProto,
    model_b: onnx.ModelProto,
    task_num: int,
    margin: float = 0.25,
) -> bool:
    """Verify a converted model (``model_b``) is safe vs the original (``model_a``).

    The G1/G2 dtype passes change raw numeric values but must not change the
    final thresholded one-hot mask, and must keep a safety margin so a
    platform-dependent ``> 0.0`` boundary cannot flip. For every example of the
    task (train + test + arc-gen, skipping grids over 30x30 exactly as
    ``convert_to_numpy`` does), this requires BOTH:

    * the thresholded masks ``(raw > 0.0)`` of ``model_a`` and ``model_b`` are
      identical (``np.array_equal``), and
    * the boundary safety margin holds: NO converted-model output cell has a raw
      value strictly inside the open interval ``(0, margin)`` (default 0.25).

    The margin guards against a platform-dependent ``> 0.0`` sign flip. A cell
    that is exactly ``0.0`` (a deterministically cleared/masked background cell,
    which is the overwhelming majority of the one-hot grid) is NOT a flip risk:
    an exact zero thresholds to ``False`` identically on every platform. Only a
    cell sitting just above zero (``0 < raw < margin``) could round across the
    boundary, so the check rejects exactly those near-boundary cells while
    allowing the exact-zero background. "On" cells are well clear of zero
    (typically ``|raw| >= 1``), so a clean conversion passes.

    Returns ``False`` if either session cannot be built, if any inference
    errors, if any mask differs, or if any converted-model cell lies in
    ``(0, margin)``. Returns ``False`` when there are no comparable examples.
    """
    sess_a = _make_raw_session(model_a)
    if sess_a is None:
        return False
    sess_b = _make_raw_session(model_b)
    if sess_b is None:
        return False

    examples = load_examples(task_num)
    all_examples = (
        list(examples.get("train", []))
        + list(examples.get("test", []))
        + list(examples.get("arc-gen", []))
    )

    compared_any = False
    for example in all_examples:
        benchmark = convert_to_numpy(example)
        if not benchmark:
            continue
        try:
            out_a = _raw_output(sess_a, benchmark["input"])
            out_b = _raw_output(sess_b, benchmark["input"])
        except _ORT_ERRORS:
            return False
        mask_a = out_a > 0.0
        mask_b = out_b > 0.0
        if not np.array_equal(mask_a, mask_b):
            return False
        abs_b = np.abs(out_b)
        # Near-boundary cells: strictly between 0 and the margin. Exact zeros
        # (background) are excluded — they cannot flip sign across platforms.
        if abs_b.size and bool(np.any((abs_b > 0.0) & (abs_b < margin))):
            return False
        compared_any = True

    return compared_any


def model_margin_stable(
    model: onnx.ModelProto,
    task_num: int,
    margin: float = 0.25,
) -> tuple[bool, float | None]:
    """Single-model boundary-stability check (no reference model).

    Runs ``model`` through a sanitized ``ORT_DISABLE_ALL`` session on every
    example of the task (train + test + arc-gen, skipping grids over 30x30 like
    ``convert_to_numpy``) and inspects the RAW float outputs (before the
    ``> 0.0`` threshold). The model is "margin-stable" iff NO raw output cell
    lies strictly inside the open interval ``(0, margin)`` on any example. An
    exact ``0.0`` cell is allowed (it thresholds to ``False`` identically on
    every platform and so cannot flip). This is the standalone analogue of the
    boundary-margin half of :func:`masks_equal_with_margin`: it guards a brand
    new candidate (with no incumbent to diff against) against a platform-
    dependent ``> 0.0`` sign flip.

    Returns ``(stable, min_nonzero_abs)`` where ``min_nonzero_abs`` is the
    smallest strictly-positive ``|raw|`` observed across all examples (the
    observed margin minimum), or ``None`` when there were no strictly-positive
    cells / no comparable examples. ``stable`` is ``False`` if the session
    cannot be built, if any inference errors, or if any cell is in
    ``(0, margin)``.
    """
    session = _make_raw_session(model)
    if session is None:
        return False, None

    examples = load_examples(task_num)
    all_examples = (
        list(examples.get("train", []))
        + list(examples.get("test", []))
        + list(examples.get("arc-gen", []))
    )

    compared_any = False
    min_abs: float | None = None
    for example in all_examples:
        benchmark = convert_to_numpy(example)
        if not benchmark:
            continue
        try:
            out = _raw_output(session, benchmark["input"])
        except _ORT_ERRORS:
            return False, None
        abs_out = np.abs(out)
        nonzero = abs_out[abs_out > 0.0]
        if nonzero.size:
            ex_min = float(nonzero.min())
            min_abs = ex_min if min_abs is None else min(min_abs, ex_min)
        # Near-boundary cells: strictly between 0 and the margin. Exact zeros
        # (background) are excluded — they cannot flip sign across platforms.
        if nonzero.size and bool(np.any(nonzero < margin)):
            return False, min_abs
        compared_any = True

    if not compared_any:
        return False, min_abs
    return True, min_abs


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass
