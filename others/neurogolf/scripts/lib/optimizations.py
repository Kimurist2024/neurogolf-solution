"""Behaviour-preserving ONNX optimization passes (S1-S4, G3).

Each pass is a pure function: it deep-copies the input ``ModelProto``, mutates
only the copy, and returns ``(new_model, stats)``. The input is never modified.
These passes correspond to the strategies in
``proposals/001-zero-risk-onnx-cost-reduction.md`` and
``proposals/002-residual-dtype-and-noop.md``:

* S1 ``s1_prune_unused_initializers`` — drop initializers no node consumes.
* S2 ``s2_dedup_initializers``        — merge byte-identical initializers.
* S3 ``s3_compress_uniform_to_scalar``— uniform initializers -> scalar, only
  when every consumer op is broadcast-safe.
* S4 ``s4_clean_value_info``          — drop stale / duplicate value_info.
* G3 ``g3_remove_noops``              — remove Identity / Mul-Div-by-ones /
  Add-Sub-by-zeros / same-shape Reshape-Squeeze-Unsqueeze-Flatten /
  identity-permutation Transpose by rewiring consumers; drop orphans.

None of these change the computation the graph performs (G3 is verified
bit-identical against the pre-pass model by the pipeline before acceptance).
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

Stats = dict[str, Any]

# Ops that broadcast a scalar operand identically to a full tensor, so an
# all-equal initializer can be replaced by a scalar without changing results.
SAFE_OPS: frozenset[str] = frozenset(
    {
        "Greater",
        "Less",
        "Equal",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Where",
        "Max",
        "Min",
        "And",
        "Or",
        "Not",
        "Clip",
        "LessOrEqual",
        "GreaterOrEqual",
        "Sum",
    }
)


def _initializer_key(init: onnx.TensorProto) -> tuple[str, tuple[int, ...], bytes]:
    """Fingerprint: (dtype str, shape tuple, raw bytes) — exact content match."""
    arr = numpy_helper.to_array(init)
    return (arr.dtype.str, tuple(arr.shape), arr.tobytes())


def s1_prune_unused_initializers(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Stats]:
    """Remove initializers not referenced by any node input."""
    new_model = copy.deepcopy(model)
    graph = new_model.graph

    used: set[str] = set()
    for node in graph.node:
        used.update(inp for inp in node.input if inp)

    to_remove = [init for init in graph.initializer if init.name not in used]
    saved_params = 0
    for init in to_remove:
        graph.initializer.remove(init)
        saved_params += int(np.prod(init.dims)) if init.dims else 1

    stats: Stats = {
        "removed": len(to_remove),
        "saved_params": saved_params,
        "removed_names": [init.name for init in to_remove],
    }
    return new_model, stats


def s2_dedup_initializers(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Stats]:
    """Merge byte-identical initializers, rewire inputs, prune orphans."""
    new_model = copy.deepcopy(model)
    graph = new_model.graph

    groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for init in graph.initializer:
        groups[_initializer_key(init)].append(init.name)

    replace: dict[str, str] = {}
    for names in groups.values():
        if len(names) <= 1:
            continue
        canonical = sorted(names, key=lambda s: (len(s), s))[0]
        for name in names:
            if name != canonical:
                replace[name] = canonical

    if not replace:
        return new_model, {"merged": 0, "saved_params": 0}

    params_before = sum(
        int(np.prod(i.dims)) if i.dims else 1 for i in graph.initializer
    )

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in replace:
                node.input[i] = replace[name]

    used = {n for node in graph.node for n in node.input if n}
    kept = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)

    params_after = sum(
        int(np.prod(i.dims)) if i.dims else 1 for i in graph.initializer
    )

    stats: Stats = {
        "merged": len(replace),
        "saved_params": params_before - params_after,
    }
    return new_model, stats


def s3_compress_uniform_to_scalar(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Stats]:
    """Replace all-equal initializers with scalars when consumers are safe."""
    new_model = copy.deepcopy(model)
    graph = new_model.graph

    arrs: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        arrs[init.name] = numpy_helper.to_array(init)

    consumers: dict[str, list[str]] = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            if name in arrs:
                consumers[name].append(node.op_type)

    total_saved = 0
    compressed = 0
    for init in graph.initializer:
        arr = arrs[init.name]
        size = max(int(np.prod(arr.shape)), 1)
        if size <= 1:
            continue
        flat = arr.ravel()
        if not np.all(flat == flat[0]):
            continue
        consumer_ops = set(consumers.get(init.name, []))
        if not consumer_ops <= SAFE_OPS:
            continue

        scalar = np.array(flat[0], dtype=arr.dtype)
        init.CopyFrom(numpy_helper.from_array(scalar, init.name))
        total_saved += size - 1
        compressed += 1

    stats: Stats = {"compressed": compressed, "saved_params": total_saved}
    return new_model, stats


def s4_clean_value_info(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Stats]:
    """Drop stale value_info and deduplicate value_info sharing a name.

    Removes ``graph.value_info`` entries whose name is not produced by any node
    output and is neither a graph input/output name nor referenced as any node
    input. Also deduplicates entries that share the same name, keeping the
    first occurrence (duplicate names make the official scorer return ``None``).
    """
    new_model = copy.deepcopy(model)
    graph = new_model.graph

    produced: set[str] = set()
    for node in graph.node:
        produced.update(out for out in node.output if out)
    referenced: set[str] = set()
    for node in graph.node:
        referenced.update(inp for inp in node.input if inp)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    keep_names = produced | referenced | io_names

    kept_entries = []
    seen: set[str] = set()
    removed_stale = 0
    removed_dup = 0
    for vi in graph.value_info:
        name = vi.name
        if name in seen:
            removed_dup += 1
            continue
        if name not in keep_names:
            removed_stale += 1
            continue
        seen.add(name)
        kept_entries.append(vi)

    del graph.value_info[:]
    graph.value_info.extend(kept_entries)

    stats: Stats = {
        "removed_stale": removed_stale,
        "removed_duplicate": removed_dup,
    }
    return new_model, stats


# --- G3: no-op node removal (proposal 002) -----------------------------------

# Shape-only ops whose output equals the data input verbatim when shapes match.
_SHAPE_NOOP_OPS: frozenset[str] = frozenset(
    {"Reshape", "Squeeze", "Unsqueeze", "Flatten"}
)


def _build_const_arrays(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    """Map every initializer / ``Constant``-node tensor name to its array."""
    arrays: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        try:
            arrays[init.name] = numpy_helper.to_array(init)
        except Exception:  # noqa: BLE001 — skip unreadable tensors.
            continue
    for node in graph.node:
        if node.op_type != "Constant" or not node.output or not node.output[0]:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                try:
                    arrays[node.output[0]] = numpy_helper.to_array(attr.t)
                except Exception:  # noqa: BLE001
                    pass
    return arrays


def _shape_of(
    name: str,
    shape_map: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Resolve a fully-static shape for ``name`` or ``None`` if unknown."""
    return shape_map.get(name)


def _is_all_value(arr: np.ndarray | None, value: float) -> bool:
    """True iff ``arr`` exists, is numeric, and every element equals ``value``."""
    if arr is None or arr.size == 0:
        return False
    if not np.issubdtype(arr.dtype, np.number):
        return False
    return bool(np.all(arr == value))


def _data_input_for_noop(
    node: onnx.NodeProto,
    const_arrays: dict[str, np.ndarray],
    shape_map: dict[str, tuple[int, ...] | None],
) -> str | None:
    """Return the surviving data input if ``node`` is a removable no-op, else None.

    Detects exactly the categories from proposal 002 / the S7 census:
    ``Identity``; ``Mul``/``Div`` where the non-data operand is all-ones;
    ``Add``/``Sub`` where the non-data operand is all-zeros; same-shape
    ``Reshape``/``Squeeze``/``Unsqueeze``/``Flatten``; identity-permutation
    ``Transpose``. Broadcast safety is enforced: for arithmetic no-ops the
    surviving input's static shape must equal the output's static shape; for
    shape ops the data input's static shape must equal the output's shape.
    Returns ``None`` whenever any required shape is unknown (fail-safe).
    """
    op = node.op_type
    if not node.output or not node.output[0]:
        return None
    out_name = node.output[0]
    out_shape = _shape_of(out_name, shape_map)

    if op == "Identity":
        if len(node.input) == 1 and node.input[0]:
            return node.input[0]
        return None

    if op == "Transpose":
        if len(node.input) != 1 or not node.input[0]:
            return None
        in_shape = _shape_of(node.input[0], shape_map)
        if in_shape is None:
            return None
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        rank = len(in_shape)
        identity_perm = list(range(rank))
        if perm is None or perm == identity_perm:
            return node.input[0]
        return None

    if op in _SHAPE_NOOP_OPS:
        if not node.input or not node.input[0]:
            return None
        data = node.input[0]
        in_shape = _shape_of(data, shape_map)
        if in_shape is None or out_shape is None:
            return None
        if in_shape == out_shape:
            return data
        return None

    if op in {"Mul", "Div", "Add", "Sub"}:
        if len(node.input) != 2 or not node.input[0] or not node.input[1]:
            return None
        a, b = node.input[0], node.input[1]
        target = 1.0 if op in {"Mul", "Div"} else 0.0
        # For Div/Sub the identity operand MUST be the second input (x/1, x-0).
        if op in {"Div", "Sub"}:
            if _is_all_value(const_arrays.get(b), target):
                data = a
            else:
                return None
        else:  # Mul / Add are commutative: either operand may be the identity.
            if _is_all_value(const_arrays.get(b), target):
                data = a
            elif _is_all_value(const_arrays.get(a), target):
                data = b
            else:
                return None
        data_shape = _shape_of(data, shape_map)
        if data_shape is None or out_shape is None:
            return None
        # Broadcast safety: the surviving data tensor must already match the
        # output shape (otherwise the constant operand changed the shape).
        if data_shape != out_shape:
            return None
        return data

    return None


def _static_shape_map(
    model: onnx.ModelProto,
) -> dict[str, tuple[int, ...] | None]:
    """Build a name -> fully-static-shape map via strict shape inference.

    Only tensors with a complete static shape (every dim a positive int) get a
    tuple; anything dynamic / unknown maps to ``None``.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:  # noqa: BLE001 — fall back to the un-inferred graph.
        inferred = model
    graph = inferred.graph
    shape_map: dict[str, tuple[int, ...] | None] = {}
    sources = (
        list(graph.input) + list(graph.value_info) + list(graph.output)
    )
    for vi in sources:
        ttype = vi.type.tensor_type if vi.type.HasField("tensor_type") else None
        if ttype is None or not ttype.HasField("shape"):
            shape_map[vi.name] = None
            continue
        dims: list[int] = []
        ok = True
        for dim in ttype.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(dim.dim_value)
            else:
                ok = False
                break
        shape_map[vi.name] = tuple(dims) if ok else None
    return shape_map


def g3_remove_noops(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Stats]:
    """Remove no-op nodes by rewiring consumers; drop orphaned constants.

    Iterates to a fixed point so chains of no-ops collapse. For each removable
    no-op node with output ``Y`` and surviving data input ``X``:

    * if ``Y`` is the graph output ``'output'``, rewire the PRODUCER of ``X`` to
      write ``'output'`` directly (and rename downstream references), then drop
      the no-op node;
    * otherwise rewire every consumer (node inputs + graph outputs) of ``Y`` to
      use ``X`` and drop the no-op node.

    After convergence, initializers and ``Constant`` nodes whose outputs are no
    longer consumed by anything are dropped (orphan cleanup). The transform is
    numeric-preserving by construction but the pipeline still verifies it
    bit-identical against the pre-pass model before accepting it.
    """
    new_model = copy.deepcopy(model)
    graph = new_model.graph

    removed_nodes = 0
    output_rewires = 0
    changed = True
    while changed:
        changed = False
        shape_map = _static_shape_map(new_model)
        const_arrays = _build_const_arrays(graph)
        graph_output_names = {o.name for o in graph.output}

        for idx, node in enumerate(graph.node):
            data_in = _data_input_for_noop(node, const_arrays, shape_map)
            if data_in is None:
                continue
            out_name = node.output[0]

            if out_name == "output":
                # The no-op writes the graph output: rewire X's producer to
                # write 'output' instead, then drop the no-op. Only safe when X
                # is produced by exactly one node and consumed only by this
                # no-op (so renaming X to 'output' breaks nothing).
                producer = _single_producer(graph, data_in)
                if producer is None:
                    continue
                other_consumers = _consumer_count(graph, data_in)
                if other_consumers != 1:
                    # X feeds something else too; renaming would break it.
                    continue
                # Rename the producer's matching output to 'output'.
                for i, o in enumerate(producer.output):
                    if o == data_in:
                        producer.output[i] = "output"
                producer.name = producer.output[0]
                _drop_node(graph, idx)
                _drop_value_info(graph, data_in)
                removed_nodes += 1
                output_rewires += 1
                changed = True
                break

            # Standard case: rewire consumers of out_name to data_in.
            for other in graph.node:
                if other is node:
                    continue
                for i, name in enumerate(other.input):
                    if name == out_name:
                        other.input[i] = data_in
            if out_name in graph_output_names:
                # out_name is a (non-'output') declared graph output: rewire it.
                for gout in graph.output:
                    if gout.name == out_name:
                        gout.name = data_in
            _drop_node(graph, idx)
            _drop_value_info(graph, out_name)
            removed_nodes += 1
            changed = True
            break

    orphans = _drop_orphan_constants(graph)

    stats: Stats = {
        "removed_nodes": removed_nodes,
        "output_rewires": output_rewires,
        "orphaned_dropped": orphans,
    }
    return new_model, stats


def _single_producer(
    graph: onnx.GraphProto, tensor: str
) -> onnx.NodeProto | None:
    """Return the unique node producing ``tensor`` or ``None`` if 0 or >1."""
    producers = [n for n in graph.node if tensor in n.output]
    if len(producers) == 1:
        return producers[0]
    return None


def _consumer_count(graph: onnx.GraphProto, tensor: str) -> int:
    """Count node-input occurrences of ``tensor`` (graph outputs not counted)."""
    return sum(1 for n in graph.node for inp in n.input if inp == tensor)


def _drop_node(graph: onnx.GraphProto, index: int) -> None:
    del graph.node[index]


def _drop_value_info(graph: onnx.GraphProto, name: str) -> None:
    """Remove any ``value_info`` entry for ``name`` (stale after rewiring)."""
    kept = [vi for vi in graph.value_info if vi.name != name]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)


def _drop_orphan_constants(graph: onnx.GraphProto) -> int:
    """Drop initializers and ``Constant`` nodes no longer consumed by anything.

    Repeats until stable so a freshly-orphaned chain collapses. Graph outputs
    are treated as consumers so a constant feeding the output is never dropped.
    """
    dropped = 0
    changed = True
    while changed:
        changed = False
        used: set[str] = set()
        for node in graph.node:
            used.update(inp for inp in node.input if inp)
        used.update(o.name for o in graph.output)

        init_remove = [i for i in graph.initializer if i.name not in used]
        for init in init_remove:
            graph.initializer.remove(init)
            dropped += 1
            changed = True

        const_remove = [
            n
            for n in graph.node
            if n.op_type == "Constant"
            and n.output
            and n.output[0]
            and n.output[0] not in used
        ]
        for node in const_remove:
            graph.node.remove(node)
            dropped += 1
            changed = True
    return dropped
