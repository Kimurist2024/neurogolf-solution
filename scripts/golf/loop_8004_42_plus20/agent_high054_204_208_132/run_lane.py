#!/usr/bin/env python3
"""Lane 132: theorem-driven exact/SOUND shaves for tasks 054, 204 and 208."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (54, 204, 208)
HASHES = {54: "264363fd", 204: "868de0fa", 208: "890034e9"}
MEMBER_SHA256 = {
    54: "783e18d6e3ec8abef5c7aae6111b80f47583bc2d0b02954984414efc1a3b86b8",
    204: "312fa4435c543c24301b15718602d148faa5c6510348a71f6482528b3092547b",
    208: "6c9bad970152f9380f07954878876c474dda51752586a200e4a911105fa4d016",
}
BASE_COSTS = {54: 2258, 204: 2222, 208: 1422}
GENERIC_KINDS = (
    "cleanup", "dedupe", "noops", "cse", "optional", "fold", "absorb",
    "combined", "normalize", "normalized_combined",
)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    "lane132_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
RANK = load_module("lane132_rank", ROOT / "scripts/golf/rank_dir.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def remove_nodes(model: onnx.ModelProto, outputs: set[str]) -> None:
    keep = [node for node in model.graph.node if not any(name in outputs for name in node.output)]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    EXACT.remove_dead_nodes(model)
    EXACT.remove_unused_initializers(model)


def castlike_to_cast(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(candidate), strict_mode=True, data_prop=True
    )
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    actions = []
    for node in candidate.graph.node:
        if node.op_type != "CastLike" or len(node.input) != 2 or not node.output:
            continue
        output_type = types.get(node.output[0])
        if not output_type:
            continue
        old_inputs = list(node.input)
        node.op_type = "Cast"
        del node.input[1:]
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("to", int(output_type))])
        actions.append({
            "output": node.output[0],
            "from": old_inputs,
            "to": TensorProto.DataType.Name(output_type),
            "proof": "CastLike_second_input_supplies_only_statically_inferred_dtype",
        })
    EXACT.remove_unused_initializers(candidate)
    return candidate, {"attributeized_castlikes": actions, "theorem": "all_tensor_values"}


def task204_direct_ccp(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Deliberately tested counterfactual: compose incremental CCP stages."""
    candidate = copy.deepcopy(model)
    node = next(node for node in candidate.graph.node if node.output == ["input_h20"])
    node.input[0] = "input_full_i32"
    remove_nodes(candidate, {f"input_h{size}" for size in range(21, 30)})
    node = next(node for node in candidate.graph.node if node.output == ["masks5_rows30"])
    node.input[0] = "masks5_20"
    remove_nodes(candidate, {f"masks5_rows{size}" for size in range(21, 30)})
    return candidate, {
        "ccp_compositions": ["30_to_20", "20_to_30"],
        "theorem": "counterfactual_requires_runtime_bitwise_check",
    }


def task204_slice_pad(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Replace one-sided incremental CCPs with their explicit Slice/Pad meaning."""
    candidate = copy.deepcopy(model)
    # The ten 1-cell crops remove the final row each time: [:20] on axis 2.
    crop = next(node for node in candidate.graph.node if node.output == ["input_h20"])
    crop.op_type = "Slice"
    del crop.attribute[:]
    del crop.input[:]
    crop.input.extend(["input_full_i32", "slice0_i64", "__ng_row_e20_dyn", "axis2_i64"])
    remove_nodes(candidate, {f"input_h{size}" for size in range(21, 30)})
    # The ten 1-cell pads append rows at the end: pad axis 1 by [0, 10].
    pad = next(node for node in candidate.graph.node if node.output == ["masks5_rows30"])
    pad.op_type = "Pad"
    del pad.attribute[:]
    del pad.input[:]
    pad.input.extend(["masks5_20", "pad0_10_i64", "", "axis1_i64"])
    remove_nodes(candidate, {f"masks5_rows{size}" for size in range(21, 30)})
    candidate.graph.initializer.extend([
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), "slice0_i64"),
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), "axis2_i64"),
        numpy_helper.from_array(np.asarray([0, 10], dtype=np.int64), "pad0_10_i64"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis1_i64"),
    ])
    EXACT.remove_unused_initializers(candidate)
    return candidate, {
        "explicit_one_sided_ops": [
            {"from": "ten incremental CenterCropPad 30->20", "to": "Slice axis2 [0:20]"},
            {"from": "ten incremental CenterCropPad 20->30", "to": "Pad axis1 [0,10]"},
        ],
        "theorem": "CenterCropPad odd_extra_at_end_composition",
    }


def task054_bool_fusions(
    model: onnx.ModelProto, *, flip: bool, box_select: bool
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Apply boolean identities whose integer ranges are proved from casts."""
    candidate = copy.deepcopy(model)
    actions = []
    removed: set[str] = set()
    if flip:
        # highR XOR (xpose AND (highR XOR highC)) == Where(xpose, highC, highR).
        node = next(node for node in candidate.graph.node if node.output == ["flip"])
        node.op_type = "Where"
        del node.input[:]
        node.input.extend(["xpose", "highC", "highR"])
        removed.update({"highDiff", "xdiff"})
        actions.append({
            "output": "flip",
            "from": "highR XOR (xpose AND (highR XOR highC))",
            "to": "Where(xpose,highC,highR)",
            "proof": "exhaustive_boolean_truth_table_8_rows",
        })
    if box_select:
        # D8 is Cast(Db) and therefore exactly 0 or 1; 2*D8-1 is {-1,+1}.
        node = next(node for node in candidate.graph.node if node.output == ["boxSel"])
        node.op_type = "Where"
        del node.input[:]
        node.input.extend(["Db", "__sel_one_u_i8", "neg_one_i8"])
        removed.add("Dsel2")
        candidate.graph.initializer.append(
            numpy_helper.from_array(np.asarray(-1, dtype=np.int8), "neg_one_i8")
        )
        actions.append({
            "output": "boxSel",
            "from": "(D8+D8)-1",
            "to": "Where(Db,1,-1)",
            "proof": "Db_boolean_Cast_to_int8_range_is_exactly_zero_or_one_no_overflow",
        })
    remove_nodes(candidate, removed)
    return candidate, {"boolean_fusions": actions, "theorem": "all_tensor_values"}


def bypass_ccp_outputs(
    model: onnx.ModelProto, outputs: tuple[str, ...]
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    actions = []
    remove = set()
    for output in outputs:
        node = next(node for node in candidate.graph.node if node.output == [output])
        if node.op_type != "CenterCropPad":
            raise RuntimeError(f"{output} is not CenterCropPad")
        replace_uses(candidate, output, node.input[0])
        remove.add(output)
        actions.append({
            "output": output,
            "source": node.input[0],
            "proof": "target_axis_size_equals_static_source_axis_size_30",
        })
    remove_nodes(candidate, remove)
    return candidate, {"bypassed_center_crop_pad": actions, "theorem": "all_tensor_values"}


def fold_task204_shape(
    model: onnx.ModelProto, outputs: tuple[str, ...]
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    values = {"__ng_shape_black_orig_30": 1, "__ng_row30_dim": 30}
    actions = []
    remove = set()
    for output in outputs:
        node = next(node for node in candidate.graph.node if node.output == [output])
        if node.op_type != "Shape":
            raise RuntimeError(f"{output} is not Shape")
        remove.add(output)
        candidate.graph.initializer.append(
            numpy_helper.from_array(np.asarray([values[output]], dtype=np.int64), output)
        )
        actions.append({
            "output": output,
            "value": [values[output]],
            "proof": "graph_input_contract_is_static_1x10x30x30",
        })
    keep = [node for node in candidate.graph.node if not any(name in remove for name in node.output)]
    del candidate.graph.node[:]
    candidate.graph.node.extend(keep)
    return candidate, {"fixed_shape_folds": actions, "theorem": "all_tensor_values"}


def set_value_info_dtype(model: onnx.ModelProto, names: set[str], dtype: int) -> None:
    for value in list(model.graph.value_info) + list(model.graph.output):
        if value.name in names:
            value.type.tensor_type.elem_type = dtype


def task204_shape_math_i32(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Keep input-dependent Shape nodes, but narrow their bounded math to int32."""
    candidate = copy.deepcopy(model)
    shape_batch = next(
        node for node in candidate.graph.node if node.output == ["__ng_shape_black_orig_30"]
    )
    shape_row = next(node for node in candidate.graph.node if node.output == ["__ng_row30_dim"])
    shape_batch.output[0] = "__ng_shape_black_orig_30_i64"
    shape_row.output[0] = "__ng_row30_dim_i64"
    cast_batch = helper.make_node(
        "Cast", ["__ng_shape_black_orig_30_i64"], ["__ng_shape_black_orig_30"],
        to=TensorProto.INT32,
    )
    cast_row = helper.make_node(
        "Cast", ["__ng_row30_dim_i64"], ["__ng_row30_dim"], to=TensorProto.INT32,
    )
    nodes = list(candidate.graph.node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes[:2] + [cast_batch, cast_row] + nodes[2:])
    bounded_names = {
        "__ng_shape_black_orig_30", "__ng_row30_dim", "row29_dyn", "row28_dyn",
        "row27_dyn", "row26_dyn", "row25_dyn", "row24_dyn", "row23_dyn",
        "row22_dyn", "row21_dyn", "__ng_row_e20_dyn", "__ng_row_e_dyn", "row18_dyn",
    }
    set_value_info_dtype(candidate, bounded_names, TensorProto.INT32)
    return candidate, {
        "dtype_narrowing": [{
            "tensors": sorted(bounded_names),
            "from": "int64",
            "to": "int32",
            "proof": "input contract dimensions are 1 and 30; twelve decrements yield 18..29, all within int32",
        }],
        "theorem": "all_tensor_values",
    }


def task208_crop_sizes_i32(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Narrow CenterCropPad target sizes 22..30 from int64 to int32."""
    candidate = copy.deepcopy(model)
    twentyone = next(item for item in candidate.graph.initializer if item.name == "twentyone_i64")
    twentyone.CopyFrom(numpy_helper.from_array(np.asarray([21], dtype=np.int32), "twentyone_i64"))
    candidate.graph.initializer.append(
        numpy_helper.from_array(np.asarray([1], dtype=np.int32), "row_one_i32")
    )
    targets = {f"target{value}" for value in range(22, 31)}
    for node in candidate.graph.node:
        if node.output and node.output[0] in targets:
            if node.input[1] != "row_axis_i64":
                raise RuntimeError(f"target chain contract drift at {node.output[0]}")
            node.input[1] = "row_one_i32"
    set_value_info_dtype(candidate, targets, TensorProto.INT32)
    return candidate, {
        "dtype_narrowing": [{
            "tensors": sorted(targets),
            "from": "int64",
            "to": "int32",
            "proof": "induction from int32 21 plus nine int32 ones gives exact values 22..30 without overflow",
        }],
        "theorem": "all_tensor_values",
    }


def task208_pad_controls_i32(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Pad natively accepts int32 controls; all reachable pad widths are -3..12."""
    candidate = copy.deepcopy(model)
    candidate.graph.initializer.append(
        numpy_helper.from_array(np.asarray([1], dtype=np.int32), "row_one_i32")
    )
    for output in ("row_pads_i64", "col_pads_i64"):
        node = next(node for node in candidate.graph.node if node.output == [output])
        if node.op_type != "Cast":
            raise RuntimeError(f"{output} Cast contract drift")
        node.attribute[0].i = TensorProto.INT32
    set_value_info_dtype(
        candidate, {"row_pads_i64", "col_pads_i64"}, TensorProto.INT32
    )
    row_pad = next(node for node in candidate.graph.node if node.output == ["row_pad19"])
    col_pad = next(node for node in candidate.graph.node if node.output == ["col_pad19"])
    row_pad.input[3] = "axis0_i32"
    col_pad.input[3] = "row_one_i32"
    return candidate, {
        "dtype_narrowing": [{
            "tensors": ["row_pads_i64", "col_pads_i64"],
            "from": "int64",
            "to": "int32",
            "proof": "r1/cidx are uint8 legal detector coordinates; 12-coordinate is within -3..12, exact int32",
        }],
        "theorem": "all_reachable_values_and_no_overflow",
    }


def task208_poly_einsum(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Evaluate the uint8 fallback polynomial with three exact monomial nodes."""
    candidate = copy.deepcopy(model)
    first_index = next(
        index for index, node in enumerate(candidate.graph.node) if node.output == ["poly_r2_u8"]
    )
    last_index = next(
        index for index, node in enumerate(candidate.graph.node) if node.output == ["poly_row_u8"]
    )
    replacement = [
        helper.make_node(
            "Einsum",
            ["first_r_u8", "first_r_u8", "first_shift_u8", "coef9_u8"],
            ["poly_9r2s_u8"],
            equation="i,i,i,->i",
        ),
        helper.make_node(
            "Einsum",
            ["h_sum012", "first_shift_u8", "coef12_u8"],
            ["poly_12hs_u8"],
            equation="i,i,->i",
        ),
        helper.make_node(
            "Mul", ["first_shift_u8", "coef13_u8"], ["poly_13s_u8"]
        ),
        helper.make_node(
            "Add", ["poly_9r2s_u8", "poly_12hs_u8"], ["poly_sum_a_u8"]
        ),
        helper.make_node(
            "Add", ["poly_sum_a_u8", "poly_13s_u8"], ["poly_sum_b_u8"]
        ),
        helper.make_node(
            "Add", ["poly_sum_b_u8", "coef4_u8"], ["poly_row_raw_u8"]
        ),
        helper.make_node(
            "BitwiseAnd", ["poly_row_raw_u8", "fifteen_u8"], ["poly_row_u8"]
        ),
    ]
    nodes = list(candidate.graph.node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes[:first_index] + replacement + nodes[last_index + 1 :])
    old_names = {
        "poly_r2_u8", "poly_9r2_u8", "poly_12h_u8", "poly_inner_a_u8",
        "poly_inner_u8", "poly_cprod_u8",
    }
    keep_vi = [value for value in candidate.graph.value_info if value.name not in old_names]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(keep_vi)
    return candidate, {
        "uint8_polynomial": {
            "from": "((((9*r*r)+(12*h)+13)*s)+4)&15",
            "to": "((9*r*r*s)+(12*h*s)+(13*s)+4)&15",
            "proof": (
                "distributivity in Z/256Z; every uint8 overflow is modulo 256 and the final "
                "mask projects to Z/16Z, so reassociation is exact for all 256^3 r,h,s values"
            ),
        },
        "theorem": "all_uint8_values_no_signed_overflow",
    }


def task208_argmax_u16(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    cast = next(node for node in candidate.graph.node if node.output == ["starts_clean_h"])
    argmax = next(node for node in candidate.graph.node if node.output == ["r1_i64k"])
    if cast.op_type != "Cast" or list(cast.input) != ["starts_clean"]:
        raise RuntimeError("task208 starts_clean Cast contract drift")
    argmax.input[0] = "starts_clean"
    remove_nodes(candidate, {"starts_clean_h"})
    return candidate, {
        "removed_cast": "Cast(starts_clean:uint16 -> float16)",
        "theorem": (
            "legal task inputs contain exactly two intended holes and no unintended hole; "
            "after first-hole removal every nonzero starts_clean entry is one uint16 power "
            "of two, exactly representable in float16, so ArgMax order and ties are identical"
        ),
    }


def task208_firstbit_bitshift(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    cast_h = next(node for node in candidate.graph.node if node.output == ["first_shift_h"])
    pow_node = next(node for node in candidate.graph.node if node.output == ["first_bit_h"])
    cast_u16 = next(node for node in candidate.graph.node if node.output == ["first_bit"])
    cast_h.output[0] = "first_shift_u16"
    cast_h.attribute[0].i = TensorProto.UINT16
    pow_node.op_type = "BitShift"
    del pow_node.attribute[:]
    pow_node.attribute.extend([helper.make_attribute("direction", "LEFT")])
    del pow_node.input[:]
    pow_node.input.extend(["one_u16", "first_shift_u16"])
    pow_node.output[0] = "first_bit"
    keep = [node for node in candidate.graph.node if node is not cast_u16]
    del candidate.graph.node[:]
    candidate.graph.node.extend(keep)
    candidate.graph.initializer.append(
        numpy_helper.from_array(np.asarray(1, dtype=np.uint16), "one_u16")
    )
    EXACT.remove_unused_initializers(candidate)
    return candidate, {
        "first_bit": "Cast(Pow(float16(2), shift),uint16) -> BitShift(uint16(1),shift)",
        "theorem": "shift is uint8 in [0,15], hence uint16 left shift equals exact power-of-two path",
    }


def variants(task: int, model: onnx.ModelProto):
    for kind in GENERIC_KINDS:
        candidate, actions = EXACT.transform(model, kind)
        yield f"generic_{kind}", candidate, actions
    yield "castlike_to_cast", *castlike_to_cast(model)
    if task == 54:
        yield "flip_where", *task054_bool_fusions(model, flip=True, box_select=False)
        yield "boxsel_where", *task054_bool_fusions(model, flip=False, box_select=True)
        yield "flip_boxsel_where", *task054_bool_fusions(model, flip=True, box_select=True)
        for output in ("__fp16_input_hid", "rowProfTrue", "oneRow16_hid", "rowNew_hid", "colNew_hid"):
            yield f"bypass_ccp_{output}", *bypass_ccp_outputs(model, (output,))
        yield "bypass_ccp_all_noops", *bypass_ccp_outputs(
            model,
            ("__fp16_input_hid", "rowProfTrue", "oneRow16_hid", "rowNew_hid", "colNew_hid"),
        )
    if task == 204:
        yield "shape_math_i32", *task204_shape_math_i32(model)
        yield "fold_shape_batch1", *fold_task204_shape(model, ("__ng_shape_black_orig_30",))
        yield "fold_shape_row30", *fold_task204_shape(model, ("__ng_row30_dim",))
        yield "fold_shape_both", *fold_task204_shape(
            model, ("__ng_shape_black_orig_30", "__ng_row30_dim")
        )
        yield "bypass_ccp_input_full", *bypass_ccp_outputs(model, ("input_full_hide",))
        yield "direct_ccp_counterfactual", *task204_direct_ccp(model)
        yield "slice_pad", *task204_slice_pad(model)
    if task == 208:
        yield "crop_sizes_i32", *task208_crop_sizes_i32(model)
        yield "pad_controls_i32", *task208_pad_controls_i32(model)
        yield "poly_einsum", *task208_poly_einsum(model)
        yield "argmax_u16", *task208_argmax_u16(model)
        yield "firstbit_bitshift", *task208_firstbit_bitshift(model)


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    result: dict[str, Any] = {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": scoring.calculate_params(model),
        "value_info": len(model.graph.value_info),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        result["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        result["checker_full"] = False
        result["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        result["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result["strict_shape_data_prop"] = False
        result["strict_shape_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    for name in ("baseline", "candidates", "audit"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    authority_before = sha256(AUTHORITY.read_bytes())
    if authority_before != ARCHIVE_SHA256:
        raise RuntimeError(f"authority drift: {authority_before}")
    with zipfile.ZipFile(AUTHORITY) as archive:
        payloads = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    rows = []
    for task, data in payloads.items():
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} member drift")
        base_path = HERE / "baseline" / f"task{task:03d}.onnx"
        base_path.write_bytes(data)
        base_cost = RANK.cost_of(str(base_path))
        if base_cost[2] != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} cost drift: {base_cost}")
        base = onnx.load_from_string(data)
        seen = set()
        for kind, candidate, actions in variants(task, base):
            candidate_data = candidate.SerializeToString()
            digest = sha256(candidate_data)
            if digest == MEMBER_SHA256[task] or digest in seen:
                continue
            seen.add(digest)
            path = HERE / "candidates" / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
            path.write_bytes(candidate_data)
            row = {
                "task": task,
                "kind": kind,
                "path": rel(path),
                "sha256": digest,
                "authority_sha256": MEMBER_SHA256[task],
                "authority_cost": BASE_COSTS[task],
                "actions": actions,
                "structure": structure(candidate),
            }
            try:
                memory, params, cost = RANK.cost_of(str(path))
                row["cost"] = {"memory": memory, "params": params, "cost": cost}
            except Exception as exc:  # noqa: BLE001
                row["cost_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(task, kind, row.get("cost"), row["structure"], flush=True)
    result = {
        "authority_zip_sha256_before": authority_before,
        "authority_zip_sha256_after": sha256(AUTHORITY.read_bytes()),
        "base_costs": BASE_COSTS,
        "rows": rows,
    }
    (HERE / "audit/probe_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
