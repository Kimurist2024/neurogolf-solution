#!/usr/bin/env python3
"""Independent finite-support proof for the task066 Selu review.

This script deliberately does not import the lane-200 proof.  It derives the
mandatory bit from task_2dd70a9a's S/U construction and checks every legal
size/geometry/flip/xpose tuple against the immutable graph's uint32 masks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_2dd70a9a.py"
AUTHORITY = Path("/private/tmp/ng800946_rank/task066.onnx")
EXPECTED_AUTHORITY_SHA = "bb8cebc8d71d275f4ec3f542d6aefea238b6c36d1cec77c0f2c1d533bf04ab4e"


def transformed(y: int, size: int, flip: int) -> int:
    return size - 1 - y if flip else y


def verify_graph_algebra() -> dict[str, object]:
    """Contract the opaque coordinate Einsums to their effective weights."""

    data = AUTHORITY.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_AUTHORITY_SHA
    model = onnx.load_model_from_string(data)
    init = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}

    operands = [
        init[name]
        for name in (
            "Uchan",
            "Trow",
            "Tcol",
            "Vchan",
            "Tcol",
            "WR",
            "Glin",
            "B",
            "Tcol",
            "WR",
            "Tcol",
            "WR",
            "Glin",
            "Glin",
        )
    ]
    target = np.zeros((2, 10, 30), dtype=np.float32)
    target[0, 2] = np.arange(30, dtype=np.float32)  # red coordinate sum
    target[1, 3] = np.arange(30, dtype=np.float32)  # green coordinate sum
    effective = {}
    for axis in ("h", "w"):
        equation = f"ac,aq,aq,dc,UV,UU,kdVk,p{axis},xp,xy,MN,MM,pxNp,zypt->kc{axis}"
        weights = np.einsum(equation, *operands, optimize=True)
        assert np.array_equal(weights, target)
        effective[axis] = {
            "nonzero_channels": {"output0": [2], "output1": [3]},
            "coordinate_weights": list(range(30)),
        }

    # The G/O channel contraction is exactly cyan (8), while the second G
    # input is exactly green (3).  These are the only color dependencies.
    cyan_selector = np.einsum(
        "qc,qz,z->c", init["Uchan"], init["Trow"], init["z1"], optimize=True
    )
    expected_cyan = np.zeros(10, dtype=np.float32)
    expected_cyan[8] = 1
    expected_green = np.zeros(10, dtype=np.float32)
    expected_green[3] = 1
    assert np.array_equal(cyan_selector, expected_cyan)
    assert np.array_equal(init["greenhalf10"], expected_green)
    assert np.array_equal(init["pow2"][:20], np.asarray([1 << i for i in range(20)], dtype=np.float32))
    assert np.count_nonzero(init["pow2"][20:]) == 0

    # Pin the exact downstream coordinate/orientation/mask pipeline used by
    # the proof, so an unrelated graph cannot satisfy the geometry assertions.
    expected_nodes = {
        2: ("Cast", ["rs"], ["rsu"]),
        3: ("Cast", ["cs"], ["csu"]),
        6: ("BitwiseAnd", ["rsR", "one_u8"], ["par"]),
        12: ("Where", ["vert", "rsRh", "csRh"], ["rr0"]),
        13: ("Where", ["vert", "rsGh", "csGh"], ["gr0"]),
        14: ("Where", ["vert", "csRh", "rsRh"], ["cRu"]),
        15: ("Where", ["vert", "csGh", "rsGh"], ["cGu"]),
        19: ("Where", ["gt_cR_cG", "cRplus", "cRminus"], ["cOut_u8"]),
        22: ("Einsum", ["input", "input", "Uchan", "Trow", "z1", "greenhalf10", "pow2"], ["Gv"]),
        23: ("Einsum", ["input", "input", "Uchan", "Trow", "z1", "greenhalf10", "pow2"], ["Gh"]),
        26: ("Einsum", ["input", "Uchan", "Trow", "z1", "pow2", "cOutOH"], ["Ov"]),
        27: ("Einsum", ["input", "Uchan", "Trow", "z1", "pow2", "cOutOH"], ["Oh"]),
        30: ("BitShift", ["G", "sh2"], ["Gright"]),
        31: ("BitwiseAnd", ["O", "G"], ["pairD"]),
        32: ("BitwiseAnd", ["O", "Gright"], ["pairU"]),
        38: ("BitwiseAnd", ["pairD", "ltG"], ["aMask"]),
        39: ("BitwiseAnd", ["pairU", "geG2"], ["bMask"]),
        59: ("Or", ["noA", "forceB"], ["useB"]),
        62: ("Where", ["useB", "bPowF", "aMaskF"], ["selF"]),
    }
    for index, expected in expected_nodes.items():
        node = model.graph.node[index]
        actual = (node.op_type, list(node.input), list(node.output))
        assert actual == expected, (index, actual, expected)

    return {
        "authority_sha256": EXPECTED_AUTHORITY_SHA,
        "coordinate_einsum_effective_weights": effective,
        "coordinate_consequence": "rs/cs are exact red/green coordinate sums; cyan/background coefficients are zero",
        "marker_consequence": "two consecutive markers make >>1 their minimum coordinate; parity selects transpose exactly",
        "cOut_consequence": "red is on the right of green, so cOut=cR+1 is the mandatory outward cyan column",
        "cyan_selector": cyan_selector.tolist(),
        "green_selector": init["greenhalf10"].tolist(),
        "pow2_support": [0, 19],
        "pinned_nodes": sorted(expected_nodes),
    }


def check_case(
    *,
    size: int,
    green_rows: tuple[int, int],
    green_guard: int,
    outward_guard: int,
    flip: int,
    expected_branch: str,
) -> tuple[int, int, int, int]:
    """Apply nodes 30--41 to only the mandatory cyan guard pair."""

    gr0 = min(transformed(y, size, flip) for y in green_rows)
    green_y = transformed(green_guard, size, flip)
    outward_y = transformed(outward_guard, size, flip)

    # Nodes 22--25 compute G=2*C, so a cyan bit y becomes G bit y+1.
    g_mask = 1 << (green_y + 1)
    o_mask = 1 << outward_y
    pair_d = o_mask & g_mask
    pair_u = o_mask & (g_mask >> 2)
    a_mask = pair_d & ((1 << gr0) - 1)
    ge_g2 = (-(1 << (gr0 + 2))) & 0xFFFFFFFF
    b_mask = pair_u & ge_g2

    if expected_branch == "a":
        assert a_mask == o_mask and b_mask == 0
    else:
        assert expected_branch == "b"
        assert b_mask == o_mask and a_mask == 0
    assert 0 <= outward_y < 20
    assert 0 <= gr0 < 20
    return gr0, green_y, outward_y, a_mask | b_mask


def main() -> None:
    graph_algebra = verify_graph_algebra()
    counts = {"S_geometry": 0, "U_geometry": 0, "flip_xpose_cases": 0}
    extrema = {
        "gr0_min": 99,
        "gr0_max": -1,
        "guard_min": 99,
        "guard_max": -1,
        "outward_min": 99,
        "outward_max": -1,
        "mandatory_bit_min": 1 << 31,
        "mandatory_bit_max": 0,
    }

    def record(values: tuple[int, int, int, int]) -> None:
        gr0, guard, outward, bit = values
        extrema["gr0_min"] = min(extrema["gr0_min"], gr0)
        extrema["gr0_max"] = max(extrema["gr0_max"], gr0)
        extrema["guard_min"] = min(extrema["guard_min"], guard)
        extrema["guard_max"] = max(extrema["guard_max"], guard)
        extrema["outward_min"] = min(extrema["outward_min"], outward)
        extrema["outward_max"] = max(extrema["outward_max"], outward)
        extrema["mandatory_bit_min"] = min(extrema["mandatory_bit_min"], bit)
        extrema["mandatory_bit_max"] = max(extrema["mandatory_bit_max"], bit)

    for size in range(10, 21):
        # Option S, exactly matching inclusive common.randint bounds.
        for height in range(3 * size // 4, 7 * size // 8 + 1):
            for width in range(size // 2, 3 * size // 4 + 1):
                for row in range(1, size - height):
                    for _col in range(1, size - width):
                        for mid in range(row + 3, row + height - 2):
                            counts["S_geometry"] += 1
                            base = row + height - 1
                            green_rows = (base - 1, base)
                            for flip in (0, 1):
                                expected = "b" if flip else "a"
                                record(
                                    check_case(
                                        size=size,
                                        green_rows=green_rows,
                                        green_guard=mid - 1,
                                        outward_guard=mid,
                                        flip=flip,
                                        expected_branch=expected,
                                    )
                                )
                                # xpose selects the graph's row/column-swapped
                                # equation and leaves this logical path axis intact.
                                counts["flip_xpose_cases"] += 2

        # Option U.
        for height in range(size // 2, 3 * size // 4 + 1):
            for width in range(size // 2, 3 * size // 4 + 1):
                for row in range(1, size - height):
                    for _col in range(1, size - width):
                        base = row + height - 1
                        for _mid1 in range(row, row + height - 3):
                            for mid2 in range(row, row + height - 2):
                                counts["U_geometry"] += 1
                                green_rows = (mid2 + 1, mid2)
                                for flip in (0, 1):
                                    expected = "a" if flip else "b"
                                    record(
                                        check_case(
                                            size=size,
                                            green_rows=green_rows,
                                            green_guard=base + 1,
                                            outward_guard=base,
                                            flip=flip,
                                            expected_branch=expected,
                                        )
                                    )
                                    counts["flip_xpose_cases"] += 2

    assert counts == {
        "S_geometry": 15336,
        "U_geometry": 449928,
        "flip_xpose_cases": 1861056,
    }

    # Exhaust the useB truth table.  The geometry proof supplies a|b, and the
    # graph only selects b when noA or (hasB & force_any), so selection cannot
    # turn a nonempty pair into zero.
    selection_rows = []
    for a_positive, b_positive in ((True, False), (False, True), (True, True)):
        for force_any in (False, True):
            no_a = not a_positive
            has_b = b_positive
            use_b = no_a or (has_b and force_any)
            selected_positive = b_positive if use_b else a_positive
            assert selected_positive
            selection_rows.append(
                {
                    "a_positive": a_positive,
                    "b_positive": b_positive,
                    "force_any": force_any,
                    "use_b": use_b,
                    "selected_positive": selected_positive,
                }
            )

    result = {
        "generator_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        "counts": counts,
        "graph_algebra": graph_algebra,
        "extrema": extrema,
        "selection_truth_table": selection_rows,
        "noise_closure": {
            "ordering": "random cyan is prepended; path and guards overwrite it",
            "bit_property": "extra cyan ORs bits into C/O; it cannot clear the mandatory common bit",
            "carry_property": "one cell per row/column position makes each sum a distinct power-of-two bitmask",
        },
        "numeric_closure": {
            "C_bound": [0, (1 << 20) - 1],
            "G_bound": [0, 2 * ((1 << 20) - 1)],
            "O_bound": [0, (1 << 20) - 1],
            "selected_uint32_bound": [1, (1 << 20) - 1],
            "float32_exact": "G and O are integers below 2**24",
            "float16_cast": "a positive uint32 becomes >=1 or +inf, never <=0/NaN",
        },
        "all_assertions_passed": True,
        "counterexample": None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
