#!/usr/bin/env python3
"""Mechanically prove task013 colour-class reduction for all 72 colour pairs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main() -> None:
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        model = onnx.load_from_string(archive.read("task013.onnx"))
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    kfeat = arrays["Kfeat"].astype(np.float32)
    qch = arrays["Qch"].astype(np.float32)
    wj = arrays["Wj"].astype(np.float32)
    gbg = arrays["Gbg"].astype(np.float32)
    tzero = arrays["T_zero"].astype(np.float32)

    colour_selector_checks = 0
    selector_margin = float("inf")
    for colour in range(1, 10):
        cfeat = np.asarray([1, colour], dtype=np.float32)
        feature = np.einsum("lk,m,alm->ak", kfeat, cfeat, qch)
        expected_feature = np.stack(
            [np.ones(10, dtype=np.float32), np.arange(10, dtype=np.float32) - colour]
        )
        assert np.array_equal(feature, expected_feature)
        selector = np.einsum("ak,ak,ra->rk", feature, feature, wj)
        expected_selector = np.stack(
            [0.25 - (np.arange(10, dtype=np.float32) - colour) ** 2, -np.ones(10, dtype=np.float32)]
        )
        assert np.array_equal(selector, expected_selector)
        assert np.array_equal(selector[0] > 0, np.arange(10) == colour)
        selector_margin = min(selector_margin, float(np.abs(selector[0]).min()))
        colour_selector_checks += 10

    background_feature = np.einsum("Tk,U,GTU->Gk", kfeat, tzero, qch)
    expected_background_feature = np.stack(
        [np.ones(10, dtype=np.float32), np.arange(10, dtype=np.float32)]
    )
    assert np.array_equal(background_feature, expected_background_feature)
    background = np.einsum("Gk,Gk,EFG->EFk", background_feature, background_feature, gbg)
    expected_background = gbg[:, :, 0, None] + gbg[:, :, 1, None] * np.arange(10, dtype=np.float32)[None, None, :] ** 2
    assert np.array_equal(background, expected_background)
    assert np.array_equal(background[0, 0] > 0, np.arange(10) == 0)

    # The selected marker axis always has positions p0=start and p1=start+sep+1.
    # In float16, all intermediate integers here are <=493 and exactly
    # representable.  Exhaustively verify the actual S/T/div sequence for all
    # reachable p0,d and all ordered distinct nonzero colour pairs.
    recovery_checks = 0
    pairs = [(c0, c1) for c0 in range(1, 10) for c1 in range(1, 10) if c0 != c1]
    for width in range(20, 31):
        for p0 in range(1, width // 2 + 1):
            for sep in range(1, 6):
                d = sep + 1
                for c0, c1 in pairs:
                    s = np.float16(c0 + c1)
                    t = np.float16(c0 * p0 + c1 * (p0 + d))
                    recovered1 = np.float16((t - np.float16(s * np.float16(p0))) / np.float16(d))
                    recovered0 = np.float16(s - recovered1)
                    assert recovered0 == np.float16(c0)
                    assert recovered1 == np.float16(c1)
                    recovery_checks += 1

    payload = {
        "task": 13,
        "proved": True,
        "ordered_distinct_nonzero_colour_pairs": len(pairs),
        "colour_selector_checks": colour_selector_checks,
        "colour_recovery_checks": recovery_checks,
        "minimum_colour_selector_absolute_margin": selector_margin,
        "identities": {
            "marker_colour_feature": "[1, k-c]",
            "marker_colour_selector": "[0.25-(k-c)^2, -1]",
            "marker_positive_iff": "k=c for integer channels k,c in 0..9",
            "background_feature": "[1,k]",
            "background_E0_F0_score": "0.0625-1000000*k^2",
            "background_positive_iff": "k=0",
            "recovery": "S=c0+c1; T=c0*p0+c1*(p0+d); (T-S*p0)/d=c1; S-c1=c0",
        },
        "colour_independent_geometry": "All geometry moments use nz_f=[0,1,...,1], so they depend on marker occupancy/positions but not nonzero colour IDs. Colours enter only through the proved recovery and selector identities.",
        "reduction": "Executing all 37,800 structural states with canonical colours [1,2] covers all 72 ordered distinct nonzero colour pairs because changing colours only relocates the exactly-positive output channel selectors.",
    }
    (HERE / "task013_colour_proof.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
