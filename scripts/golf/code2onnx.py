#!/usr/bin/env python3
"""code2onnx — compile a hand-written rule (PyTorch module) into a NeuroGolf ONNX.

Workflow this enables (the user's "write code -> convert to ONNX" idea):
  1. A human/agent writes the task's transformation RULE as a tiny torch.nn.Module
     operating on the one-hot grid tensor.
  2. This script exports it to ONNX with the exact NeuroGolf I/O contract
     (input "input" [1,10,30,30] f32 one-hot, output "output"; correctness is
     (output > 0.0) compared by array_equal to the one-hot expected grid).
  3. It scores the result with the real scorer and (if available) compares cost to
     the live incumbent, so you immediately see whether the compiled net is
     correct AND whether it actually beats what we already ship.

Solver file contract: define `build() -> torch.nn.Module`. Its forward(x) takes
x: FloatTensor[1,10,30,30] and returns FloatTensor[1,10,30,30] where >0 marks the
active channel per cell (out-of-grid cells must be <= 0 in every channel).

Usage:
    uv run python scripts/golf/code2onnx.py <solver.py> --task NNN \
        [--opset 18] [--out path.onnx] [--incumbent /tmp/incumbent_live/taskNNN.onnx]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import onnx
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _load_builder(solver_path: Path):
    spec = importlib.util.spec_from_file_location("solver_mod", str(solver_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {solver_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{solver_path} must define build() -> torch.nn.Module")
    return mod.build


def compile_to_onnx(solver_path: Path, out_path: Path, opset: int = 18) -> onnx.ModelProto:
    build = _load_builder(solver_path)
    module = build().eval()
    dummy = torch.zeros((1, 10, 30, 30), dtype=torch.float32)
    torch.onnx.export(
        module,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes=None,  # fully static shapes (NeuroGolf requirement)
    )
    return onnx.load(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("solver", type=Path, help="python file defining build() -> nn.Module")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--incumbent", type=Path, default=None,
                    help="incumbent onnx to compare cost against "
                         "(default /tmp/incumbent_live/taskNNN.onnx)")
    args = ap.parse_args()

    from lib import scoring  # noqa: E402

    out_path = args.out or Path(tempfile.mkdtemp()) / f"task{args.task:03d}_compiled.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = compile_to_onnx(args.solver, out_path, args.opset)
    n_nodes = len(model.graph.node)

    res = scoring.score_and_verify(model, args.task, tempfile.mkdtemp(),
                                   label="c2o", require_correct=False)
    print(f"=== compiled: {out_path}")
    print(f"    nodes={n_nodes}  result={res}")
    if res is None:
        print("    REJECTED (size/load/sanitize/unscorable). See above.")
        return 1
    if not res.get("correct"):
        print("    NOT CORRECT on public pairs — rule is wrong or contract mismatch.")

    inc_path = args.incumbent or Path(f"/tmp/incumbent_live/task{args.task:03d}.onnx")
    if inc_path.exists():
        inc = onnx.load(str(inc_path))
        incres = scoring.score_and_verify(inc, args.task, tempfile.mkdtemp(),
                                          label="inc", require_correct=False)
        if incres:
            dc = res["cost"] - incres["cost"]
            verdict = "WIN" if (res.get("correct") and dc < 0) else "no-gain"
            print(f"    incumbent cost={incres['cost']}  compiled cost={res['cost']}  "
                  f"Δ={dc:+d}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
