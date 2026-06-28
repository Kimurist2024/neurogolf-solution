#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import onnx


def _worker(model_bytes: bytes, task: int, label: str, queue: mp.Queue) -> None:
    sys.path.insert(0, "scripts")
    from lib import scoring

    with tempfile.TemporaryDirectory() as workdir:
        result = scoring.score_and_verify(
            onnx.load_model_from_string(model_bytes),
            task,
            workdir,
            label=label,
            require_correct=True,
        )
    if result and result.get("score") is not None:
        queue.put(
            {
                "ok": True,
                "cost": int(result["cost"]),
                "memory": int(result["memory"]),
                "params": int(result["params"]),
                "correct": bool(result["correct"]),
                "score": float(result["score"]),
            }
        )
    else:
        queue.put({"ok": False, "reason": "score_and_verify_failed"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--label", default="i5")
    parser.add_argument("--ban-topk", action="store_true")
    args = parser.parse_args()

    model_bytes = args.onnx.read_bytes()
    model = onnx.load_model_from_string(model_bytes)
    if args.ban_topk and any(node.op_type == "TopK" for node in model.graph.node):
        print(json.dumps({"ok": False, "reason": "topk_banned"}, sort_keys=True))
        return 2

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(model_bytes, args.task, args.label, queue))
    proc.start()
    proc.join(args.timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        print(json.dumps({"ok": False, "reason": "timeout"}, sort_keys=True))
        return 124
    if proc.exitcode != 0:
        print(json.dumps({"ok": False, "reason": f"exit_{proc.exitcode}"}, sort_keys=True))
        return 1
    if queue.empty():
        print(json.dumps({"ok": False, "reason": "empty_result"}, sort_keys=True))
        return 1
    result = queue.get()
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
