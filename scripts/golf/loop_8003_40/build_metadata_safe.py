#!/usr/bin/env python3
"""Build an audited NeuroGolf ZIP without changing archive layout metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import shape_inference


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.golf.check_conv_bias import check_model  # noqa: E402


TASK_RE = re.compile(r"task[_-]?(\d{1,3})\.onnx$", re.IGNORECASE)
MAX_MODEL_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {
    "Loop",
    "Scan",
    "NonZero",
    "Unique",
    "Script",
    "Function",
    "Compress",
    "SequenceAt",
    "SplitToSequence",
    "SequenceConstruct",
    "SequenceInsert",
    "ConcatFromSequence",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def task_from_name(name: str) -> int | None:
    match = TASK_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def tensor_shape_is_static(value_info: onnx.ValueInfoProto) -> bool:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    return all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for dimension in tensor_type.shape.dim
    )


def audit_replacement(task: int, path: Path) -> tuple[bytes, dict[str, object]]:
    data = path.read_bytes()
    errors: list[str] = []
    if len(data) > MAX_MODEL_BYTES:
        errors.append(f"serialized_size={len(data)} exceeds {MAX_MODEL_BYTES}")

    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        return data, {
            "task": task,
            "path": str(path),
            "sha256": sha256_bytes(data),
            "serialized_size": len(data),
            "errors": [f"checker: {exc!r}"],
            "valid": False,
        }

    ops = sorted({node.op_type for node in model.graph.node})
    banned = sorted(set(ops) & BANNED_OPS)
    if banned:
        errors.append(f"banned_ops={banned}")
    if model.functions:
        errors.append("model contains local functions")
    if model.graph.sparse_initializer:
        errors.append("model contains sparse initializers")
    if any(
        attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attribute in node.attribute
    ):
        errors.append("model contains nested graphs")

    declared = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    if any(not tensor_shape_is_static(value_info) for value_info in declared):
        errors.append("model has a missing, dynamic, or non-positive declared tensor shape")

    try:
        shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        strict_shape_inference = "PASS"
    except Exception as exc:  # noqa: BLE001
        strict_shape_inference = repr(exc)
        errors.append(f"strict_shape_inference: {exc!r}")

    bias_issues = check_model(model)
    if bias_issues:
        errors.append(f"conv_bias_issues={bias_issues}")

    return data, {
        "task": task,
        "path": str(path),
        "sha256": sha256_bytes(data),
        "serialized_size": len(data),
        "ops": ops,
        "strict_shape_inference": strict_shape_inference,
        "conv_bias_issues": bias_issues,
        "errors": errors,
        "valid": not errors,
    }


def comparable_info(info: zipfile.ZipInfo) -> dict[str, object]:
    return {
        "filename": info.filename,
        "date_time": info.date_time,
        "compress_type": info.compress_type,
        "comment": info.comment,
        "extra": info.extra,
        "create_system": info.create_system,
        "create_version": info.create_version,
        "extract_version": info.extract_version,
        "reserved": info.reserved,
        "flag_bits": info.flag_bits,
        "volume": info.volume,
        "internal_attr": info.internal_attr,
        "external_attr": info.external_attr,
    }


def build(
    baseline: Path,
    output: Path,
    replacements: dict[int, Path],
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    replacement_bytes: dict[int, bytes] = {}
    replacement_audits: list[dict[str, object]] = []
    for task, path in sorted(replacements.items()):
        data, audit = audit_replacement(task, path)
        replacement_audits.append(audit)
        if not audit["valid"]:
            raise RuntimeError(f"task{task:03d} replacement failed: {audit['errors']}")
        replacement_bytes[task] = data

    temp_output = output.with_suffix(output.suffix + ".tmp")
    try:
        with zipfile.ZipFile(baseline, "r") as source:
            source_infos = source.infolist()
            source_names = [info.filename for info in source_infos]
            source_tasks = [task_from_name(name) for name in source_names]
            task_ids = [task for task in source_tasks if task is not None]
            if len(task_ids) != 400 or sorted(task_ids) != list(range(1, 401)):
                raise RuntimeError("baseline must contain exactly one member for every task001..400")
            if len(set(task_ids)) != 400:
                raise RuntimeError("baseline contains duplicate task members")
            missing_replacements = sorted(set(replacements) - set(task_ids))
            if missing_replacements:
                raise RuntimeError(f"replacement tasks missing from baseline: {missing_replacements}")

            source_payloads = {info.filename: source.read(info.filename) for info in source_infos}
            with zipfile.ZipFile(temp_output, "w") as target:
                target.comment = source.comment
                for info, task in zip(source_infos, source_tasks, strict=True):
                    data = replacement_bytes.get(task, source_payloads[info.filename])
                    target.writestr(copy.copy(info), data)

        os.replace(temp_output, output)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    with zipfile.ZipFile(baseline, "r") as source, zipfile.ZipFile(output, "r") as built:
        source_infos = source.infolist()
        built_infos = built.infolist()
        names_equal = [info.filename for info in source_infos] == [info.filename for info in built_infos]
        comments_equal = source.comment == built.comment
        metadata_equal = all(
            comparable_info(left) == comparable_info(right)
            for left, right in zip(source_infos, built_infos, strict=True)
        )
        changed_tasks: list[int] = []
        unchanged_payloads_equal = True
        for left in source_infos:
            task = task_from_name(left.filename)
            before = source.read(left.filename)
            after = built.read(left.filename)
            if before != after:
                if task is None or task not in replacements:
                    unchanged_payloads_equal = False
                elif task not in changed_tasks:
                    changed_tasks.append(task)

        integrity_error = built.testzip()

    valid = bool(
        names_equal
        and comments_equal
        and metadata_equal
        and unchanged_payloads_equal
        and integrity_error is None
        and sorted(changed_tasks) == sorted(replacements)
    )
    return {
        "baseline": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "replacement_audits": replacement_audits,
        "changed_tasks": sorted(changed_tasks),
        "member_order_equal": names_equal,
        "archive_comment_equal": comments_equal,
        "member_metadata_equal": metadata_equal,
        "unchanged_payloads_equal": unchanged_payloads_equal,
        "integrity_error": integrity_error,
        "valid": valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="append", default=[], metavar="TASK=MODEL")
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()

    replacements: dict[int, Path] = {}
    for item in args.replace:
        if "=" not in item:
            parser.error(f"invalid --replace {item!r}; expected TASK=MODEL")
        task_text, path_text = item.split("=", 1)
        task = int(task_text)
        if task in replacements:
            parser.error(f"duplicate replacement task {task}")
        replacements[task] = Path(path_text)
    if not replacements:
        parser.error("at least one --replace is required")

    audit = build(args.baseline, args.output, replacements)
    rendered = json.dumps(audit, indent=2, default=str)
    print(rendered)
    if args.audit_json:
        args.audit_json.write_text(rendered + "\n")
    return 0 if audit["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
