#!/usr/bin/env python3
"""Independent SOUND audit of scan155 CastLike-to-Cast candidates.

This lane is read-only with respect to the submission and score ledgers.  It
pins every input SHA, measures competition-style actual cost on every known
case with two independent profilers, and refuses to run fresh unless all
static, runtime-shape, four-configuration, and strict-lower gates pass.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, defs, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ROOT_SCORES = ROOT / "all_scores.csv"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
ROOT_SCORES_SHA256 = "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78"
TASKS = (71, 133, 216, 285, 388)
EXPECTED = {
    71: ("61798cc38df4cde5275141ce77900eb47fb61f83987a86ba7ad36fd38fb749a6", "1abcca8e1b56070a40e8f2c86335b2af7b782148491a6d2c2f97c9991d7d2e6c"),
    133: ("6c5dc3a593b0900e16966b9d4c40af509a34c1dd1f0264c31cd30eaf9b4570e5", "34fe1c446e77782b7c1268d115198d3e21e7156b3f4284165cc9d3aa015331b3"),
    216: ("9a5f4f10d6e014b3f053ce1dabeb39cbeaf95964ae685aa71514fd695caf0756", "c86015a124a0a1a21872f708e029edeb6ec894c014819765bdca7cb99695eecf"),
    285: ("366212e29105fde0295030f3ec3bb014bd300f23aa8259ccd79da2eea720b9e2", "3eec54cb97d9ef117ab63837ad4f3ed75835b474682d668356109d9e4165169f"),
    388: ("f27fa5f4f0bcade23d02fed2a74e3c2b826b11140bd03d29f47e0c59c382a8e1", "aca14029d7b152eaeed10994de21f473ff520de99c858353a68a281d0c1e6496"),
}
PRIVATE = {133: "confirmed-private-0", 216: "unsound-incumbent-monitor", 285: "unsound-incumbent-monitor"}
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE_AUDIT = load_module(
    "castlike156_base_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
BASE_SCAN = load_module(
    "castlike156_base_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
TEAM = load_module(
    "castlike156_team_validator",
    ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def quiet_team_session(model: onnx.ModelProto, profile_prefix: str | Path | None = None) -> ort.InferenceSession:
    """Team-validator session contract with diagnostics silenced, not gates."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.log_severity_level = 4
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


TEAM.make_session = quiet_team_session


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha(path.read_bytes())


def dtype_name(dtype: int) -> str:
    return TensorProto.DataType.Name(dtype)


def opset(model: onnx.ModelProto) -> int:
    return max(int(item.version) for item in model.opset_import if item.domain in ("", "ai.onnx"))


def attr_int(node: onnx.NodeProto, name: str) -> int | None:
    for item in node.attribute:
        if item.name == name:
            return int(helper.get_attribute_value(item))
    return None


def saturate_default(op: str, version: int) -> int | None:
    schema = defs.get_schema(op, version, "")
    attribute = schema.attributes.get("saturate")
    if attribute is None or not attribute.default_value.HasField("i"):
        return None
    return int(attribute.default_value.i)


def effective_saturate(node: onnx.NodeProto, version: int) -> int | None:
    explicit = attr_int(node, "saturate")
    return explicit if explicit is not None else saturate_default(node.op_type, version)


def formal_rewrite_proof(base_data: bytes, cand_data: bytes) -> dict[str, Any]:
    base = onnx.load_model_from_string(base_data)
    cand = onnx.load_model_from_string(cand_data)
    version = opset(base)
    if len(base.graph.node) != len(cand.graph.node):
        return {"pass": False, "reason": "node_count_changed"}
    changed = [i for i, (left, right) in enumerate(zip(base.graph.node, cand.graph.node)) if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True)]
    if len(changed) != 1:
        return {"pass": False, "reason": "not_one_changed_node", "changed": changed}
    index = changed[0]
    before = base.graph.node[index]
    after = cand.graph.node[index]
    initializers = {item.name: item for item in base.graph.initializer}
    witness = initializers.get(before.input[1]) if len(before.input) == 2 else None
    target = int(witness.data_type) if witness is not None else None
    candidate_to = attr_int(after, "to")
    before_sat = effective_saturate(before, version)
    after_sat = effective_saturate(after, version)
    float8_targets = {
        getattr(TensorProto, name)
        for name in (
            "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ",
            "FLOAT4E2M1", "FLOAT8E8M0",
        )
        if hasattr(TensorProto, name)
    }
    fields_ok = bool(
        before.op_type == "CastLike"
        and after.op_type == "Cast"
        and witness is not None
        and list(after.input) == [before.input[0]]
        and list(after.output) == list(before.output)
        and after.name == before.name
        and candidate_to == target
        and before.domain == after.domain
    )
    saturate_equivalent = bool(target not in float8_targets or before_sat == after_sat)

    expected = copy.deepcopy(base)
    node = expected.graph.node[index]
    node.op_type = "Cast"
    del node.input[:]
    node.input.extend([before.input[0]])
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("to", target)])
    still_used = {name for item in expected.graph.node for name in item.input if name}
    kept = [item for item in expected.graph.initializer if item.name in still_used]
    del expected.graph.initializer[:]
    expected.graph.initializer.extend(kept)
    exact_expected_bytes = expected.SerializeToString(deterministic=True) == cand.SerializeToString(deterministic=True)
    return {
        "pass": bool(fields_ok and saturate_equivalent and exact_expected_bytes),
        "changed_node_index": index,
        "before": {"name": before.name, "op": before.op_type, "inputs": list(before.input), "outputs": list(before.output)},
        "after": {"name": after.name, "op": after.op_type, "inputs": list(after.input), "outputs": list(after.output)},
        "witness": before.input[1] if witness is not None else None,
        "target_dtype": dtype_name(target) if target is not None else None,
        "candidate_to": candidate_to,
        "opset": version,
        "castlike_saturate": before_sat,
        "cast_saturate": after_sat,
        "saturate_relevant": target in float8_targets,
        "saturate_equivalent": saturate_equivalent,
        "exact_expected_model": exact_expected_bytes,
        "semantic_statement": "CastLike(x, fixed witness) casts x to witness element type; Cast(x,to=witness_dtype) has the same numeric conversion. Saturate only affects float8-like targets and is irrelevant for this target.",
    }


def scoring_profile(task: int, data: bytes, label: str) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"castlike156_{task:03d}_", dir="/tmp") as work:
        result = scoring.score_and_verify(copy.deepcopy(model), task, work, label=label, require_correct=False)
    if result is None:
        raise RuntimeError(f"score_and_verify returned None: {label}")
    return result


def team_profile(task: int, data: bytes, label: str) -> dict[str, Any]:
    audit, failures = TEAM.audit_model_bytes(
        data,
        task,
        ROOT / "inputs/neurogolf-2026",
        source=label,
        keep_trace=False,
        trace_dir=HERE / "traces",
    )
    row = dataclasses.asdict(audit)
    row["failures"] = failures
    return row


def structural_inventory(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    row = BASE_SCAN.structural(copy.deepcopy(model))
    hist = Counter(node.op_type for node in model.graph.node)
    max_einsum_inputs = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    lookup = sorted({node.op_type for node in model.graph.node if node.op_type in {"TfIdfVectorizer", "CategoryMapper"}})
    row.update(
        {
            "op_histogram": dict(sorted(hist.items())),
            "max_einsum_inputs": max_einsum_inputs,
            "giant_einsum": max_einsum_inputs > 8,
            "lookup_ops": lookup,
            "lookup_free": not lookup,
        }
    )
    return row


def runtime_trace(task: int, data: bytes) -> dict[str, Any]:
    try:
        return BASE_AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    root_before = {"submission": sha_file(ROOT_SUBMISSION), "all_scores": sha_file(ROOT_SCORES)}
    if sha_file(AUTHORITY) != AUTHORITY_SHA256 or root_before["submission"] != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority/root submission changed")
    if root_before["all_scores"] != ROOT_SCORES_SHA256:
        raise RuntimeError("root all_scores hash changed")

    result: dict[str, Any] = {
        "lane": "agent_castlike_exact_156",
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "root_before": root_before,
        "policy": {
            "known": "all known train/test/arc-gen; default/disable-all x threads1/4; candidate and authority correct; raw/sign equal; errors/nonfinite zero",
            "runtime_shape": "strict inferred declaration checked against every node output on the first legal known input",
            "fresh": "two independent >=1500-case seeds only after all pre-gates; >=90% each; private/monitor additionally needs authority all-input pass-through closure",
        },
        "tasks": {},
        "winners": [],
    }
    for task in TASKS:
        base_data = (HERE / "baseline" / f"task{task:03d}.onnx").read_bytes()
        cand_data = (HERE / "candidates" / f"task{task:03d}.onnx").read_bytes()
        if (sha(base_data), sha(cand_data)) != EXPECTED[task]:
            raise RuntimeError(f"pinned task{task:03d} SHA mismatch")
        print(f"task{task:03d}: profile authority", flush=True)
        base_scoring = scoring_profile(task, base_data, f"castlike156_base_{task:03d}")
        base_team = team_profile(task, base_data, f"castlike156_base_{task:03d}")
        print(f"task{task:03d}: profile candidate", flush=True)
        cand_scoring = scoring_profile(task, cand_data, f"castlike156_cand_{task:03d}")
        cand_team = team_profile(task, cand_data, f"castlike156_cand_{task:03d}")
        static = structural_inventory(cand_data)
        base_trace = runtime_trace(task, base_data)
        trace = runtime_trace(task, cand_data)
        known = {
            label: BASE_AUDIT.known_config(task, base_data, cand_data, disable, threads)
            for disable, threads, label in CONFIGS
        }
        formal = formal_rewrite_proof(base_data, cand_data)
        scoring_agrees = base_scoring["cost"] == base_team["cost"] and cand_scoring["cost"] == cand_team["cost"]
        actual_lower = cand_scoring["cost"] < base_scoring["cost"] and cand_team["cost"] < base_team["cost"]
        known4 = all(item.get("perfect", False) for item in known.values())
        policy = PRIVATE.get(task, "not-catalogued")
        pass_through_closed = bool(formal.get("pass") and trace.get("truthful") and known4)
        reasons: list[str] = []
        if not static.get("pass", False):
            reasons.append("static_structure_failed")
        if not trace.get("truthful", False):
            reasons.append("runtime_node_shapes_not_truthful")
        if not known4:
            reasons.append("known_four_config_failed")
        if not scoring_agrees:
            reasons.append("independent_actual_profilers_disagree")
        if not actual_lower:
            reasons.append("competition_actual_not_strict_lower")
        if not formal.get("pass", False):
            reasons.append("formal_rewrite_proof_failed")
        if policy != "not-catalogued" and not pass_through_closed:
            reasons.append("private_monitor_pass_through_not_closed")
        accepted_pre_fresh = not reasons
        fresh = {
            "status": "not_run",
            "reason": "rejected_before_fresh" if not accepted_pre_fresh else "required_but_not_implemented_error",
            "seeds": [],
        }
        if accepted_pre_fresh:
            raise RuntimeError(f"task{task:03d} unexpectedly reached fresh gate")
        result["tasks"][f"{task:03d}"] = {
            "authority_sha256": sha(base_data),
            "candidate_sha256": sha(cand_data),
            "private_zero_policy": policy,
            "formal_semantics": formal,
            "declared_profiles": {
                "authority": BASE_SCAN.official_cost(base_data, f"castlike156_decl_base_{task:03d}"),
                "candidate": BASE_SCAN.official_cost(cand_data, f"castlike156_decl_cand_{task:03d}"),
            },
            "competition_actual": {
                "scoring_authority": base_scoring,
                "scoring_candidate": cand_scoring,
                "team_authority": base_team,
                "team_candidate": cand_team,
                "independent_costs_agree": scoring_agrees,
                "strict_lower": actual_lower,
            },
            "structural": static,
            "runtime_node_shape_trace": {"authority": base_trace, "candidate": trace},
            "known_four_configs": known,
            "authority_all_input_pass_through_closed": pass_through_closed,
            "fresh": fresh,
            "accepted": False,
            "reasons": reasons,
        }
        (HERE / "result.partial.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"task{task:03d}: actual {base_scoring['cost']}->{cand_scoring['cost']} known4={known4} truthful={trace.get('truthful')} reasons={reasons}", flush=True)

    root_after = {"submission": sha_file(ROOT_SUBMISSION), "all_scores": sha_file(ROOT_SCORES)}
    result["root_after"] = root_after
    result["root_unchanged"] = root_before == root_after
    result["projected_gain"] = 0.0
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
