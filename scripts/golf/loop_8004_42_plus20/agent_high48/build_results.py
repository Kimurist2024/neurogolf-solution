#!/usr/bin/env python3
"""Build final machine-readable decisions for high48."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TASKS = (8, 275, 134, 112, 168, 109, 160, 170)
RULE_TYPES = {
    8: "global translation/bounding-box geometry",
    275: "global Kronecker substitution",
    134: "private-risk global role/scale inference (Type D)",
    112: "private-risk data-dependent four-way reflection (Type D)",
    168: "bounded diagonal propagation (Type B)",
    109: "global pivot removal and four-way reflection",
    160: "local exact 3x3 sprite rewrite (Type A)",
    170: "private-risk global mosaic decode/masking (Type D)",
}
BEST_PROBES = {
    8: "static 427 lead has truthful actual cost 22021; static 430 lead costs 454, so neither beats 431",
    275: "cost 317 archive model is only 128/266 known and uses a 41-input giant Einsum",
    134: "cost 320/322 known100 models use lookup machinery and score only 4840/5000+4823/5000 and 4803/5000+4825/5000 fresh",
    112: "no retained lower model; ordinary truthful control costs 19891 and the compact cost-498 control is dominated/unsafe",
    168: "cost 166 and 285 known100 models use lookup plus giant contraction; ordinary sound control costs 20403",
    109: "all static 177..193 archive leads reprofile to >=273/unscorable and are shape cloaks; ordinary truthful control costs 28110",
    160: "cost 384/384/402 candidates are 0/265 known in both ORT modes; truthful local control costs 2978",
    170: "no retained lower model; ordinary truthful control costs 24855 and historical private-zero model costs 691",
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    known = json.loads((HERE / "known_baseline_dual.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    history = json.loads((HERE / "history_audit.json").read_text())
    factors = json.loads((HERE / "exact_factor_audit.json").read_text())
    base_by_task = {int(row["task"]): row for row in baseline["targets"]}
    known_by_task = {int(row["task"]): row for row in known["rows"]}
    rule_by_task = {int(row["task"]): row for row in rules["rows"]}
    factor_by_task = {int(row["task"]): row for row in factors["rows"]}
    hist_by_task = {
        task: [row for row in history["rows"] if int(row["task"]) == task]
        for task in TASKS
    }
    rows = []
    for task in TASKS:
        cheaper = [row for row in hist_by_task[task] if row["strictly_cheaper"]]
        rows.append(
            {
                "task": task,
                "rule_type": RULE_TYPES[task],
                "rule_known": rule_by_task[task]["known"],
                "baseline_member": base_by_task[task]["member"],
                "baseline_sha256": base_by_task[task]["sha256"],
                "baseline_cost": base_by_task[task]["actual_cost"],
                "baseline_known_dual": known_by_task[task],
                "baseline_structure": base_by_task[task]["structure"],
                "history_models_screened": len(hist_by_task[task]),
                "history_strictly_cheaper": len(cheaper),
                "history_known100_cheaper": sum(bool(row["known100_dual"]) for row in cheaper),
                "history_safe_pre_fresh": sum(bool(row["pre_fresh_finalist"]) for row in cheaper),
                "exact_factor_audit": factor_by_task[task],
                "best_probe": BEST_PROBES[task],
                "candidate": None,
                "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
                "candidate_known_dual": "NOT_RUN_NO_SAFE_CANDIDATE",
                "candidate_fresh_seed_1_dual": "NOT_RUN_NO_SAFE_CANDIDATE",
                "candidate_fresh_seed_2_dual": "NOT_RUN_NO_SAFE_CANDIDATE",
                "projected_gain": 0.0,
            }
        )
    result = {
        "lane": "agent_high48",
        "baseline": baseline["baseline"],
        "policy": {
            "normal": "strictly lower official-like actual cost, known100 dual, fresh>=90% on two independent >=2000 sets, runtime0, full checker, strict/data_prop, truthful shapes, standard domain, no lookup/cloak/giant, Conv UB0, margin",
            "private_risk": "known/fresh100 dual plus decoded true-rule guarantee, with every normal structural gate",
            "fail_closed": "fresh starts only after cost, structure, and known gates; no candidate reached that stage",
        },
        "targets_requested": list(TASKS),
        "targets_completed": len(rows),
        "history_coverage": history["coverage"],
        "history_models_screened": history["models_screened"],
        "history_strictly_cheaper": history["strictly_cheaper"],
        "safe_pre_fresh_finalists": len(history["pre_fresh_finalists"]),
        "reused_fresh_evidence": {
            "task134_r04": "scripts/golf/loop_7999_13/lane_b10/task134_r04_fresh5000.json",
            "task134_r06": "scripts/golf/loop_7999_13/lane_b10/task134_r06_fresh5000.json",
            "task168_current": "scripts/golf/loop_7999_13/lane_sound/winner_manifest.json",
            "task275_current": "scripts/golf/loop_7999_13/lane_a29/task275_shared_gate_router_fresh5000.json",
        },
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
        "zip_integration": False,
        "rows": rows,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "lane": "agent_high48",
                "baseline": baseline["baseline"],
                "accepted": [],
                "accepted_count": 0,
                "projected_gain": 0.0,
                "zip_integration": False,
                "verdict": "NO_SAFE_CANDIDATE",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
