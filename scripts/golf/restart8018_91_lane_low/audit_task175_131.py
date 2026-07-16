#!/usr/bin/env python3
"""Run the common fail-closed task175 audit on the cost-131 exact factor reuse."""

from __future__ import annotations

import audit_task175 as audit


audit.CANDIDATE = audit.HERE / "candidates" / "task175_gauge_s_factor_reuse.onnx"
audit.EVIDENCE = audit.HERE / "task175_cost131_evidence.json"
audit.CANDIDATE_SHA256 = "22fe38f6428dbc2f98b7135825325044f1898a7da23e2bea9b7584d97bfe4265"
audit.EXPECTED_COST = 131
audit.SEEDS = (801_891_177, 801_891_178)


if __name__ == "__main__":
    raise SystemExit(audit.main())
