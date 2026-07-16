#!/usr/bin/env python3
"""Isolated official-cost measurement; malformed models may crash this process."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


memory, params, cost = cost_of(sys.argv[1])
print(json.dumps({"memory": int(memory), "params": int(params), "cost": int(cost)}))
