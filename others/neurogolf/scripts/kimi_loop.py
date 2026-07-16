#!/usr/bin/env python3
"""Kimi golf loop — task12-type cheap band, dig EACH task until +1 point, advance.

Per user direction:
  - score=max(1,25-ln(cost)) => a >=e (2.718x) cost cut earns +1 full point.
    Cheap/simple tasks (task12 archetype) reach +1 with the smallest absolute
    cut, and "+1 on all 400 => +400" is the macro plan.
  - No kill-timeout: a Kimi worker is NOT SIGKILLed mid-work; it ends on its own
    stop rules. A large per-round hang-guard only prevents true hangs.
  - Dig each task until it gains >=1.0 score point (capped at MAX_ROUNDS Kimi
    re-invocations), then move to the next cheapest OPEN task.
  - Full ONNX rebuild is allowed (encoded in worker_prompt.py).

Separate lane from Codex: own scratch (scratch_kimi/ via kimi_wave sed), own logs
(kimi_logs/), disjoint targets (kimi_exclude.json), own process. Shared only the
result store handcrafted/ via try_candidate (monotonic + visible-gold = safe).
NO auto-submit (the LB judge is run manually via the real-cost aggressive build).

Env: KIMI_CONCURRENCY(3) KIMI_MAX_ROUNDS(3) KIMI_MAX_TASKS(60)
     KIMI_HANG_GUARD_SEC(3600) GAIN_TARGET(1.0)
"""
from __future__ import annotations
import json, math, os, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import onnx  # noqa: E402
from lib import scoring  # noqa: E402

ASK_KIMI = Path.home() / ".claude" / "bin" / "ask-kimi"
SUB12 = REPO / "artifacts" / "sub12_base"
OPT = REPO / "artifacts" / "optimized"
HC = REPO / "artifacts" / "handcrafted"
LOGDIR = REPO / "artifacts" / "kimi_logs"
LOOPLOG = LOGDIR / "loop.log"
ATTEMPTED = REPO / "docs" / "golf" / "kimi_attempted.json"

CONCURRENCY = int(os.environ.get("KIMI_CONCURRENCY", "3"))
MAX_ROUNDS = int(os.environ.get("KIMI_MAX_ROUNDS", "3"))
MAX_TASKS = int(os.environ.get("KIMI_MAX_TASKS", "60"))
HANG_GUARD = int(os.environ.get("KIMI_HANG_GUARD_SEC", "3600"))
GAIN_TARGET = float(os.environ.get("GAIN_TARGET", "1.0"))

_lock = threading.Lock()


def sc(c: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, c)))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}\n"
    with _lock:
        with open(LOOPLOG, "a") as f:
            f.write(line)


def _cost(path: Path, task: int, require_correct: bool) -> int | None:
    if not path.is_file():
        return None
    with tempfile.TemporaryDirectory(prefix="kloop_") as wd:
        try:
            r = scoring.score_and_verify(onnx.load(str(path)), task, wd,
                                         label="kloop", require_correct=require_correct)
        except Exception:
            return None
    return int(r["cost"]) if r else None


def incumbent_cost(task: int) -> int:
    """Best real cost over the result store: sub12 (official-correct fallback) and
    any locally-correct handcrafted/optimized net."""
    cands = []
    s = _cost(SUB12 / f"task{task:03d}.onnx", task, require_correct=False)
    if s is not None:
        cands.append(s)
    for d in (HC, OPT):
        c = _cost(d / f"task{task:03d}.onnx", task, require_correct=True)
        if c is not None:
            cands.append(c)
    return min(cands) if cands else 10**9


def make_prompt(task: int, thash: str, cost: int) -> str:
    raw = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/factory/worker_prompt.py"),
         str(task), thash, str(cost)],
        capture_output=True, text=True, check=True).stdout
    # Codex->Kimi self-identity + scratch isolation (same as kimi_wave.sh)
    sed = subprocess.run(
        ["sed", "s/detached Codex CLI/detached Kimi/g; s/Codex CLI/Kimi/g; "
         "s/\\bCodex\\b/Kimi/g; s#scripts/golf/scratch/task#scripts/golf/scratch_kimi/task#g"],
        input=raw, capture_output=True, text=True).stdout
    return sed


def run_kimi_once(task: int, thash: str, cost: int, round_no: int) -> None:
    (REPO / f"scripts/golf/scratch_kimi/task{task:03d}").mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(task, thash, cost)
    logpath = LOGDIR / f"task{task:03d}.log"
    with open(logpath, "ab") as lf:
        lf.write(f"\n===== KIMI round {round_no} (start {time.strftime('%H:%M:%SZ', time.gmtime())}) =====\n".encode())
        lf.flush()
        try:
            subprocess.run([str(ASK_KIMI)], input=prompt, text=True,
                           stdout=lf, stderr=subprocess.STDOUT,
                           timeout=HANG_GUARD, cwd=str(REPO))
        except subprocess.TimeoutExpired:
            lf.write(f"\n[hang-guard {HANG_GUARD}s hit; worker reaped]\n".encode())


def mark_attempted(task: int) -> None:
    with _lock:
        cur = set(json.load(open(ATTEMPTED)))
        cur.add(task)
        json.dump(sorted(cur), open(ATTEMPTED, "w"), indent=2)


def dig_task(spec: str) -> None:
    task_s, thash, cost_s = spec.split(":")
    task = int(task_s)
    mark_attempted(task)
    before = incumbent_cost(task)
    log(f"task{task:03d} START before_cost={before} score={sc(before):.3f} target=+{GAIN_TARGET}")
    best = before
    for rnd in range(1, MAX_ROUNDS + 1):
        run_kimi_once(task, thash, best, rnd)
        after = incumbent_cost(task)
        gain = sc(after) - sc(before)   # positive = improvement (lower cost -> higher score)
        log(f"task{task:03d} round {rnd}: cost {best}->{after} cumulative_gain={gain:+.3f}")
        best = after
        if gain >= GAIN_TARGET:
            log(f"task{task:03d} DONE +{gain:.3f} (>= {GAIN_TARGET}) in {rnd} round(s): {before}->{after}")
            return
        if rnd >= 2 and after >= before:
            log(f"task{task:03d} STALL (no progress in {rnd} rounds); advancing")
            return
    log(f"task{task:03d} PARTIAL after {MAX_ROUNDS} rounds: {before}->{best} gain={sc(best)-sc(before):+.3f}; advancing")


_processed = 0


def next_target() -> str | None:
    """Atomically fetch the cheapest OPEN target and claim it (mark attempted), so
    concurrent workers never grab the same task. Returns a `task:hash:cost` spec
    or None when the band is exhausted. This is the self-replenishing pull: every
    time a slot frees (a task hit +1 or stalled), it pulls one fresh target."""
    global _processed
    with _lock:  # claim atomically; do NOT call log() here (_lock is non-reentrant)
        if _processed >= MAX_TASKS:
            return None
        out = subprocess.run(
            [str(REPO / ".venv/bin/python"), str(REPO / "scripts/golf/kimi_targets.py"), "1"],
            capture_output=True, text=True).stdout.split()
        if not out:
            return None
        spec = out[0]
        task = int(spec.split(":")[0])
        cur = set(json.load(open(ATTEMPTED)))
        cur.add(task)
        json.dump(sorted(cur), open(ATTEMPTED, "w"), indent=2)
        _processed += 1
        n = _processed
    log(f"ADD #{n}: pulled task{task:03d} (self-replenish)")
    return spec


def worker(wid: int) -> None:
    while True:
        try:
            spec = next_target()
        except Exception as exc:  # noqa: BLE001
            # a transient fetch error must NOT silently kill the thread (that
            # was shrinking the pool from 8 to 3); log and retry.
            log(f"worker {wid}: next_target errored: {exc!r}; retrying")
            continue
        if spec is None:
            log(f"worker {wid}: no more OPEN targets; exiting")
            return
        try:
            dig_task(spec)
        except Exception as exc:  # noqa: BLE001
            log(f"worker {wid}: task {spec.split(':')[0]} errored: {exc!r}; continuing")


def main() -> int:
    import threading
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log(f"================ KIMI self-replenishing pool START (conc={CONCURRENCY} "
        f"rounds={MAX_ROUNDS} cap={MAX_TASKS} hang_guard={HANG_GUARD}s) ================")
    log("rule: each slot digs a task until +{:.0f} (or stall), then pulls one fresh "
        "target. repeats until the band is exhausted.".format(GAIN_TARGET))
    threads = [threading.Thread(target=worker, args=(i,), daemon=False)
               for i in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("================ KIMI self-replenishing pool END "
        f"(processed={_processed}) ================")
    print("KIMI LOOP DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
