"""Read the latest Codex rate-limit usage from ~/.codex/logs_2.sqlite.

Prints `PRIMARY_REMAIN SECONDARY_REMAIN` (percent remaining) and a STOP/OK verdict.
The user's rule: STOP replenishing when remaining usage < 20% (secondary/weekly is
the binding budget; primary is a 5h throttle that resets).
Exit code 0 = OK to continue, 3 = STOP (secondary remaining < 20%).
"""
import sqlite3, json, re, os, sys

DB = os.path.expanduser("~/.codex/logs_2.sqlite")
STOP_REMAIN = 20  # stop when secondary remaining < this %


def latest_rate_limits():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT feedback_log_body FROM logs "
        "WHERE feedback_log_body LIKE '%codex.rate_limits%' ORDER BY id DESC LIMIT 1"
    ).fetchall()
    con.close()
    if not rows:
        return None
    txt = rows[0][0]
    start = txt.find('{"type":"codex.rate_limits"')
    if start < 0:
        return None
    # brace-balanced extraction of the JSON object
    depth = 0
    for i in range(start, len(txt)):
        if txt[i] == '{':
            depth += 1
        elif txt[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[start:i + 1])["rate_limits"]
                except Exception:
                    return None
    return None


def main():
    rl = latest_rate_limits()
    if not rl:
        print("UNKNOWN: could not read rate limits", file=sys.stderr)
        return 0  # don't block on read failure
    pri = 100 - rl["primary"]["used_percent"]
    sec = 100 - rl["secondary"]["used_percent"]
    pri_reset = rl["primary"]["reset_after_seconds"] // 60
    sec_reset = rl["secondary"]["reset_after_seconds"] // 3600
    verdict = "STOP" if sec < STOP_REMAIN else ("THROTTLE" if pri < 8 else "OK")
    print(f"primary_remain={pri}% (reset {pri_reset}min) "
          f"secondary_remain={sec}% (reset {sec_reset}h) verdict={verdict}")
    if sec < STOP_REMAIN:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
