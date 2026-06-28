#!/usr/bin/env python3
"""
提出 → LBポーリング → 一致確認 → CSV自動更新

使い方:
  python3 scripts/golf/submit_and_update.py \
      --message "説明" \
      --expected 7656.28 \
      --updates /tmp/csv_update_plan.json \
      --best 7651.11 \
      [--threshold 0.5]
"""
import argparse, csv, json, math, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KAGGLE_PYTHON = '/opt/anaconda3/bin/python3'

def sc(c): return max(1.0, 25 - math.log(c)) if c > 0 else 0


def submit(zip_path, message):
    r = subprocess.run(
        ['kaggle', 'competitions', 'submit', '-c', 'neurogolf-2026',
         '-f', zip_path, '-m', message],
        capture_output=True, text=True, cwd=str(REPO)
    )
    print(r.stdout.strip())
    if r.returncode != 0:
        print(f"提出失敗: {r.stderr.strip()}")
        return False
    return True


def get_latest_lb(retries=20, wait=15):
    """最新提出のLBスコアをポーリング取得"""
    script = (
        "import kaggle, json; api=kaggle.KaggleApi(); api.authenticate();"
        "subs=api.competition_submissions('neurogolf-2026');"
        "s=subs[0]; print(json.dumps({'status':str(s.status),'score':str(s.public_score)}))"
    )
    for i in range(retries):
        r = subprocess.run([KAGGLE_PYTHON, '-c', script], capture_output=True, text=True)
        if r.returncode != 0:
            time.sleep(wait); continue
        try:
            d = json.loads(r.stdout.strip())
            status = d.get('status', '')
            score_str = d.get('score', 'None')
            print(f"  [{i+1}/{retries}] status={status} score={score_str}")
            if status not in ('PENDING', 'RUNNING') and score_str not in ('None', 'null', ''):
                return float(score_str)
        except Exception as e:
            print(f"  parse error: {e}")
        time.sleep(wait)
    return None


def update_csv(updates, label):
    csv_path = REPO / 'all_scores.csv'
    shutil.copy(str(csv_path), str(csv_path) + f'.bak_{label}')
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))
    updated = 0
    for row in rows:
        try:
            tid = int(row['task'].replace('task', ''))
        except:
            continue
        if tid in updates:
            nc = updates[tid]
            row['cost'] = str(nc)
            row['score'] = f"{sc(nc):.4f}"
            updated += 1
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    total = sum(float(r['score']) for r in rows if r.get('score'))
    print(f"CSV更新: {updated}件 / 合計スコア: {total:.3f}")
    return total


def update_best(lb):
    best_file = REPO / 'docs/golf/campaign_best.txt'
    lines = best_file.read_text().splitlines() if best_file.exists() else []
    lines.append(f"submission.zip\t{lb:.2f}")
    best_file.write_text('\n'.join(lines) + '\n')
    print(f"campaign_best.txt更新: {lb:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--message', '-m', required=True)
    p.add_argument('--expected', '-e', type=float, required=True)
    p.add_argument('--updates', '-u', required=True)
    p.add_argument('--best', '-b', type=float, default=0.0)
    p.add_argument('--threshold', '-t', type=float, default=0.5)
    p.add_argument('--zip', default='submission.zip')
    args = p.parse_args()

    updates = {int(k): v for k, v in json.load(open(args.updates)).items()}
    zip_path = str(REPO / args.zip)

    print(f"=== 提出 ===")
    if not submit(zip_path, args.message):
        sys.exit(1)

    print(f"\n=== LBポーリング開始(期待値: {args.expected:.3f}) ===")
    lb = get_latest_lb(retries=20, wait=15)

    if lb is None:
        print("LB取得タイムアウト。手動で確認してください")
        sys.exit(1)

    print(f"\nLB実測: {lb:.3f} / 期待: {args.expected:.3f} / 誤差: {abs(lb-args.expected):.3f}")

    if lb <= args.best:
        print(f"ベスト({args.best:.3f})を超えていないのでCSV更新しません")
        sys.exit(0)

    if abs(lb - args.expected) > args.threshold:
        print(f"誤差>{args.threshold}: graderエラー/退行の可能性。CSV更新しません")
        sys.exit(0)

    print(f"\n=== 一致確認 → CSV更新 ===")
    update_csv(updates, time.strftime('%H%M%S'))
    update_best(lb)
    print("完了")


if __name__ == '__main__':
    main()
