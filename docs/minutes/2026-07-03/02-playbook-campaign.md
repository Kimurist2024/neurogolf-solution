# プレイブック — codex / Fable キャンペーンの運用

`scripts/gpt_rebuild.sh` を使って、cost削減ワーカーを N 本並走させる。採用は `try_candidate.py`
経由(strictly-cheaper + correct + margin)だけなので**リスクフリー**(現ベストは絶対に壊れない)。

## モードとエンジン

- `GR_MODE=rebuild`(既定): ルールを1から再構築して小型ONNX化(高コスト帯・床未確定タスク向け)
- `GR_MODE=memshave`: incumbent保存でmemory footprintだけ削る(dtypeナローイング/ノード融合/定数畳込)。
  ヒントは `docs/golf/gpt_hints_memshave.md` が自動付与
- `GR_ENGINE=codex`(GPT-5.5)/ `claude`(Fable)/ `kimi`

## 起動(必ずリポジトリ直下から)

### 前提: PATH
Bashツールの PATH には `/opt/homebrew/bin`(codex)と `$HOME/.local/bin`(claude)が無い。**起動前に前置**:
```bash
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
```

### ターゲットは必ず最新CSVから再生成(ユーザー厳命「ターゲットは最新に」)

```bash
# 例: cost≤150 帯(マイクロ帯、1param削減が高効率)
python3 -c "
import csv, json
h=json.load(open('docs/golf/task_hash_map.json'))
rows=[r for r in csv.DictReader(open('all_scores.csv')) if 10 < int(r['cost']) <= 150]
rows.sort(key=lambda r:-int(r['cost']))
json.dump([{'task':int(r['task'][4:]),'hash':h[r['task'][4:]],'cost':int(r['cost'])} for r in rows],
          open('docs/golf/gpt_targets_le150.json','w'), indent=1)"
echo '[]' > docs/golf/gpt_assigned.json     # 担当リストをリセット
```

### memshave ターゲット(72件、床除外・mem≥300・gain見込み≥0.15)
`docs/golf/mem_targets.json` を現ベストの実ネットから再生成する(00-summaryの手順 or メモリ memshave-campaign 参照)。

### 起動コマンド

```bash
# codex マイクロ帯
GR_ENGINE=codex GR_SLOTS=6 GR_MODE=rebuild \
GR_TARGETS_FILE=docs/golf/gpt_targets_le150.json GR_TARGET_GOAL=10 GR_GOAL=10 \
GR_HINTS=docs/golf/gpt_hints_micro50.md GR_TIMEOUT=1500 GR_HOURS=8 \
GR_LOGDIR=artifacts/gpt_rebuild_logs_le150 \
nohup bash scripts/gpt_rebuild.sh > artifacts/gpt_rebuild_logs_le150_nohup.out 2>&1 &

# Fable memshave
GR_ENGINE=claude GR_MODEL=claude-fable-5 GR_MODE=memshave GR_SLOTS=4 \
GR_TARGETS_FILE=docs/golf/mem_targets.json GR_TARGET_GOAL=400 GR_GOAL=400 \
GR_TIMEOUT=4200 GR_HOURS=8 \
GR_LOGDIR=artifacts/gpt_rebuild_logs_fable_memshave \
nohup bash scripts/gpt_rebuild.sh > artifacts/gpt_rebuild_logs_fable_memshave_nohup.out 2>&1 &
```

- **並走OK**(プールが交わらなければ)。`GR_LOGDIR` を分けて `.pids` を分離すること(必須)
- Fable は `--verbose --output-format stream-json` でログが逐次成長(kill後も残る)
- Fable の memshave 大物は 70分(4200s)でも採用まで届かないことがある

## 停止

```bash
kill <masterPID>                       # ps -eo pid,etime,command | grep gpt_rebuild
pgrep -f "codex exec" | xargs kill -KILL 2>/dev/null    # codexワーカー
pgrep -f "claude -p" | xargs kill -KILL 2>/dev/null     # Fableワーカー
# master kill 後の孤児(launch subshell / scan孤児)も掃除
```

## 回収(採用分をマージ)

キャンペーンの採用は `artifacts/handcrafted/task*.onnx` に溜まる。
→ [01-playbook-harvest.md](01-playbook-harvest.md) の handcrafted スキャン → k=100 → マージへ。

**重要**: `try_candidate.py` は handcrafted の**旧ネットと比較**して採用する。base が前進すると
handcrafted の古い高コストネットが残り「偽MOVED_ON(base以下なのに採用扱い)」を起こす。
定期的に **handcrafted を base zip と同期**して偽採用の温床を消すこと。

## トラブル: codex が切れる

- **usage-limit**(ChatGPT/Codexサブスク枠枯渇): ログに `You've hit your usage limit ... try again at H:MM`。
  リセット時刻まで待てば master 放置で自動再開(8h予算内)。`codex_quota.py` は検知できない
- **`refresh_token_invalidated`**(Your session has ended / refresh token revoked): **自己回復しない別物**。
  `codex login status` は嘘をつく。確定診断は実打ち:
  ```bash
  export PATH="/opt/homebrew/bin:$PATH"
  codex exec -s read-only --skip-git-repo-check "Reply with exactly: OK"
  # → 401 Unauthorized / refresh token was revoked が出たら復旧はユーザーの `codex login` のみ
  ```
  この状態でcampaignを走らせるとワーカーが~30秒で即死し空回りする
