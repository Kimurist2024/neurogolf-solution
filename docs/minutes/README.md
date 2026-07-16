# 議事録インデックス — NeuroGolf 2026

このディレクトリは日付ごとの作業議事録と、他インスタンスが同じ文脈・手順を再現するための
プレイブック/TODO を階層構造でまとめる。

## 構成

```
docs/minutes/
├── README.md                         ← このファイル(目次)
└── 2026-07-03/
    ├── 00-summary.md                 セッション全体サマリ(LB推移・成果・現状)
    ├── 01-playbook-harvest.md        【最重要】外部ソース→検証→マージの標準手順
    ├── 02-playbook-campaign.md       codex/Fable キャンペーンの起動・停止・回収
    ├── 03-intel-grader-facts.md      グレーダー確定事実・スコアラトリック・禁止事項
    ├── 04-todo.md                    次にやること(優先順)
    └── 05-incident-bash-stdout.md    環境障害(Bash stdout汚染)の記録と回避策
```

## 他インスタンスへの引き継ぎ最短ルート

1. **[00-summary.md](2026-07-03/00-summary.md)** で現状(現ベスト・稼働状況)を把握
2. **[01-playbook-harvest.md](2026-07-03/01-playbook-harvest.md)** で標準ワークフローを理解
3. **[03-intel-grader-facts.md](2026-07-03/03-intel-grader-facts.md)** で採点仕様と地雷を把握
4. **[04-todo.md](2026-07-03/04-todo.md)** で次の一手を選ぶ
5. 環境が不安定なら **[05-incident-bash-stdout.md](2026-07-03/05-incident-bash-stdout.md)** の回避策を適用

## 正典(このリポジトリで信頼すべきファイル)

| ファイル | 役割 |
|---|---|
| `submission.zip` | 現在の提出物(= 最新の submission_base_XXXX.zip のコピー) |
| `docs/golf/campaign_best.txt` | `<base zip 名>\t<実LB>` の1行。マージ土台の正本 |
| `best_score.json` | 現ベストの score/md5/由来 |
| `all_scores.csv` | 全400タスクの cost/score(正典計算= `scripts/golf/dump_scores.py`) |
| メモリ `current-best.md` | LB推移の全履歴 |
