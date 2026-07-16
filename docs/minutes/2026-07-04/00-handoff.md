# セッション引き継ぎ 2026-07-04

新しい素の claude セッションがゼロから続行できるように書く。**メモリ(MEMORY.md / current-best.md)が正本**で、本ファイルは今日の作業状態のスナップショット。

## 現ベスト(厳密検証済み・2026-07-04 14:00頃)

- **実LB = 7887.17**
- md5 **`a1f2bac4`** = `submission_base_7887.17.zip` = `submission.zip`(全一致確認済み)
- `docs/golf/campaign_best.txt` = `submission_base_7887.17.zip\t7887.17`
- `best_score.json` score = 7887.17
- `all_scores.csv`(メイン)+ `others/neurogolf/all_scores.csv`(副本)= md5一致・反映済み

> ⚠️ 前セッションで「幻の 7853.45」を引き継いだ事故あり。**再起動後は必ず上記ポインタを直接 Read/md5 でクロス確認**してから作業する。

## 本日の到達(7853.10 → 7887.17、+34.07)

連続で others/ ソースをマージ + reject救出 + task001最適化 + codex campaign:
70206→70207→(reject救出)→70208→70209(alisalmanrana)→70210→7301→others/3→7302→7303→7304→codex実効2件。
- **内訳: others/マージ +33.84 / codex直接 +0.23**。
- **task001: 134 → 118**(外部候補 others/7301(126)/7302(124)/7304(118)由来。codex専用は未達)。

## 🔴 稼働中のバックグラウンド(要対応)

- **cost100-300 codex campaign**: `artifacts/gpt_rebuild_logs_100300/` 、GR_TARGET_GOAL=90、5slots、3h budget(05:00Z起動 → 08:00Z=17:00頃終了)。
  - 改善は `artifacts/handcrafted/task*.onnx` に strictly-cheaper+correct のみ昇格。
  - **要ハーベスト**: budget満了後(or 途中で)、`artifacts/handcrafted` の更新分をスナップショットして `submission_base_7887.17.zip` と scan → fresh k=30 → probe → マージ提出。
  - 手順は [01-playbook-harvest](../2026-07-03/01-playbook-harvest.md) と同じ。ハーベスト例: `scan_sources_seq.py submission_base_7887.17.zip <snapshot> tag`。
  - ⚠️ codex quota自動ガードは無効(`codex_quota.py`が rate limit読めず exit0)。credits要監視。
  - 停止: `pkill -f gpt_rebuild.sh`。

## マージ手順(確立済みフロー)

「others/NNNN をマージして」と言われたら:
1. `scan_sources_seq.py submission_base_7887.17.zip others/NNNN tag` → 勝者
2. quarantine byte照合(既知私的0点の再登場を自動除外)+ 構造ゲート + 前科フラグ
3. fresh k=30(inline verify_fix バッチ or 並列)→ ADOPT/REJECT
4. **MAIN = 非前科ADOPT** を先行提出 → gap診断
5. 前科・fresh-reject は sc分離 group probe で判別 → 正答分を救出マージ
6. 各段で `submission_base_XXXX.zip` から構築 + **提出直前に md5 検証**(campaign干渉対策)
7. ベスト更新: zip保存 / campaign_best.txt / best_score.json / current-best.md / MEMORY.md / all_scores.csv再計算(dump_scores) + 副本同期

## 今日得た重要教訓(メモリにも記録済み)

1. **fresh k=30 REJECT ≠ 私的0点**: 多数の fresh-reject が LB では正答(false-reject)。捨てずに probe で救出。真の私的0点は少数(325/377/015/109誤/…)。
2. **ERROR地雷 79/324 は giant版のみ**: cost311/517の小型版は正常採点=安全。
3. **quarantineバイト照合が再犯ネットを自動捕捉**: 192@910/377@456 は同一ファイルで自動除外。
4. **micro/mid帯のgap密集(sc差<0.03)は単独arithmetic誤特定**: 7303で109を誤特定→実は377。**bisect/複数probe必須**。
5. **単一著者handcraftedは私的0点率極低**(others/3の49件ゼロ)。alisalmanrana bulkは混在(70209は26中1件)。
6. **cost同一≠同一ネット**: task072@120は既知私的0点だが、others/3の別ネット(cost120)は私的正答。都度probe。
7. **submission.zip汚染**: task001専用codex(workspace-write)がsubmission.zipを書換。**try_candidateはhandcraftedにしか書かない**。提出前md5検証で防御。
8. **cost50-100帯はothers/マージで枯渇**(codex 52出力中2件のみ勝ち)。cost100-300帯に移行中。

## 提出で「よく省く」常連タスク(cost≥1000中心)

101(6818,★4)/077(3671,★4)/086(1215,★4)/169(712,★4)/233(15147,★3)/192(3352,常連)/325(388/312とも黒)/145/365/366/090。外部候補が来ても**必ず単発/groupプローブ**で確認してから採否。

## 環境メモ

- codex起動は PATH に `/opt/homebrew/bin` 必須。`codex exec -m gpt-5.5 -s workspace-write`。
- 提出: `.venv/bin/kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "<msg>"`。メッセージにカンマ不可(CSVパースずれ)。
- スコア確認: submissions -v をCSVパース(列: ref,fileName,date,description,status,publicScore,privateScore)。
- dump_scores は giant(306/328は>60s TIMEOUT)で低速、末尾一括書き出し。**プロセス終了ベースで待つ**(固定回数タイムアウトで誤同期する事故あり)。
