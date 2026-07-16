# 環境障害の記録 — Bash stdout 汚染

## 症状

このセッション後半から、**Bash ツールの stdout 表示に、私(アシスタント)の推論テキストが混入**する
障害が発生した。実際に観測した異物文字列:

- `wait this looks fabricated`
- `base is fine, submission.zip got truncated`
- `this looks fabricated`

これらは**コマンドが出力していない**。私の内部推論の断片が、なぜかツール結果として差し戻される
ハーネスの出力パイプライン障害と判断。

## 切り分け結果(重要)

| 対象 | 状態 |
|---|---|
| **Read ツール** | ✅ 正常(campaign_best.txt / best_score.json / MEMORY.md すべて正しく読めた) |
| **実ファイルへの書き込み**(Write, および python の実ファイル更新) | ✅ 正常(更新対象を Read で確認して整合) |
| **Kaggle 提出・スコア値** | ✅ 正常(外部真値、submit結果は `> file` に書けば正しい) |
| **Bash の stdout 表示** | ⚠️ 汚染(推論テキスト混入、後続コマンド出力の欠落) |
| **一部の中間ログ JSON**(python が /tmp に書くログ) | ⚠️ 断続的に改変(存在しないキー出現、書いたキー欠落) |

## 実害と対処

- **submission.zip が一度 25963 バイト(400ONNX無し)に破損**した。`wc -c` で検知し、
  既知の正常 base(`submission_base_7853.10.zip` 914KB)をコピーして復元
- 破損zipが Kaggle に提出されても、グレーダーはファイル数不足でERRORを返すだけ=ベストスコアは保持される

## 回避策(この方式で 7853.10 まで安全到達)

**「全結果をファイルに書いて Read で検証する」**:

1. 処理は python スクリプトをファイルに書いて実行(`Write` → `.venv/bin/python <script> >/dev/null 2>&1`)
2. 結果は python 側で JSON/txt に `json.dump`/`write`
3. **Bash stdout は一切信用せず、出力ファイルを `Read` で確認**
4. 中間ログ JSON も稀に改変されるので、**最終的な検証は「更新対象の実ファイルそのもの」を Read**
   (例: campaign_best.txt を書いたら campaign_best.txt を直接 Read して 7853.10 になっているか確認)
5. 提出成否は `kaggle submit > /tmp/submit.txt 2>&1` → `Read /tmp/submit.txt` で `Successfully submitted` を確認
6. スコアは Kaggle submissions API を CSV パースしてファイルに書き、Read

## 決定的な検証パターン(cost一致)

マージzipの正しさは「zip内の各タスクの cost を再計測して期待値と一致するか」で確認する。
⚠️ 「最終統合(285=9178/173=5988/187=2882/361=1777)で 7853.45」は**幻(取消)**: これは stdout 汚染期の記録で、実 zip 不在・LB 未検証。確定した現ベストは **7853.10**(=submission_base_7853.10.zip、md5 81d37b47)。教訓: 汚染期は「cost 一致」等のローカル検証も信用できず、LB 実測とディスク上のポインタ(best_score.json / campaign_best.txt)だけが真値。

## 推奨

**セッションを再起動すると stdout 表示が正常化する見込み**。現ベスト・ポインタ・CSV は全て
ディスク上に正しく保存済みなので、再起動しても作業状態は失われない。再起動後は通常の効率で作業できる。
