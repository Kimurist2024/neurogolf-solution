# Proposal 002: Lane G 残余最適化(dtype 縮小 + no-op 除去 + ハーネス決定化)

- 提案者: Fable(メインセッション)
- 実装者: Opus サブエージェント
- 前提: proposal 001 + Amendment 1 適用済みパイプライン(run-008 = local 6347.8189)
- 根拠データ: `docs/research/s6-memory-census.{md,json}` / `s7-noop-census.{md,json}` / `dtype-feasibility.{md,json}`

## 1. リサーチ結論(要旨)

- 現提出セットは**既に大部分が FP16 化済み**(中間メモリの 54.6%)。全面 FP16 化は入力 Cast ペナルティ(+18,000B)が支配し、利得があるのは 3 タスクのみ
- FLOAT→BOOL 化(比較/論理演算のみが消費するテンソル)は **275 テンソル / 81 タスク / 89,319 bytes** がペナルティなしで削減可能
- no-op ノードは 157 個 / 19 タスク(+0.11 点)
- ORT 1.24 CPU は本セットの全 op で FP16 カーネルあり。BOOL の欠落は Where(データ入力)/ OneHot / Resize のみ。Cast は全方向 OK
- ローカル検証は ORT スレッド並列で実行ごとに揺れる(task230 が反転した実績)

## 2. 実装項目

### G1: 選択的 FP16 化(+0.425 点)
対象は **task170 / task097 / task064 の 3 タスクのみ**(census で delta > 0 のもの)。
- 変換器は feasibility 実証済みの方式: FLOAT 中間+initializer を FLOAT16 化、`input` 直後に Cast、宣言済み出力 dtype を尊重、fp16→fp16 の no-op Cast は除去(残すと ORT の InsertCastTransformer が落ちる)
- Equal を含むモデルは opset≥11 が必要(census の opset を確認し、必要なら opset_import を 11 に上げる — ai.onnx ドメインのみ、スコアラは domain しか見ないので安全)

### G2: FLOAT→BOOL 化(81 タスク、推定 +2〜3 点)
census の 275 テンソル(比較/論理 op のみが消費、または Where の condition 入力)を対象に:
- **producer が比較 op(Greater/Less/Equal/...)で既に bool を出せる場合のみ**変換し、途中の Cast(bool→float) と再 Cast を除去
- producer が bool を出せない場合は Cast 追加が必要になり net 損になり得るため、**テンソル単位で net バイト利得 > 0 のものだけ**適用
- Where のデータ入力(X/Y)は bool 不可(ORT カーネル欠落)— condition のみ bool 化可

### G3: no-op 除去(+0.11 点)
census の 157 ノード(Mul×1: 67 / Add±0: 65 / 同形 Reshape 系: 17 / Identity: 8)を rewire で除去。
- 注意: Add-zero 除去は -0.0 の符号反転で下流が変わり得る → 検証ゲートで弾く(下記)

### H1: ハーネス決定化
`scripts/lib/scoring.py` の全 InferenceSession に `intra_op_num_threads=1` + `inter_op_num_threads=1` を設定し、ローカル検証の実行間揺れを排除する。スコア計算には影響しない(memory は proto/プロファイラ由来)。

## 3. 検証ゲート(全項目共通・タスク単位)

1. G3(数値不変のはず)→ `outputs_bit_identical`(生出力のビット一致)で合格必須
2. G1/G2(数値が変わる)→ 以下を**全例(train+test+arc-gen)**で満たすこと:
   - 閾値後マスク `(raw > 0.0)` が元モデルと完全一致
   - **マージン条件**: 変換後モデルの生出力の最小 |raw| ≥ 0.25(プラットフォーム差で符号が反転しない余裕)。
     満たさないタスクは変換を破棄(revert)
3. 適用後コスト ≥ 適用前コストになったタスクも破棄
4. 既存の divergent タスク(ローカル検証が落ちるもの)は G1/G2 の対象外(identity 検証しかできないため)

## 4. 成果物・受け入れ基準

- 実装: `scripts/lib/dtype_passes.py`(G1/G2)、`scripts/lib/optimizations.py` に G3 追加、`scripts/optimize_submission.py` に統合(CLI フラグ `--passes` で 001 セットと 002 セットを選択可)、`scoring.py` に H1
- パイプライン全体を `--tasks all --zip` で完走し、run レポートの合計が run-008(6347.8189)を上回ること
- 悪化タスクゼロ(検証ゲートで保証)
- 対象外のタスクのバイト列が run-008 の成果物と一致すること(無関係な変更をしない)

## 5. Lane T への引き継ぎメモ(proposal 003 予定)

外部ツールの per-task スコアは我々の最弱タスク群より +1.0〜+3.1 高い(例: task255 我々 12.20 vs ツール 15.29 = コスト約 1/22)。Lane G 残余(本提案 ~+3)とは桁が違うため、本提案完了後は Lane T(タスク個別の構造再設計)に主軸を移す。優先キュー(run-008 の下位): 255, 101, 133, 158, 096, 286, 367, 233, 285, 018, 209, 118, 077, 251, 110。
