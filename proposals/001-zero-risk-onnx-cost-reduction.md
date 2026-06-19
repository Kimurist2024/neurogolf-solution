# Proposal 001: ゼロリスク ONNX コスト削減パイプライン

- 提案者: Fable(メインセッション)
- 実装者: Opus サブエージェント
- ステータス: 承認済み・実装待ち
- スコープ: 検証ハーネス + 戦略 S1〜S5(挙動を変えない最適化のみ)

## 1. スコアラ分析(提案の根拠)

公式スコアラは `inputs/neurogolf-2026/neurogolf_utils/neurogolf_utils.py`。

```
score(task) = max(1, 25 - ln(max(1, memory + params)))
```

### params(`calculate_params`)
- `graph.initializer` の `prod(dims)` の合計(スカラー dims=[] は 1)
- `sparse_initializer` も同様
- `Constant` ノードの `value` / `sparse_value` / `value_floats` / `value_ints` / `value_strings` も加算

### memory(`calculate_memory`)— **こちらが支配項**
- `infer_shapes(strict_mode=True)` 後のグラフで計算
- 対象 = **ノード出力テンソル全部** ∪ `graph.input` ∪ `graph.value_info` ∪ `graph.output`。
  ただし名前が文字列 `'input'` / `'output'` のものは除外
- 各テンソル: `要素数 × dtype の itemsize`(さらに ORT プロファイラ実測との max)
- **initializer は memory に入らない**(params 側で数える)
- float32 の `[1,10,30,30]` 中間テンソル 1 個 = **36,000 bytes**。中間テンソルの数と dtype がコストの本体
- 失格条件(None が返る): 複数 input/output、dim_param や非正の次元、`value_info` の重複名、
  ノード出力に対応する `value_info` 欠落、sequence 型、functions、カスタム domain、サブグラフ属性、
  initializer 名と graph input/output 名の衝突
- **重要な抜け穴**: `graph.value_info` に「どのノードも生成しない stale なエントリ」が残っていると、
  その分の bytes も memory に加算される。削除すればノーリスクで memory が減る

### 検証(`verify_subset` 相当)
- `(session.run(...)[0] > 0.0).astype(float)` が one-hot 期待値と `np.array_equal` で完全一致
- train + test + arc-gen 全ペア(30×30 超のグリッドは無視)
- セッションは `ORT_DISABLE_ALL`(グラフ最適化なし)+ プロファイリング有効
- `sanitize_model` で全名前が `safe_name_N` に置換される。名前に依存した挙動は不可。
  `kernel_time` を含む名前は失格
- 禁止 op: Loop / Scan / NonZero / Unique / Script / Function / **Compress** / *Sequence*
- ファイルサイズ ≤ 1.44 × 1024 × 1024 bytes

## 2. 入力素材(3 候補源)

| 源 | パス | 内容 |
|---|---|---|
| A | `inputs/neurogolf-6347-80/overrides/` | LB 63.4780 系の改良版 400 個 |
| B | `inputs/neurogolf-6347-80/base_submission/` | 同ベース版 400 個 |
| C | `inputs/neurogolf-6347-76/` | LB 63.4776 系 400 個 |

## 3. 戦略(S1〜S5 を本提案で実装)

### S1: 未使用 initializer の削除(params↓、ゼロリスク)
どのノードの input にも現れない initializer を削除。

### S2: 同一 initializer の重複排除(params↓、ゼロリスク)
(dtype, shape, raw bytes) が一致する initializer 群を 1 つに統合し、ノード input を canonical 名に付け替え、孤児を削除。

### S3: 一様テンソルのスカラー化(params↓、低リスク)
全要素が同値の initializer を、消費ノードがすべてブロードキャスト安全な op
(`Greater, Less, Equal, Add, Sub, Mul, Div, Where, Max, Min, And, Or, Not, Clip, LessOrEqual, GreaterOrEqual, Sum`)
の場合に限りスカラー(dims=[])に置換。
**task 054 と task 158 はスキップ**(参考ノートブックで「ローカルは通るが公式グレーダーで落ちる」と報告済み)。

### S4: stale value_info の削除 + 重複 value_info の解消(memory↓、ゼロリスク)【新規】
- どのノードも出力せず、graph input/output でもない `value_info` エントリを削除
- 同名の `value_info` が複数あれば 1 つに(放置すると**スコアラが None を返し 0 点**)

### S5: タスクごとのベスト候補選択(両方↓、ゼロリスク)【新規】
各タスクについて A/B/C の 3 候補それぞれに S1〜S4 を適用し、
**正しさ検証を通過した中で cost = memory + params が最小のもの**を採用する。

## 4. パイプライン仕様

```
for task in 1..400:
    for cand in [A, B, C]:
        m = load(cand)
        m = S1(m); m = S2(m); m = S4(m)
        if task not in {54, 158}: m = S3(m)   # 失敗したら S3 だけ巻き戻し
        score = validate_and_score(m)          # 正しさ + cost。失敗なら候補ごと除外
    best = min(valid_candidates, key=cost)     # 全滅なら元ファイル(A)をそのまま採用し WARN
    save(best, artifacts/optimized/taskNNN.onnx)
report + submission.zip
```

- 各最適化パスは適用→`onnx.checker.check_model` + `infer_shapes(strict_mode=True)`→失敗なら**そのパスだけ**巻き戻し
- 検証は移植スコアラで実施(正しさ NG の候補は採用しない)
- 乱数・時刻に依存しない決定的なパイプラインにする

## 5. 実装契約(Opus へ)

| ファイル | 内容 |
|---|---|
| `scripts/lib/scoring.py` | 公式 `neurogolf_utils.py` から scoring/検証部を**挙動を変えずに**移植。IPython/matplotlib/onnx_tool 依存は持ち込まない。データパスは `inputs/neurogolf-2026/` 起点に変更 |
| `scripts/lib/optimizations.py` | S1〜S4 の純関数群(モデルを deepcopy して返す。入力は変異させない) |
| `scripts/optimize_submission.py` | CLI。`--tasks 1-5` / `--tasks all`、`--sources A,B,C`、出力先 `artifacts/optimized/`、レポート `artifacts/reports/run-<連番>.md`、`--zip` で `artifacts/submission.zip` 生成(フラットに task*.onnx を格納) |

### 環境
- `uv venv .venv --python 3.12` → `uv pip install numpy onnx onnxruntime`(ノートブック準拠は onnx 1.21 / onnxruntime 1.24 / numpy 2.4 系。プラットフォーム都合で近いバージョンになる場合は実際の版をレポートに記録)
- 実行は `.venv/bin/python`

### レポート要件
- タスクごと: 採用候補 (A/B/C)、params/memory/cost/score の前後、適用パスと巻き戻し
- 合計: 期待スコア合計の前後比較(= Σ max(1, 25 - ln(cost)))
- 正しさ検証に失敗した候補・タスクの一覧

### 受け入れ基準
1. `--tasks 1-5` が完走し、5 タスク全てで正しさ検証パス
2. 最適化後の cost が最適化前(候補 A 単体)以下であること(悪化ゼロ)
3. メモリ計算が公式実装と同値(任意の 3 タスクで `neurogolf_utils` 由来のロジックと突合)
4. 提案書にない最適化を入れていない

## 6. 次回以降の候補(本提案では実装しない)
- S6: 中間テンソルの dtype 縮小(bool/float16 化)— memory が支配項なので効果大だが要実測
- S7: Identity / 同型 Cast / no-op 演算の除去(中間テンソル数の削減)
- S8: コスト比較付き定数畳み込み

## 7. Amendment 1(2026-06-12、初回提出後の修正)

**事実**: ローカル(macOS arm64)で source A が検証失敗する 6 タスク(191/219/264/230/282/317)は、
公式グレーダー(Linux)では合格している(ノートブック出力の公式スコア 6347.81 が証拠。
浮動小数点の `> 0.0` 境界がプラットフォーム間で反転する)。

**帰結**: S5 の「ローカル検証ベースのソース置換」は、ベースライン A がローカルで落ちるタスクでは
**公式スコアを毀損する**(B 置換 3 件で推定 -8.6 点)。

**修正ポリシー**:
1. ソース間置換(S5)は「**ベースライン A がローカル検証に合格しているタスクのみ**」で行う
2. A がローカルで落ちるタスクは **A をそのまま採用**(S1/S2/S4 は数値に影響しないので適用可)
3. 正しさの判定基準を「gold との一致」から「**元モデルの出力とのビット一致**」に変更すれば、
   プラットフォーム差の影響を受けずに最適化の安全性を保証できる(S6 実装時に必須)
