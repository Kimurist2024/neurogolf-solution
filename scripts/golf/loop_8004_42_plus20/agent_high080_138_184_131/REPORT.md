# high131 — authority 8009.46 task080 / task138 / task184

## Outcome

採用候補は **0件**、projected gain は **+0.000000** です。探索は
`submission.zip`（SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`、
`submission_base_8009.46.zip` と byte-identical）から抽出した3メンバーだけを
起点に実施しました。archive候補は再利用していません。root提出物、CSV、score
metadataは変更していません。

| task | authority cost | exact scan | runtime shape | known 4-config audit | decision |
|---:|---:|---|---|---|---|
| 080 | 3050 | constant Cast foldのみ、cost 3050のまま | truthful、0 mismatch | 各設定 231/266、runtime error 0。残り35件は固定30x30 carrierへ変換不能 | no strict decrease |
| 138 | 2705 | Shape foldはstructural reject。value_info全削除はprofile `-1` | 36 mismatches | 266/266を4設定すべて通過、runtime error 0 | no truthful strict-lower graph |
| 184 | 421 | value_info全削除はprofile `-1` | 6 mismatches | disable-allは169/266、default ORTはsession生成失敗 | current-only shaveは不適格。true-rule control 1996 > 421 |

## Exact transformations

`scan_exact.py`はcurrent memberごとに dead-node/unused initializer/value-info、
initializer dedupe、optional-output pruning、Identity/no-op bypass、CSE、constant
fold、組合せ、およびmetadata normalizationを再生成して審査しました。

- task080: `Cast(wcol) -> kvec_u8` の定数化だけが成立しました。memory
  `2837 -> 2827` と引き換えに params `213 -> 223` となり、公式costは
  **3050 -> 3050**。全入力での定数畳込み自体はexactですが、strict decreaseが
  ないため却下です。
- task138: `Shape(qcol)` の定数化は実寸法を露出させ、既存の偽shape metadataと
  整合せずstructural gateで却下。153 value_info削除版は公式プロファイルが
  `memory=params=cost=-1`、runtime traceも152 mismatchで、候補ではありません。
- task184: 25 value_info削除版は同様に公式プロファイル `-1`、runtime trace
  23 mismatchで却下しました。

どのtaskにも dead node、unused initializer、initializer alias、duplicate producer、
removable optional output、CSE/no-opはありませんでした。dtype幅変更は公式param
costが要素数課金なので削減を生まず、shape metadata継承だけに依存する変形は
数値同値証明が成立しないため採用していません。

## SOUND disposition

- task080のcurrentはfull checker、strict data propagation、standard domain、UB0、
  runtime-shape truthfulを通過しています。独立既存監査
  `agent_rebase_new21/fresh_audit.json` では2 seedそれぞれ5000/5000をdefault/
  disable-all両ORTで通過し、runtime error 0です。しかし今回のcurrent-only exact
  scanには安いグラフがありません。
- task138のtrue ruleは4本の色付きborderを検出し、囲まれた矩形をcropして内部色を
  対応border方向へ伝播するものです。既存のtruthful known-perfect再構築はcost
  2762/2822で、現行2705より高価です。現行は既知出力を全通過しますがruntime
  shape 36件が偽なので、そのmetadataを温存した局所shaveは採用していません。
- task184のtrue ruleはzero separatorで2x2--3x3 block gridへ分割し、各blockの
  unique nonzero patch colorを出力するものです。独立decoderはknown 266/266と
  2 seed各5000/5000ですが、truthful ONNX controlはcost 1996で現行421を下回りません。

strict-lower候補が0件だったため、候補に対する2-seed fresh実行はpre-gateで省略
しました。`candidates/` は空です。profile `-1`だった2モデルは誤採用防止のため
`rejected_probes/` に隔離しています。

## Evidence

- `authority_inventory.json`: authority SHA、公式cost、full/strict、runtime shape、UB監査。
- `exact_scan.json`: currentから再生成した全exact transformの結果。
- `current_known_four_configs.json`: disable/default ORT × threads 1/4 の既知監査。
- `task080_graph.txt`, `task138_graph.txt`, `task184_graph.txt`: current graph inventory。
- `scan_authority.py`, `scan_exact.py`, `audit_current_known.py`: 再現スクリプト。

Final decision: **do not merge any model from this lane**.
