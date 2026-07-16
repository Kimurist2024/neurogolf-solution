# high133 — authority 8009.46 task069 / task187 / task277

## Outcome

採用候補は **0件**、projected gain は **+0.000000** です。authorityは
`submission.zip` SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
で、`submission_base_8009.46.zip` とbyte-identicalです。3モデルはこのZIPから
抽出し、archive候補を再利用せずcurrent-onlyで変形しました。root提出物、
`all_scores.csv`、`others/` は変更していません。

競技 `score_and_verify` によるauthority costは以下です。declared shapeだけの
静的値ではなく、この値を比較基準にしました。

| task | authority SHA-256 | memory | params | official cost | runtime shape | known 4 configs | decision |
|---:|---|---:|---:|---:|---|---|---|
| 069 | `8a7bdff92bede718fe90be1b6c1fbf9481556b02554f1cfbf0533f217fefd5e1` | 479 | 45 | 524 | 79 mismatch | disable-all 264/264、default session failure | reject |
| 187 | `bb40138844229a6ede66203b2a99e3a474e43ac385e08f7fb0c079bed0231126` | 1657 | 141 | 1798 | 13 mismatch | disable-all 266/266、default TopK failure | reject |
| 277 | `4e5e5ff2f5f49e9d4d0c7cda067144e702eb4024fe1cc6a0ed669a2f6372e1af` | 578 | 53 | 631 | 2 mismatch | 266/266 in all four | reject: fresh/private unsound |

## Current-only exact scan

`scan_exact.py`はdead/unreachable payload、unused initializer/value_info、initializer
dedupe、optional output、Identity/no-op、CSE、constant fold、metadata normalization、
および組合せをcurrentから再生成しました。strict-lower候補は0件です。

- **task069**: 92 value_infoの全削除だけが機械的差分となりましたが、競技
  sessionを作れずcost `-1`、runtime traceも87 mismatchでした。現行の
  declared-shape-only静的値は242ですが、競技値は524です。この差を削減扱いして
  いません。
- **task187**: `k4h = Identity(seed_shape)` の除去は値としてはexactですが、
  `TopK`へliteral `k=4`を露出させると、推論軸が4未満と判定されsession生成が
  失敗します。これは計算削減ではなくshape-inference barrierの破壊なので却下です。
  Shape/Identity/Concatの定数foldもfull checkerを通りません。
- **task277**: dead node、duplicate producer、initializer alias、no-op、CSE、
  fold可能定数は0件です。8 value_info削除版はnonstaticとなりstructural rejectです。

## Fixed integer / initializer / dtype audit

優先指定された固定整数式は別probeで確認しました。

- task069: authority initializer `codes_i8` はshape `[1,10,1,1]`なので
  `Size(codes_i8)=10` は全入力で厳密です。int64定数置換にoverflow・丸めは
  ありません。しかし置換すると3個の`CenterCropPad`で、推論channel 10と既存宣言1
  の矛盾が顕在化しfull/strict checkerが失敗します。
- task187: authority initializer `shape_seed` はshape `[4]`なので
  `Shape(shape_seed)=[4]` は全入力で厳密です。これもint64でoverflow・丸め無しです。
  定数化すると`CenterCropPad(ish)`の推論長4と宣言長1が衝突しchecker失敗です。

3 taskすべてでbyte-identicalかつ同shapeのinitializer aliasは0、unused initializerは
0、CastLikeの型指定だけに使われるinitializerは0です。initializer parameter costは
dtypeのbyte幅ではなく要素数課金なので、単純なint64→int32やfloat32→float16は
削減になりません。QLinearConv/Resize/TopK/CenterCropPad/Gatherの定数入力はschema上
の実データ入力であり、同値attributeへ移せません。task277のscalar `q`など再利用
可能な値は既に共有済みです。

## SOUND true-rule disposition

- **task069**: 唯一の非0・非8色templateを取り出し、各8形状へ同じ相対座標で
  色をstampし、元templateを消す規則です。現行は71個のCenterCropPadと偽shapeに
  依存し、default ORTを作れません。固定Sizeを表面化しただけでもshape witnessが
  崩れるため、current-only shaveは保証不能です。
- **task187**: 灰色の水平・垂直線で区切られる矩形領域を判定し、外部背景を3、
  両方向から囲われた内部を2へ塗り分け、線を保存する大域矩形規則です。現行は
  13 shape mismatchとdefault TopK failureを持ちます。既存のshape-clean true-rule
  controlはcost 56264で1798を大幅に上回ります。既存のcheaper実行可能leadも
  fresh 4695/5000かつ14 shape mismatchで不適格です。
- **task277**: 3個のcyan spriteのうち2個のfull copyをblue、1列欠けた小さいcopyを
  redにする規則です。現行はknown 266/266を全4設定で通しますが、独立freshでは
  1921/2000と1914/2000（両ORT同値、error 0）で、private保証がありません。
  `g`,`u`は宣言 `[1,1,1,1]`、実行 `[1,10,30,30]`です。truthful component
  mass/width controlsはcost 3831/5341で631を下回りません。

strict-lowerかつfull/strict/truthful/known4を通る候補が0件だったため、候補に対する
2-seed freshはpre-gateで省略しました。`candidates/` は空です。固定整数probeは
誤採用防止のため`rejected_probes/`にのみ保存しています。

## Evidence

- `official_profiles.json`: competition `score_and_verify` cost。
- `authority_inventory.json`: exact SHA、static/full/strict、runtime shape、UB、graph inventory。
- `current_known_four_configs.json`: disable/default ORT × threads 1/4。
- `exact_scan.json`: current-only exact transform scan。
- `fixed_integer_probes.json`: Size/Shape exact値、範囲、overflow/丸め証明と拒否理由。
- `initializer_analysis.json`: initializer共有、dtype、CastLike attribute化可能性。
- `task069_graph.txt`, `task187_graph.txt`, `task277_graph.txt`: current graph。

Final decision: **do not merge any model from this lane**.
