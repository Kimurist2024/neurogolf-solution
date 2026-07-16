# high148 exact deep audit — task014 / task239 / task255

## 結論

**strict-lower かつ SOUND/no-error/strict-shape/UB0 を満たす候補は0件です。**
`submission.zip`、score/CSV、`docs/`、`others/71407` は変更していません。

- authority: `submission.zip == submission_base_8009.46.zip`
- authority SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- 新規 projected gain: `0.0`
- SparseTensor: 使用なし（root横断監査の担当外）

| task | model SHA-256 | official memory/params/cost | known 4構成 | runtime shape | strict-lower winner |
|---:|---|---:|---:|---:|---:|
| 014 | `15a7de7d7ad0...` | 309 / 51 / **360** | 全266/266 | mismatch 17 | 0 |
| 239 | `e15519d37cca...` | 328 / 56 / **384** | 全267/267 | truthful | 0 |
| 255 | `5bcf1caa5a31...` | 1108 / 199 / **1307** | 全265/265 | mismatch 32 | 0 |

既知検証は ORT optimization disable/default × threads 1/4 の4構成。全authorityが
完全一致した。候補は mandatory gate の official strict-lower または strict-shape より前で
全滅したため、fresh dual-ORT は fail-closed 方針により実行対象なし。

## task014

真ルールは「非0色のうち最少頻度色を選び、その bounding box をcrop」。現行360は
頻度選択、モーメントによるbbox推定、scalar-label scatterで構成され、official goldは通るが
17個のruntime/declaration shape不一致を持つ。

今回の追加探索:

- 全履歴inventoryからauthority以外5 SHAを再screen。4件はcost 1122以上、唯一のcost 201案は
  official不正解かつshape不一致で脱落。
- optimizerの見かけcost 193案をofficial再計測するとcost 360で、shape mismatch 17のまま。
- 非負である `fg_h`、`counts`、`mask_h` に対する3箇所の
  `ReduceL1(x) == ReduceSum(x)` を、非空の全7部分集合で実証探索。
  official costは360、378、2158、2176のみで、360未満は0件。全案shape mismatch 17。
- initializer 11個/51要素を用途単位で監査。alias、unused、CastLike専用initializerはいずれも0。
  attribute化できる軸はSlice/ReduceMean等にも共有され、initializer削除にはならない。
- 既存の直接crop、bool ArgMax、動的TopK、算術label、Cast置換などの失敗履歴も照合。
  truthfulな可変crop再構築は現行360未満に到達していない。

## task239

真ルールは「出現色を頻度降順に並べ、頻度を高さとする色付き棒グラフを作る」。現行384は
TopK頻度と2-channel bar/feature fieldをQLinearConvで10色へdecodeし、runtime shapeもtruthful。

- authority以外3 SHAを再screen。1件はTopK runtime非対応、cost 379/374の2件は
  shape truthfulだが既知4構成で不一致。
- 既存の個別プローブ結果でも、inactive-feature mask除去は既知24/267、fresh75/1000、
  inactive bar sentinel除去は既知2/267、fresh2/1000。いずれも明確に黒。
- normalizerはofficial相当384、optimizer passは変化なし。
- initializer 7個/56要素にalias/unused/CastLike専用品なし。
  40要素の `emb`/`qW` を因数化するとdecode用の追加activationが必要で、現行の
  2×12×5 fieldを消せない。inactive列と長方形内背景を区別する2つのWhereも削除不能。

現行表現では既知のstrict-lower 2族が反例で完全に切られ、archiveにも生存案なし。

## task255

真ルールは黒いartery/vein内部で入力から隠された緑セルの復元。ただしgeneratorには
`low_right_tall=3` と `4` が同一入力を生成し、期待出力が15セル異なる有限反例がある。
よって合法generator全域に対する deterministic input-only ONNX の100% exact解は存在しない。

- authority以外4 SHAを再screen。見かけcost 878/1162はofficialで1336/1342となり、
  baseline 1307未満ではなかった。
- optimizerの見かけcost 1133案もofficial再計測で1307、shape mismatch 32のまま。
- initializer 19個/199要素にalias/unused/CastLike専用品なし。
- 既存の独立fresh 2 seed × 5000ではauthorityは95.16%と94.44%。これは上の非関数性と整合し、
  SOUND exact候補の証拠にはならない。

## 探索範囲と証拠

- archive rescreen: authority以外12 unique SHA（5 + 3 + 4）、stage survivor 0
- normalization / constant-fold scan: preliminary lower 0
- ORT optimizer sweep: 見かけlower 2件をofficial再監査し0件
- exact nonnegative algebra: task014の7変種、strict lower 0
- initializer alias/unused/CastLike-only、attribute embedding、factor/CSEを全3タスクで監査
- full checker、strict data-prop、standard domains、finite initializer、Conv-family UB0を適用

Machine-readableな全測定値は `authority_audit.json`、`archive_rescreen.json`、
`optimizer_candidate_audit.json`、`reducel1_sum_scan.json` に保存した。
最終採用リストは `winner_manifest.json` の空配列がauthorityである。
