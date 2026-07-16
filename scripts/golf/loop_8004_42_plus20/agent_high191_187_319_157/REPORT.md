# task187 / task191 / task319 strict-lower SOUND再監査

## 結論

8009.46 authorityの3タスクを追加対象として、current-only exact変換397通りと、リポジトリ全履歴322 SHAを監査した。SOUNDかつstrict-lowerなwinnerは0件、改善量は`+0.00`。rootのsubmission・score・`all_scores.csv`・`others/`・docsはこのlaneから変更していない。

Authorityは`submission_base_8009.46.zip`、SHA256は`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`で固定した。

| task | authority cost | current exact subsets | current best | history unique SHA | history strict-lower | admit |
|---:|---:|---:|---:|---:|---:|---:|
| 187 | 1798 | 127 | 1798 | 108 | 1 | 0 |
| 191 | 3436 | 15 | 3436 | 81 | 3 | 0 |
| 319 | 1003 | 255 | 1003 | 133 | 0 | 0 |

## Current authority exact探索

全`CastLike`を型参照値ではなく静的element typeだけを使う`Cast`へ置換し、`Identity` bypassおよび置換後にoutputへ到達しないnode/initializerのDCEを全subsetで走査した。

- `CastLike`点: task191=3、task187=6、task319=8
- `Identity`点: task191=1、task187=1、task319=0
- `PRelu`: 3タスクとも0（従って`PRelu -> LeakyRelu`適用点なし）
- subset総数: `15 + 127 + 255 = 397`
- 全候補を静的costではなく競技runtime profileで測定
- strict-lower: 0件

task191の3つ目の`CastLike`はinitializerではなく計算tensorを型参照しており、既存単独scanでは対象外だった。この点も静的推論したdtypeへの`Cast`置換として含めたが、costは下がらなかった。

3 authorityはいずれもfull checkerとstrict `data_prop`を通る一方、元モデル自体にruntime shape不一致がある（task191=35、task187=13、task319=26）。誤ったshapeに依存する変換は、数式上exactでもSOUND gateでは採っていない。

## 全履歴SHA監査

`.git/.venv/node_modules`と本lane生成物を除くリポジトリを走査した。

- ZIP: 1,284ファイル、対象member 3,627件
- loose ONNX: 2,091件
- 重複排除後: 322 SHA（task187=108、task191=81、task319=133）
- 読み込みエラー: 0
- full checker / strict data propagation / standard domain / bannedなし / nestedなし / functionsなし / sparseなし / Conv UB0を先に確認
- 構造通過SHAを競技runtime profileで再測定

strict-lowerになった履歴4件はいずれもauthorityと同一canonical computationではなかった。深掘り結果は次の通り。

| task | cost | known DISABLE threads1/4 | authority raw equality | runtime shape mismatch | default ORT | 判定 |
|---:|---:|---:|---:|---:|---|---|
| 187 | 1737 | 266/266 | 0/266 | 14 | TopK shape load error | reject |
| 191 | 3426 | 267/267 | 0/267 | 0 | CenterCropPad shape load error | reject |
| 191 | 3430 | 267/267 | 0/267 | 27 | CenterCropPad shape load error | reject |
| 191 | 3435 | 267/267 | 0/267 | 34 | CenterCropPad shape load error | reject |

4件とも二値正答は既知全件で一致するが、raw tensorはauthorityと全例で異なる。task187にはprivate-zero履歴があり、task191も過去の局所ルール候補である。非canonicalなbehavior changeについて全入力真ルールの形式証明は閉じていないため、既知正答率だけでは採用しない。task191 cost3426はruntime shapeだけはtruthfulだったが、raw差分・default load error・全入力証明欠如により同様に不採用。

fresh検証はfinalist用gateだが、4件はその前段のuniversal-equivalence / dual-ORT gateで全て落ちたため実施していない。

## 成果物

- `subset_scan.json`: current exact全397 subsetの実競技profile
- `history_inventory.json`: 全履歴322 SHAとsource一覧
- `history_screen.json`: 全履歴の構造・実競技profile結果
- `deep_audit.json`: strict-lower 4件のruntime shape、known、threads1/4、raw比較
- `result.json`: 集計
- `winner_manifest.json`: 空winner manifest

`history_candidates/`はreject根拠の再現用であり、採用候補ではない。
