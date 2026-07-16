# 8008.14 exact-white all-input golf scan

## 結論

`submission_base_8008.14.zip` のうち、8006.61 から新たに置換された exact SHA 37件を全走査したが、固定採用できる exact・strict-lower 候補は0件だった。rootのzip・`all_scores.csv`・`others/`は変更していない。

Authority:

- LB: `8008.14`
- ZIP SHA256: `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`
- predecessor SHA256: `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`
- exact差分: 37 tasks

## 実施内容

対象37件すべてへ、以下のall-input semantics-preserving変換を機械走査した。

- dead node / unused initializer / dead `value_info`
- initializer dedupe
- Identity、neutral Add/Mul、同型Cast/Reshape等のno-op除去
- 同一pure nodeのCSE
- schema上optionalな未使用末尾output除去
- initializerのみで閉じる部分グラフのconstant folding
- 連続Add/Mulの定数吸収
- advisory `value_info`除去を組み合わせた正規化

生成69候補の内訳は、`REJECT_STRUCTURE_SCHEMA_UB=40`、`REJECT_NOT_STRICTLY_LOWER=29`、採用0。`actual_screen=None`による見落としを避けるため、structural通過候補と利得余地の大きいIdentity/neutral-Add候補の計13件を競技profileへ直接投入したが、全件不採用だった。

## 深掘り結果

- `task089`: 未使用`ReduceMax`除去で見かけ上 `1340 -> 1171`。ただしORTのbuffer再利用でshape mismatchが発生し、`correct=false`。
- `task165`: 重複`CastLike`のCSEで見かけ上 `587 -> 546`。ただしORT allocator mismatchで`correct=false`。runtime traceにも88件のdeclared/actual mismatchがあり、truthful-shape条件を満たさない。
- `task182`, `task191`: Identity除去はCenterCropPadのshape inferenceでload不能。
- `task209`: neutral Add除去はConcat rank mismatchでload不能。
- task107はnegative Conv pads、task182はdynamic/external Conv bias、task209/245/268/284/308/363/370は非static shape等の既存hard gateを正規化後も解消できなかった。

dead nodeやCSEは数式上exactでも、現モデル群では誤った`value_info`を利用したORTのmemory planner/allocator挙動に依存している。shapeをtruthfulに直すとmemory costが増えるため、今回のstrict-lower条件とは両立しなかった。

initializer dtypeだけを狭める案も確認したが、競技parameter costはbyte数ではなく要素数で数えるため、dtype-only shrinkに得点効果はない。initializer重複、schema-optional output、吸収可能なAdd/Mul定数も0件だった。

## 成果物

- `audit/mechanical_scan.json`: 37件・69候補の全判定、変換証明、構造監査
- `audit/deep_verification.json`: 13候補の直接競技profileとruntime shape trace
- `result.json`: 集計
- `winner_manifest.json`: 空（固定winnerなし）
- `probe_manifest.json`: 空（LB probe正当化候補なし）
- `scan_exact.py`, `deep_profile.py`: 再現スクリプト

`candidates/`は監査用の中間ONNXであり、どれも採用対象ではない。判定のauthorityは上記2つのaudit JSONである。
