# 全400 task inference/default exact探索

## 結論

`submission_base_8009.46.zip`の全400 ONNXを固定authorityとして、Dropout inference化、Clip無限optional bound省略、Reshape schema-default属性削除を走査した。strict-lower候補は0件、winnerは0件、改善量は`+0.00`。

Authority SHA256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`

rootのsubmission ZIP、`all_scores.csv`、`others/`、docsはこのlaneから変更していない。

## 全400 inventory

| family | nodes seen | eligible exact sites | variants profiled | strict-lower |
|---|---:|---:|---:|---:|
| inference-mode Dropout → Identity | 0 | 0 | 0 | 0 |
| Clip min=-inf / max=+inf optional省略 | 15 | 0 | 0 | 0 |
| Reshape `allowzero=0`属性削除 | 55 | 3 | 4 | 0 |

Dropoutノードはauthority全体に存在しなかった。Clipは15ノード存在したが、initializerおよびConstant producerを含めてscalar `-inf` min / `+inf` maxは0点だった。

Reshapeの明示`allowzero=0`は次の3点だけだった。

- task209 node55（opset schema since 14）
- task209 node90（opset schema since 14）
- task300 node6（opset schema since 21）

## Competition actual profile

| task | variant | baseline actual | candidate actual | full checker | strict data_prop | result |
|---:|---|---:|---:|---|---|---|
| 209 | node55単独 | 2087 | 2087 | pass | pass | tie |
| 209 | node90単独 | 2087 | 2087 | pass | pass | tie |
| 209 | node55+90 | 2087 | 2087 | pass | pass | tie |
| 300 | node6単独 | 175 | 175 | pass | pass | tie |

task209はadvisory/static profileでは`1721→1721`だが、競技runtime profileでは`2087→2087`だった。判定には後者を使用した。task300はstatic/runtimeとも`175→175`。

Reshapeのactive schema defaultが0であることを各モデルのopsetから照合しており、変換自体は全入力等価である。しかし得点はstrict-lowerのみ採用するため、tieは保存・採用していない。

既存`root_schema_default_scan_164`の全default属性1,171 profileも再参照した。同scanのstrict-lowerは0件で、今回の実競技profile結果と矛盾しない。

strict-lower candidateが0件だったため、known default/disable × threads1/4、raw比較、fresh検証へ進むfinalistは存在しない。

## 成果物

- `scan.py`: authority固定・全400 discovery・変換・actual profile再現スクリプト
- `scan.json`: 全400 member SHA、op/site inventory、4候補のchecker/profile結果
- `result.json`: 集計
- `winner_manifest.json`: 空winner manifest

`candidates/`は空であり、rootへのmergeは行っていない。
