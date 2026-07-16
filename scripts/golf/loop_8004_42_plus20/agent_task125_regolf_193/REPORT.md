# task125 regolf 193 — current 8009.46 authority

## 結論

**採用可能な strict-lower 候補は0件**。winner setは空、projected gainは
`+0.0`。root `submission.zip` / `all_scores.csv` と `others/71407` は変更して
いない。`try_candidate.py` も使用していない。

immutable authority:

- `submission_base_8009.46.zip` SHA-256
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- task125 member SHA-256
  `c30ac7a079a4d5a91053c7748015a8c3a86ad594e542050e8826d46f1f84c529`
- official profile: memory `916` + params `129` = cost **1045**

authority自体はfull checker/strict data propagation/UB0を通るが、実測shape
witnessは32箇所不一致である。例として宣言 `[1,1,1,1]` の `rcn00` は実測
`[1,27,27,27]`、宣言 `[1,1,1,1]` の `qWdyn` は実測
`[10,2,3,3]`、宣言 `[1,1,20,20]` の最終出力は実測
`[1,10,30,30]`。この既存LB-white例外は新payloadへ継承していない。

## exact局所変換の実測

9変換を独立payloadとして生成し、competition profiler、full/strict/data_prop、
standard domain、finite、Conv bias UB0、runtime-shape witnessの順でfail-closed
評価した。

| 変換 | official profile | full/strict | truthful | 判定 |
|---|---:|:---:|:---:|---|
| `wshape10` 個別fold | unscorable | fail | — | static shape conflict |
| `s13` 個別fold | unscorable | fail | — | shape長1対axes長2を露出 |
| `s26` 個別fold | unscorable | fail | — | shape長1対axes長2を露出 |
| `s27` 個別fold | unscorable | fail | — | shape長1対axes長3を露出 |
| `s25` 個別fold | unscorable | fail | — | shape長1対axes長2を露出 |
| 5 shape式の全fold | unscorable | fail | — | 上記conflictを全て露出 |
| `vW = Transpose(hW)` 共有 | `923+122=1045` | pass | fail (32) | **同点**、非truthful |
| 終端`qWdyn`直接initializer化 | unscorable | fail | — | 10出力channelを露出し宣言と衝突 |
| binary `Add -> Sum` | unscorable | fail | — | opset18 `Sum`のint64 carrier不受理 |

方向カーネル共有はall-input exactだが、7 parameterの削減がTranspose出力7
memoryへそのまま移り、`916+129 -> 923+122` の同点だった。唯一のstatic-pass
変換でもauthority由来の32 shape mismatchを残すため採用対象ではない。

## 終端QLinearConvの整数factor / sparse配置

実測 `qWdyn` は `[10,2,3,3]`、非零出力rowは `{3,4,6,8}` の4本だけで、
rankは4。各rowの整数gcdは全て1なので共通整数factorはない。

- 直接格納: 180 params
- 非零4本だけ: 72 params
- 現authority: 6本108 paramsを3つの宣言1要素cropで配置し、合計見かけ
  `108+3=111`
- truthfulに72 paramsから非連続位置 `{3,4,6,8}` へ戻す場合、少なくとも
  180要素の完成kernelを一度materializeするため、楽観下限でも
  `72+180=252`

従ってsparse/factor化は108 paramsを節約しても141以上costを増やす。rank-4
二段factorはさらに中間activationが必要になる。`qS` は全3 QLinearConvで、
`z0` と `m1` も必要箇所ですでに共有済み。unused initializer、同shape/value
alias、dead node、duplicate pure node、unused optional outputはいずれも0件だった。

## truthful局所topology下限

currentの `p -> horizontal/vertical -> Min -> 2ch feature -> terminal` を保つ
局所変換について、以下をすべて無料とする楽観的下限を置いた。

- shape tensorとinput extractionの全cost
- `bd2` materialization
- `hW -> vW` Transpose出力
- ゼロ出力6本と非連続channel位置への復元
- 最終output activation

それでもtruthful中間は `p/h/v/bd = 169*4` と2ch feature `338`、合計
**1014 memory**。さらに方向kernelを1本だけ7、非零終端4本を72、量子化scalar
を3とした過小評価でも **82 params**、局所topology下限は **1096** となる。
これはauthority 1045より51高い。したがってCenterCropPad shape-chain、
directional QLinearConv、終端kernelの局所regolfでは、truthful strict-lowerは
成立しない。1045未満には別の意味アーキテクチャが必要である。

## 14-wide true-rule control

過去の `task125_pool14.onnx` を現環境で再profileした結果は
`memory=1051, params=116, cost=1167`。full/strictはpassするがruntime witnessは
40 mismatchでtruthfulではなく、そもそも1045より122高い。

generator真ルールと7-tap directional maskについては、過去の2000 fresh / 
20,000 mask auditに加え、今回未使用の独立seed range
`31,000,000..31,001,999` と `47,000,000..47,001,999` を再検証し、双方
**2000/2000 pass**。これはrule理解の確認であり、候補ONNXのadmission結果には
数えていない。

## gate disposition

strict-lower候補が0件だったため、候補に対するknown全件、4ORT、fresh二seedの
入口へ到達したpayloadはない。非strict-lowerやstatic/truthful失敗をfreshで
救済することはせず、private-zero近似も作成していない。

機械可読証拠:

- `audit.json`: authority/profile/shape witness、9候補全判定、initializer/terminal
  分析、truthful局所下限、独立generator audit、integrity hash
- `run_audit.py`: 再現driver
- `candidates/`: すべてreject済み監査payload。採用候補ではない

最終integrity:

- `submission.zip`: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv`: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- `others/71407` tree:
  `c0a03e29747daaab6fe639d732d80b570476cdc3414d1d6b92e1b6c8d736f226`

いずれも作業前後で一致した。
