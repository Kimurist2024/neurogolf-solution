# task245 Selu candidate — independent review 197

## Verdict

**独立監査PASS。候補を採用可能と判定する。**

- authority: `submission_base_8009.46.zip::task245.onnx`
  - SHA-256 `228b6ad9f24579bc6f5840da4e5a18f08343b76a26538f500e7e77d328d6e9d5`
  - official profile `memory=306, params=79, cost=385`, correct
- candidate: `root_task245_regolf_196/task245_selu_cost384.onnx`
  - SHA-256 `1b777a51c55fa98ed720fb993a9305bcca2218627592d23e46d8d5a6bce91ba9`
  - official profile `memory=306, params=78, cost=384`, correct
- strict decrease: **1**、projected gain `ln(385/384)=+0.0026007817000574403`

root側のREPORT/auditは検証入力に使わず、authority ZIPからpayloadを再抽出し、
独立driverで全項目を再測定した。

## Graph delta

payload全差分を列挙し、次だけであることを確認した。

1. 4 Einsumの出力式を `->` から `->n` へ変更。
2. 対応するcode/log value_infoをscalar `[]` から `[1]` へ変更。
3. 4個の `Div(log, two_f16)` を
   `Selu(log, alpha=1, gamma=.5)` へ変更。
4. 唯一未使用となるinitializer `two_f16=[2]` を削除。

他のinitializerはdtype/shape/valueがbitwise同一。追加initializer、他node変更、
dead side effectはない。whitelist外差分は0件。

## 4 Log sourceのgenerator-domain証明

generatorは `task_a1570a43.py`。`size=7`、5×5 red spriteを
`conway_sprite(5,5,14)` で生成する。

- spriteは25点から最大14回だけ削除するため最低11点を残す。
- あるrow/colの最後の1点になる削除は拒否されるため、local row=4とcol=4にも
  必ず最低1点残る。
- vertical shift時のsource row、horizontal shift時のsource colは、その
  local=4点で最低4。非shift軸では最低5。
- `red_sel_f16`のactive係数と`pos_min_f16`は正であり、上記点だけで1を超え、
  残る10点以上も正項を加える。よって `rr_code, rc_code > 1`。
- greenは `(brow,bcol)` から6離れた四隅を持つ。bottom/right座標は最低6。
  `q4`, `pos_min_f16`, `theta_base_f16`は全て非負で、thetaには正係数がある。
  よって `gr_code, gc_code > 1`。

したがって4 `Log`入力は厳密に1超、4 `Log`出力は厳密に正である。

known 267件＋独立fresh 6000件を全4 ORT構成でinstrumentした実測下限:

| tensor | source code min | Log min |
|---|---:|---:|
| rr | 63.3125 | 4.1484375 |
| rc | 63.28125 | 4.1484375 |
| gr | 441.5 | 6.08984375 |
| gc | 441.5 | 6.08984375 |

`code<=1` と `log<=0` は全て0件。高座標の指数和をgraph outputとして露出した
際にsource codeがfloat16 `+inf`になる例はあるが、負・zero・NaNではない。
候補が実際に受け取るLog outputは全監査例で正かつfiniteだった。演算単体監査は
念のため `+inf` も含めた。

## Einsum `->n` rank carrier

input batchは静的に `n=1`。

- authorityは `n,c,r,s` を全reduceしたscalarを作り、shape `[1]` の
  `two_f16`とのDivでrank-1へbroadcastする。
- candidateは`n`だけ残し、同じ `c,r,s` contractionを直接shape `[1]` で作る。

`n=1`なので実数上の項集合と順序対象は同じで、downstream carrierも双方 `[1]`。
さらに4 code＋4 Logを全case/configで露出し、authority scalarをflattenした1要素と
candidate `[1]` の**float16 bitsが全件一致**した。

## Div2 versus Selu(.5)

独立の単一opモデルで、非負float16 bit pattern
`0x0000..0x7c00`（+0、全正subnormal/normal、+infを含む）**31,745値**を
一括評価した。

| ORT config | Div/Selu bit differences |
|---|---:|
| DISABLE_ALL, threads=1 | 0 |
| DISABLE_ALL, threads=4 | 0 |
| default, threads=1 | 0 |
| default, threads=4 | 0 |

Selu出力も4構成間でbitwise同一。実モデルのSelu入力は前節どおり厳密に正なので、
常にpositive branch `gamma*x = .5*x` を取る。

## Whole-model verification

known corpusはtrain/test/arc-gen合計267件。freshはroot側と重ならないseedを使用:

- seed `245197101`: 3000/3000
- seed `245197102`: 3000/3000

各集合を以下4構成すべてでauthority/candidate同時比較した。

- ORT_DISABLE_ALL × threads 1/4
- default ORT × threads 1/4

全構成で:

- candidate truth: 100%
- authority truth: 100%
- final raw bitwise equality: 100%
- threshold equality: 100%
- 4 code＋4 Log bitwise equality: 100%
- runtime error: 0
- candidate final nonfinite: 0

## Structure and inherited failure

authority/candidateともに:

- full checker PASS
- strict shape inference (`data_prop=False`) PASS
- standard domainのみ、functions/nested graph/sparse initializer/banned opなし
- Conv-family bias UB 0

`data_prop=True` は双方で同じ既存AffineGrid error:

```text
Inferred shape and existing shape differ in dimension 0: (2) vs (1)
```

error文字列まで一致し、candidate固有のstructural errorは0。既存LB-white cloakの
継承であり、新規shape failureではない。

## Cost measurement caveat

最終costはknown実入力を通すcompetition `score_and_verify`相当pipelineで測定した。
`rank_dir.cost_of`のzero-input簡易profileをauthorityへ単独使用すると、Log経路が
途中終了してmemoryを過小計上することがあるため、このtaskのauthority cost根拠には
使えない。完全profileでは再現可能に `385 -> 384` となる。

## Integrity and artifacts

このreviewはroot submission、score ledger、`others/71407`を変更していない。
作業前後hashは一致:

- `submission.zip`: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv`: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- `others/71407` tree:
  `51c7d120dc3e6d65650288a7e9fbb02db92405ff17546f1df25d004e27210ae0`

成果物:

- `audit.json`: 全case/configの詳細とgraph delta、operator exhaustive、profile、integrity
- `audit_review.py`: root auditから独立した再現driver

`try_candidate.py`は使用していない。
