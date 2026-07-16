# high134 — authority 8009.46 task035 / task284 / task396

## Outcome

採用候補は **0件**、projected gainは **+0.000000** です。authorityは
`submission.zip` SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
で、`submission_base_8009.46.zip` とbyte-identicalです。全モデルをこのZIPから
抽出し、task396を含めarchive/known-black候補は一切再利用していません。
root `submission.zip`、`all_scores.csv`、`others/` は変更していません。

| task | authority SHA-256 | official cost | full/strict | runtime shape | known4 | decision |
|---:|---|---:|---|---|---|---|
| 035 | `82b9e298e974085968d76c33de1cd7a7fafd951a86b53d75817046070279a966` | 496+48=544 | pass | truthful、0 mismatch | 266/266 ×4 | no strict-lower transform |
| 284 | `0d03efd73a591a1f2c61885cb2c987cd63ebc6bd3eb3c5906f758011d424811a` | 465+52=517 | static pass | 11 mismatch | 266/266 ×4 | shape-witness lineage、reject |
| 396 | `ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94` | 924+95=1019 | pass | duplicate node nameでraw trace不可。名前修正版は0 mismatch | 266/266 ×4 | cost-neutral only / private-risk lineage |

## Current-only exact scan

`scan_exact.py`はdead/unreachable node、unused initializer/value_info、initializer
dedupe、optional outputs、Identity/no-op、CSE、constant fold、metadata normalization、
および組合せをcurrentから再生成しました。preliminary/official strict-lowerは0件です。

- **task035**: 65 nodesすべてoutput-reachableで、dead/unused/duplicate/no-op/CSE/
  fold可能定数がありません。value_infoも0です。
- **task284**: 差分は6個のvalue_info全削除だけで、推論後shapeがnonstaticとなり
  structural rejectです。既存のShape/CenterCropPad witnessを壊すため候補化して
  いません。
- **task396**: 差分は64個のvalue_info全削除だけで、同様にnonstatic rejectです。
  currentから生成した以外の黒lineage graphは読み込み・候補化していません。

## Fixed integer / initializer / dtype audit

task284の2個の固定Shape式をcurrentから個別に定数化しました。

- canonical input `[1,10,30,30]`から
  `Shape(input,start=0,end=1)=[1]`。int64範囲は常に`[1,1]`でoverflow・丸め無し。
  しかしliteral化すると`CenterCropPad`の実channel 10と宣言1の矛盾が顕在化し、
  full/strict checkerが失敗します。
- `x70`は56 scalar項のConcatなので
  `Shape(x70,start=0,end=1)=[56]`。範囲は常に`[56,56]`でoverflow・丸め無し。
  literal化すると後続CenterCropPadのrankおよび先頭次元56と宣言0/1が衝突し、
  checkerが失敗します。

3件ともunused initializer、同dtype・同shape・同bytesのinitializer alias、
CastLikeの型指定だけに使われるinitializerは0です。parameter costはdtype byte幅
ではなく要素数課金なので、単純なdtype narrowingは削減を生みません。task035の
QLinear scale/zero-pointは既に共有され、task284のShape値やtask396のTopK/Gather/
Slice定数はschema上の入力であり同値attributeへ移せません。

task396にはnode name `a4`が2件あります。node nameだけを一意化したcurrent-only
diagnostic（SHA `f26009e6af0f829f769b38151baa1b75687dcf02d713ed01495db1515f50d742`）は、
入力・出力・属性・順序を変えないためONNX意味論上exactで、runtime shape 0 mismatch
を確認できました。ただしnode名はcost対象外なので改善は **0**、候補ではありません。

## SOUND true-rule disposition

- **task035**: 四方向それぞれについて、対応する外周色を、その方向から見て最初の
  gray(8)セルへ置き換える局所/有界規則です。currentはfull/strict/truthful/
  known4を満たし、exact LB oracleでcost544 SHAとして白固定済みです。以前の
  `[x,7x mod 256]` feature lane削除はuint8 wrapを無視して0/266となり、6 laneの
  全64 subsetに対する整数分離監査でも1 laneも除去不能でした。今回のcurrent-only
  scanにも追加余地はありません。
- **task284**: 異色のcollinear seedを内向きに伸ばし、直交5-cell capと2 serifを
  付け、任意transposeにも対応する規則です。referenceはknown266/266、fresh
  3000/3000、境界parameter 1960/1960。現currentも既存独立2 seedで500/500ずつ
  両ORT、error 0ですが、11 runtime/declaration mismatchを持つためshape-witness
  保持の局所shaveは採用禁止です。実行テンソルを真に計上すると現行517を大幅に
  上回ります。
- **task396**: 最も幅/高さの大きいhollow rectangleをcropし、borderと内部の静的
  pixelsをもう一方の非0色へrecolorする大域幾何規則です。currentはHardmax、
  ScatterElements、bit-packed row heuristicを使う既知private-risk lineageです。
  archive黒候補は再利用していません。generator-SOUND corner parserはcost1245で、
  current1019を下回りません。

strict-lower候補が0件だったため、候補に対する2-seed freshはpre-gateで省略しました。
`candidates/`は空です。task284固定Shape probeは`rejected_probes/`、task396の
cost-neutral name修正版は`diagnostics/`に隔離しています。

## Evidence

- `authority_audit.json`: authority SHA、competition cost、full/strict、known4、runtime shape。
- `exact_scan.json`: current-only mechanical transform scan。
- `initializer_analysis.json`: initializer共有・dtype・CastLike attribute化監査。
- `fixed_integer_probes.json`: task284固定Shape値、範囲、overflow/丸め証明。
- `task396_name_diagnostic.json`: duplicate-name-only exact診断。
- `task035_graph.txt`, `task284_graph.txt`, `task396_graph.txt`: current graph。

Final decision: **do not merge any model from this lane**.
