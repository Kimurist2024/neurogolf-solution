# task349 affine/max29 independent review 204

## Decision

**ACCEPT_GENERATOR_SUPPORT_EXACT**。最終候補
`root_task349_affine_203/task349_affine_max29.onnx`は、現在の親task349に対して
generator support上でpass-through exactかつstrict-lowerです。

| model | SHA-256 | memory | params | cost |
|---|---|---:|---:|---:|
| authority | `179bbed5bd313a1f6ec62f573fd725ab71ff55a9509daaceff3f40274ac514c7` | 3229 | 327 | **3556** |
| affine no-scalar intermediate | `849d49e462ca94b5e4f9120434a39e1982d9dce521863e80b29d80ad9b02406b` | 3233 | 316 | **3549** |
| final max29 candidate | `f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2` | 3233 | 315 | **3548** |

最終差分は **3556 -> 3548、cost -8**。現行score式のprojected log gainは
`+0.002252253204325053`です。

## Exactness proof, stage 1: affine table removal

authorityの全11 table行をint16で独立再計算しました。

- `radius = [5,0,4,1,2,0,0,0,3,0,0]`
- stored `hstart = [-14,1,-11,-2,-5,1,1,1,-8,1,1]`
- 全11行で `top = 1 - 2*r`、`hstart = top - r = 1 - 3*r`
- int8 intermediate範囲は`2*r: 0..10`、`top: -9..1`、
  `hstart: -14..1`でoverflowなし

authorityからintermediateへの機械差分も確認しました。削除されたものは
`hstart_offset_by_mod_i8`とhstart Gather、既存のtop Addだけで、追加されたものは
`r+r`、`1-2r`、`top-r`の3 nodeです。全common initializer、全その他nodeと
model fieldはbyte-identicalです。従ってこの段階は、同じradius Gatherが定義される
すべての入力でauthorityと代数同値です。

## Exactness proof, stage 2: max30 scalar removal

intermediateからfinalへの差分は次だけです。

- scalar int8 `max30_i8 = 30`を削除
- `halo_end_is30`: `Equal(halo_end, 30)` -> `Greater(halo_end, 29)`
- `beam_end_is30`: `Equal(side_i8, 30)` -> `Greater(side_i8, 29)`

既存`max29_i8`はscalar int8 29です。上記2 node以外の全node、全common
initializer、全model fieldはbyte-identicalでした。

generator ASTを独立確認すると`factor = common.randint(2,6)`で、Python
`random.randint`のinclusive rangeを使い、`size = 5*factor`です。従ってsupport上の
square sideは`{10,15,20,25,30}`。モデルはone-hot inputを`ReduceSum -> Sqrt ->
Cast -> Cast(int8)`してsideを求め、`halo_end`を`Clip(..., 0, side_i8)`します。
比較対象はどちらもint8で、必ず`x <= 30`です。

全int8値`-128..30`の159値を列挙し、
`Equal(x,30) == Greater(x,29)`を全件確認しました。よってfinalはgenerator support上で
intermediateと厳密同値です。stage 1との合成により、finalはauthorityに対して
generator-support pass-through exactです。

## Static and runtime audit

final candidateについて以下を独立再検証しました。

- full ONNX checker: pass
- strict shape inference: pass
- strict shape inference + data propagation: pass
- runtime typed trace: 120 tensors、shape mismatch 0、nonfinite 0、truthful
- standard domain opset18
- functions 0、sparse initializers 0、nested graphs 0
- Conv/ConvTranspose/QLinearConv 0、従ってshort-bias UB0
- nonfinite initializers 0

### Known

known 267件を4 ORT構成
（disable-all/default x threads 1/4）で実行しました。

- authority right: 267/267 x4
- candidate right: 267/267 x4
- raw-bitwise equality: 267/267 x4
- threshold equality: 267/267 x4
- runtime error / nonfinite: 0 / 0

### Independent fresh

親監査と異なる2 seedを使い、各2,500件を4 ORT構成で比較しました。

| seed | cases | authority/candidate right | raw-bitwise equality x4 |
|---:|---:|---:|---:|
| 20434947 | 2500 | 2492/2500 (99.68%) | 2500/2500 |
| 20434961 | 2500 | 2486/2500 (99.44%) | 2500/2500 |

合計 **20,000/20,000 case-config comparisons**でraw-bitwise一致、threshold一致、
runtime error 0、nonfinite 0です。freshのgenerator誤答はauthorityとcandidateで
完全に同一であり、候補由来の退行ではありません。依頼条件どおりpass-through exactを
採用根拠とします。

同じfresh 5,000件のfinal-candidate traceでは両seedともside
`{10,15,20,25,30}`をすべて観測し、`halo_end`範囲は3..30、Clip bound violation 0、
trace error 0でした。

## Integrity and evidence

- machine evidence for final: `audit_max29.json`
- final reproducible audit: `audit_max29.py`
- preceding all-input affine evidence: `audit.json`, `audit.py`
- protected root `submission.zip` / `submission_base_8009.46.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- protected `all_scores.csv` SHA-256:
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`

このreview laneはroot、`others/71407`、candidate、stageを一切編集していません。

Concurrent-state note: `audit_max29.json`は01:42:21に旧authority SHA `179b...`との
比較を完了しました。その後01:43:11に別laneが`others/71407/task349.onnx`をfinal
candidate SHA `f753...`へ更新しています。監査証跡は更新前の旧authorityを明示的に
SHA guardして実行したもので、このreview laneによる更新ではありません。
