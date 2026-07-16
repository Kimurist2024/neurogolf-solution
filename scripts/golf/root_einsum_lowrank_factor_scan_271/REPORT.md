# Exact R≥2 Einsum constant-factor scan 271

## 結論

immutable authority 400件と `others/71407/MANIFEST.json` のactive descendant
19件を走査し、active descendantがあるtaskではそれを採用したcomposite-best 400件を
候補基準にした。Einsum constant operandの全unordered nonempty axis bipartitionを、
serialized値から作ったexact rational matrixとしてrank計算した。

R≥2かつ `removed params > unique new params` となるbipartitionは6件あったが、
同一Einsum内の反復operandごとに必要な独立latent indexの空きが足りない。
したがってcandidate graphは **0件**、strict-lower候補は **0件**、winnerはnull。

authority SHA-256:
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`

active-stage manifest SHA-256:
`6f0cea4ca1c6744e5deb9f6f2ad505222d7d3203d1921114966a8c9584300016`

## 全数census

| collection | models | Einsum nodes | constant initializers | exact bipartitions | rank1 overlap | R≥2 saving | buildable |
|---|---:|---:|---:|---:|---:|---:|---:|
| authority | 400 | 605 | 867 | 1,254 | 13 | 6 | 0 |
| active descendants | 19 | 67 | 91 | 149 | 1 | 1 | 0 |
| composite-best | 400 | 607 | 869 | 1,252 | 13 | 6 | 0 |

rank1の13 partitionはlane270との重複として記録のみ行い、このlaneでは採用していない。

## R≥2 exact decompositions

| task / initializer | dtype / shape | axis partition | exact rank | params | graph uses | unused labels | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| 013 `Qor` (active descendant) | f16 / 2×2×2×2×2 | 03 \| 124 | 2 | 32→24 | 8 | 3 | reject |
| 107 `Tleft` | f16 / 3×2×2×2×3 | 014 \| 23 | 3 | 72→66 | 4 | 0 | reject |
| 398 `K` | f32 / 3×2×2×2 | 0 \| 123 | 2 | 24→22 | 10 | 5 | reject |
| 398 `K` | f32 / 3×2×2×2 | 01 \| 23 | 2 | 24→20 | 10 | 5 | reject |
| 398 `K` | f32 / 3×2×2×2 | 02 \| 13 | 2 | 24→20 | 10 | 5 | reject |
| 398 `K` | f32 / 3×2×2×2 | 03 \| 12 | 2 | 24→20 | 10 | 5 | reject |

6件すべてでcanonical exact rank decomposition `M=C·F` を作り、各factor値が元dtype
に完全表現でき、serialized factorのexact rational積が元係数と完全一致することを確認した。
また対象initializerの全graph useはfactor可能なEinsum useであり、部分置換はしていない。

しかし、同一Einsumで同じinitializerが複数回現れる場合、各
`A[...] = Σ_r U[...,r]V[r,...]` の和は独立でなければならない。同じlatent labelを
再利用すると複数のrank和が結合され、元の積とは一致しない。task013は8個の独立label
に対し空きが `X,Y,Z` の3個、task107は4個に対し0個、task398は10個に対し
`V,W,X,Y,Z` の5個しかない。このため「同一Einsum内の2 operands＋新latent index」
という指定形では構築不能である。

数字、`_`、`@`、Unicode文字を追加indexとして使うmicrographもONNX full checkerと
ORT 1.24 `ORT_DISABLE_ALL` の双方でrejectされたため、ASCII文字外でlabel枠を増やす
逃げ道もない。

## Validation gate / policy

candidate ONNXを形成できないため、full checker / strict shape inference + data_prop、
actual profile / truthful runtime shape、known-4 raw、fresh 2×2000、error/nonfinite=0、
UB0は実行していない。未検証候補を残したのではなく、実行対象が構造ゲート前に空である。

approximate/SVD近似、rank1採用、shared initializerの部分置換、runtime shape cloak、
private-zero由来候補はすべて不採用。`submission.zip` と `others/71407` は変更していない。

再現可能な全partition台帳は `scan.py` / `scan.json`、独立全rank再計算とfactor/label
監査は `audit.py` / `audit.json`、空候補台帳は `candidates.json` に保存した。
