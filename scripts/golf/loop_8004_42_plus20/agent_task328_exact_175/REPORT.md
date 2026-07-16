# task328 exact/private-zero audit (authority 8009.46)

## 結論

task328 に strict-lower 候補を1件確定しました。

| | authority | candidate |
|---|---:|---:|
| cost | 558 | **553** |
| memory | 200 | 200 |
| params | 358 | **353** |
| task score | 18.6756410376 | **18.6846419985** |
| projected gain | — | **+0.0090009609** |

Candidate:
`candidates/task328_exact553_split_p0.onnx`

SHA-256:
`cc2718047fec6d65bf1e6336fc3aac21c4fc8838774f5e0407c570fbec39fd5b`

通常の margin gate は通りませんが、ユーザーが明示した「private-zero 由来でも
全通過保証があれば可」に対し、official target の ORT CPU 実行範囲で全 generator
support を有限証明したため、判定は **SAFE_PRIVATE_ZERO** です。root の ZIP、
CSV、score ledger はこの lane では変更していません。

## 5 parameter shave

authority の task328 member は SHA
`08ba1aa525d67f290c13e7b79aef339aeb5912bf0d1b0b379ff6ab8792cf576a`、
actual cost 558 です。候補は次の二つの恒等変形だけを行います。

1. `e[4] = [1,0,0,0]` の4回の使用を既存 `J` の
   `e[a] = sum_t J[t,a,a]` で置換し、4 parameters を削除。
2. `z[0]=one=1` を既存 `ninvB=-1/3` へ置換し、全 `CoreB[:,:,0]`
   を `-3` 倍。float32 で `(-1/3)*(-3) == 1` が厳密に成立し、1 parameter
   を削除。

ノード数8、最大 Einsum inputs 58、runtime tensor shapes は不変です。

## 全71,136 generator states の保証

generator `task_d22278a0.py` の状態数は次のとおりです。

```text
13 sizes (6..18) *
  [C(4,2)*P(9,2) + C(4,3)*P(9,3) + C(4,4)*P(9,4)]
= 71,136
```

非ゼロ色の permutation を除けば、幾何状態は
`13 * (6 + 4 + 1) = 143` 代表です。色縮約は例依存の仮定ではありません。

- `MaxPool` は channelwise。
- `ReduceL2` は channel permutation invariant。
- color axis 10 を持つ initializer は `Ssel` のみ。
- `Ssel` の非ゼロ色 columns 1..9 はすべて同一。
- したがって非ゼロ色の任意の permutation は free output color axis を同じ
  ように permute し、logit と符号を保存する。

143代表を実 ONNX で全実行した結果は以下です。

| configuration | correct | wrong | errors | nonfinite | false positive |
|---|---:|---:|---:|---:|---:|
| ORT_DISABLE_ALL, thread 1 | 143/143 | 0 | 0 | 0 | 0 |
| ORT_DISABLE_ALL, thread 4 | 143/143 | 0 | 0 | 0 | 0 |
| default ORT, thread 1 | 143/143 | 0 | 0 | 0 | 0 |
| default ORT, thread 4 | 143/143 | 0 | 0 | 0 | 0 |

これにより official target の4構成について全71,136状態が被覆されます。
さらに独立 seed `328175101` と `328175102` を各10,000件生成し、両方で
10,000/10,000 が証明済み集合に写り、invalid 0、各 seed とも143代表を
すべて観測しました。

## known・structure gate

- known 267件: 4構成すべて 267/267、wrong/error 0。
- full checker / strict data propagation: pass / pass。
- static shapes: 全て正、runtime/declaration mismatch 0。
- standard domain のみ。banned/Sequence/nested graph/function/sparse/lookup なし。
- finite initializers、Conv bias finding なし。
- actual profile: memory 200 + params 353 = cost 553。

## margin の明示

この candidate は通常の margin-stable candidate ではありません。

- finite support 143代表あたり `(0,0.25)` の true values: 1,182。
- minimum positive: `2.7771885962947234e-19`。
- maximum absolute raw: `1.4418116801831403e34`。
- nonfinite: 0、false positive: 0。

したがって任意の ORT/provider/hardware まで一般化する主張はしません。採用保証は
NeuroGolf の official ORT CPU profile と、検証した disabled/default × thread
1/4 に限定します。この限定はユーザーの private-zero 完全保証例外に対応します。

## 追加探索

### CoefAB margin repair

終端 Einsum から16要素 `CoefAB` を外し、全143代表から1,287,000 feature rows、
66,869 unique feature vectors を構築して unit-margin LP を解きました。7 vectors
が正負両 label で衝突し、さらに true cell に zero feature vector が1件あるため
infeasible です。現 topology のまま CoefAB だけを通常 margin classifier に
置換することはできません。

### 指定 factor shave

- `CoreB[4,4,5]` は mode rank `(4,4,5)`。疎な exact CP rank-6 表現は
  80→78 parameters ですが、8回の独立使用には private rank letters が8個必要。
  exact554 equation の未使用 letters は7個だけで、共有すると独立和が結合して
  非同値になります。
- e を `CoreB[0,:,:]` に直接吸収すると rank-1 factor でも +5 parameters、
  CP と併用しても +3 parameters。
- `sum_q Ssel[q,y]*frow[q]` の fused vector は10 elements で、現 frow 2 elements
  より +8。
- `TFeat` の sign/coordinate 分離は exact ですが 120→136 parameters。

よって今回の safe floor は cost 553 です。

## 成果物

- `final_audit.json`: 4構成、全support、known、fresh mapping、structure。
- `stable_coef_search.json`: 全support CoefAB feature/LP 証拠。
- `factor_shave_analysis.json`: CoreB/frow/Ssel/sign factor のexact計数。
- `candidate_screen.json`: authority/exact554/exact553 補償配置 witness。
- `result.json`, `winner_manifest.json`: 採用判断と固定SHA。
