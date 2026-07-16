# task012 8×8 group10 Conv no-bias search 274

## 結論

親のnormal-POLICY90候補
`9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947`
は公式profile `memory0 + params650 = cost650`。bias `[10]` を完全に除いた同じ
output-only group10 Convは `memory0 + params640 = cost640` になる。

しかし、背景用kernelを独立にし、非zero channels 1..9に同一kernelを共有させる指定
familyでは、完全な196 generator statesのうち **単一stateすらcenter/armを同時に
線形分離できない**。case-level最適上限は厳密に **0/196** で、POLICY90に必要な
177/196へ届かない。candidateとwinnerはnullで、cost650親を維持する。

## Exact linear census

対象はgenerator `task_0962bcdd` の
`7 col0 × 7 col1 × 4 gravity = 196` states。代表色 `[1,2]` を使った。group10で
channels 1..9のkernelが同一なら、全非zero色置換へそのまま一般化する。

各stateについて、30×30出力位置の8×8 patchを背景、center、armに分けて列挙した。
scorerの判定は `raw > 0` なので、biasなしkernel `w` の制約は次の通り。

- positive patch `p`: `w·p ≥ 1`（任意の正margin解はscale可能）
- negative patch `n`: `w·n ≤ 0`（raw=0は正しくnegative）

| 判定 | 個別feasible states |
|---|---:|
| background kernel | 196/196 |
| shared center/arm kernel | 0/196 |
| complete case | 0/196 |

foregroundの全196 infeasibilityについてexact rational Farkas certificateを保存した。
各certificateは7、11、または13本のpatch制約だけを使い、係数はすべて `1/2` または
`1/4`。非負係数付きsigned patch和が64成分すべてexact zero、positive-row係数和が
exact 1なので、制約右辺の同じ和は `-1` となり `0 ≤ -1` の矛盾を与える。
したがってこれは係数boundや浮動小数MILP toleranceに依存しない上限証明である。

親weightからbiasだけを落としたdiagnostic runtimeも196 states中0件正解だったが、
採否根拠はこの実測ではなく上記の全state exact certificateである。

## Candidate gate / policy

最適上限0/196のため、>=90%解だけに要求されたknown265、domain196 candidate runtime、
fresh独立2×10000×4 ORT configurations、full/strict/truthful/static/standard/finite、
UB0、error0、shape0のcandidate gateは実行していない。未検証候補は保存していない。

lookup、fixture correction、shape cloak、private-zero routeは使っていない。
`submission.zip`、root artifacts、`others/71407` は変更していない。

再現可能な探索と全certificateは `search.py` / `search.json`、独立再生成・exact証明
監査は `audit.py` / `audit.json`、空候補台帳は `candidates.json` に保存した。
