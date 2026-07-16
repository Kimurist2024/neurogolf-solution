# Selector absorption / exact factor audit 177

## 結論

task246・task335・task348について、`S` の因子吸収、重複 `C` の事前合成、`C1/C2 × D` の事前合成、行列・3階テンソルのexact factor化、EinsumからMatMulへ分ける場合のactual costを監査した。

**SAFE 0件、PROBE_ONLY 0件、改善量0。** 8009.46 authorityへの統合は行っていない。strict-lower候補が1件も存在しなかったため、fresh各seed 10,000の候補ゲートは起動していない。

## Authority

| task | SHA-256 | memory | params | cost | known DISABLE_ALL | known default |
|---:|---|---:|---:|---:|---:|---:|
| 246 | `9d9428878051ec1c` | 0 | 109 | 109 | 266/266 | 266/266 |
| 335 | `79da8462ed32fe2e` | 0 | 109 | 109 | 266/266 | 266/266 |
| 348 | `b21fdf675e2415c2` | 0 | 130 | 130 | 265/265 | 265/265 |

3件ともfull checker、strict shape inference、公式相当採点を通過した。グラフは1ノードEinsumで、非出力中間テンソルは0、実行時出力shapeは常に`[1,10,30,30]`だった。

## 実モデル化したexact control

| control | 変形 | actual cost | authority raw一致 | 判定 |
|---|---|---:|---:|---|
| task246 coupled | 3本の`C×S`を単一の`S` bondへ結合 | 109 | 266/266 bitwise | REJECT: 同額 |
| task335 coupled | 同上 | 109 | 266/266 bitwise | REJECT: 同額 |
| task246 absorbed | `T=S@C`を事前計算 | 116 | 266/266 bitwise | REJECT: +7 |
| task335 absorbed | 同上 | 116 | 266/266 bitwise | REJECT: +7 |
| task246 C-square | 重複`C[t,i]C[t,i]`を`Csq`へ合成 | 139 | 266/266 bitwise | REJECT: +30 |
| task335 C-square | 同上 | 139 | 266/266 bitwise | REJECT: +30 |
| task348 CD | 全14組の`C1/C2 × D`を2行列へ合成 | 148 | 265/265 bitwise | REJECT: +18 |

全controlはDISABLE_ALLとdefault ORTの既知全件、full checker、strict inference、公式相当採点を通過し、非有限値0だった。これらは候補ではなく、局所変形のコスト下限を実測するための監査controlとして`audit_controls/`に隔離している。

## task246 / task335

`S=[0,0,-1]`のsupportは1点だけなので、3本の独立な縮約は厳密に

```text
(Σn C[n,K]S[n])(Σd C[d,A]S[d])(Σx C[x,W]S[x])
= Σn S[n]C[n,K]C[n,A]C[n,W]
```

へ結合できる。Einsum operandは44から42へ減るが、スコアはunique initializer要素数で決まるため109のままである。

直接吸収`T=S@C`は`S`の3要素を消す一方、`C`は他の縮約で必要なまま、10要素の`T`を追加する。そのため`109 - 3 + 10 = 116`となる。MatMulで実行時に作る場合は10個のfloat32中間が40 bytesとして加算され、最低でも149となる。

`S`はtask246では`C[:,3]`、task335では`C[:,8]`と一致する。しかしEinsum式だけでその列を固定することはできず、列selectorまたは事前抽出した3要素ベクトルが必要になる。後者は元の`S`そのもので削減0、既存の動的color labelと同一視すると任意入力に対する多項式が変わる。単純な`sum_n C[n,k]`も`S`とは一致しない。

重複`C[t,i]`の二乗を合成しても、`C`本体は他で使用されるため30要素の`Csq`が純増する。`Q`はexact rank 2、`C`はexact rank 3で、密行列60/30要素に対するrank factorは64/39要素となり、それぞれ+4/+9である。`B`と`M01`は全mode rank 2なので6要素のrank-1化は不可能、rank-2 CP形は12要素で元の8要素より大きい。

## task348

`D`は巨大な第1特異値のため近似rank-1に見えやすいが、binary-rational値に対するexact Gaussian eliminationではrank 3である。特異値は診断用に約`[6.8854e10, 11.4613, 2.2603]`、したがってexact rank factorは`3×3 + 3×30 = 99`要素で、直接格納90要素より9多い。

全ての`C1/C2`はprivateな3次元bondで`D`と隣接する。2つの積を共有事前計算すると、`D+C1+C2=102`要素を`C1D+C2D=120`要素へ置き換えるため+18となる。MatMulとして動的に作ると`2×30` float32中間が2本、計480 bytes増え、最低costは610である。

`C1`と`C2`は同一ではなく、片方を共有するには`D`の第0・第1行を入れ替えた別の90要素表が必要になる。6要素の`C2`削除に対し90要素追加で+84。`D`、`H`、`C1`、`C2`、`K`、`R`はいずれもfull matrix rankで、各rank factorは直接格納より大きい。

`D`の後半20列が共通でも、単一の純積factorではexact rank 3を下げられない。加法分解にはcomponent bondまたは実行時Addが必要で、追加parameterまたは中間memoryが発生する。30次元の入力空間labelを10次元表へ直接縮めるEinsumの合法なbroadcastもない。

## 安全ゲート

- sparse initializerは使用していない。保存要素数だけなら減るが、`docs/golf/BANNED_STRUCTURES.md`でgrader ERRORが実証済みのため候補外。
- lookup、private-zero、shape cloak、追加input、dynamic shape、function、nested graph、foreign domainは使用していない。
- strict-lower actual costをfreshの前提ゲートとした。通過0件なのでfresh runは0件。
- root `submission.zip` / `submission_base_8009.46.zip` SHA-256は`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`、`all_scores.csv`は`8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`のまま。

## 成果物

- `result.json`: 最終判定、root guard、known/actual-cost監査
- `audit/selector_factor_audit.json`: initializer exact rank、全control、縮約コスト内訳
- `winner_manifest.json`: `NO_WINNER`、統合なし
- `audit_controls/*.onnx`: exact変形の同額・高コストcontrol 7件
- `build_candidates.py` / `audit_candidates.py`: 再現コード

