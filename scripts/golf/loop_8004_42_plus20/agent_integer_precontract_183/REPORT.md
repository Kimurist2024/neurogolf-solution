# Integer Einsum precontraction / selector / channel audit 183

## Outcome

不採用です。8009.46 authorityの task074/task200/task211 を固定し、scan182の単一pairを
そのまま縮約するだけでなく、各initializerの全usageを覆うexact構成、selector吸収、
channel permutationまで列挙しました。valid known入力を使ったcompetition profilerで
strict-lowerは0件でした。

- 実測graph class: 98
- 左右吸収が同形になるtask211の別名まで展開した具体候補: 140
- ORT_DISABLE_ALL first-known実行: 98/98成功
- first-known符号一致: 98/98
- strict-lower: 0/140
- root ZIP / `all_scores.csv`: 変更なし

| task | authority SHA prefix | baseline actual cost | cheapest candidate | actual cost | delta |
|---|---|---:|---|---:|---:|
| 074 | `75c2b5fd...` | 135 = mem0 + params135 | global degree-channel permutation | 135 | 0 |
| 200 | `8c91d1a6...` | 346 = mem200 + params146 | bias-eliminating channel swap | 348 | +2 |
| 211 | `9ae2c13f...` | 66 = mem0 + params66 | global latent-channel permutation | 66 | 0 |

strict-lowerがないため、指示どおりstandalone full/strict-data-prop/truthful、known4、
fresh各seed 10000は走らせていません。competition profiler内で必須のfull checkerは
全候補を通過しています。

## task074 — Tfeat/Bfeat/Gfeat/poly3

Authorityは `Tfeat` 8 use、`Bfeat` 8 use、`Gfeat` 8 use、`poly3` 24 use、
`sel_hi` 12 useを単一Einsum内で共有しています。

### Pair/triple precontraction

各featureについてplain/selectedを分けた6 familyを定義し、非空subset 63通りを全実測しました。

- plain: `C[...,i] = Σd feature[...,d] * poly3[d,i]`
- selected: `C[...,i] = Σd feature[...,d] * sel_hi[d] * poly3[d,i]`

各familyの4 occurrenceは同一Cを共有します。最小のT/B単一familyでも60要素のCを追加し、
元のfeature/polyは他usageのため残るので135→195（+60）。全6 familyを覆って元feature・
poly・selectorを消しても、生成済み表が480要素となり135→498（+363）でした。

### Selector absorption

selected occurrenceの`sel_hi`をT/B/Gへ吸収する全非空subset 7通りと、`poly3`側へ一括吸収する
1通りを実測しました。TまたはBだけなら+6、全featureへ吸収して`sel_hi`を削除しても
24要素のclone−3要素のselectorで+21です。poly側は+87でした。

### Channel permutation

degree-3軸の非自明な5 permutationを、T/B/Gの最終軸、polyの先頭軸、selectorへ同時適用。
これは全有限和の単なる全単射reindexでexactですが、shape/params不変のためcost135で同点です。

## task200 — Br / conv_bias

`Br[2,30]`はfinal Einsumで2 use、`conv_bias=[1,0]`はConv biasとfinal selectorの2 useです。

- `Σr Br[r,w]*conv_bias[r]`をvector[30]へ事前縮約: 346→376（+30）。Brもbiasも別usageで残る。
- selectorをBr clone[2,30]へ吸収: 346→406（+60）。
- exact channel swap:
  - `W_kernel`と`Br`の2 channelをswap。
  - swap後のConv bias `[0,1]`は既存`oh_values`を再利用し、`conv_bias` 2 paramsを削除。
  - 動的`Bfac`は事前swapできないため、final Einsumへswap matrix[2,2]を吸収。
  - 正味 `−2 +4 = +2`、346→348。
- 上記swap後にBr/oh pairもvector化: 346→378（+32）。

この+2が最も近い候補です。動的Bfac channelを無料でpermuteできないことがstrict-lowerを阻む
graph-levelの境界です。

## task211 — P / D / M

Authorityは`P[30,2]` 14 use、`D[2]` 4 use、`M[2,2]` 3 useを共有し、cost66は全てparamsです。

### D family

- scan pair `P[c,t]*D[t]`だけをvector[30]化: +30。
- 3個の`P-D-P`と最後の`P-D`を`PD=P*D`で全coverage:
  PはDなしusageにも必要なのでPとPDの両方が残り、D 2要素だけ削除。正味+58。
- 各`P-D-P`でDを左P/右Pのどちらへ吸収するか8通りは同じshape/costとして展開済みです。

### M family

- dangling M軸まで縮約した共有`PM_rowsum[30]`を2 occurrenceで再利用: +30。
- `PM=P@M[30,2]`でMの全3 usageを覆う: Pが他usageで残り、M4だけ消えるため+56。
- chainだけのPM/MP: 各+60。
- chainを`PMP[30,30]`、dangling pairをvector[30]へ完全縮約: +926。

D/M modeの直積17 classも実測し、相互に隠れたinitializer削除がないことを確認しました。

### Channel permutation

Pのlatent軸、D、Mの両軸を同じswap `[1,0]`でreindexする構成は全入力exactですが、
shape不変でcost66の同点です。

task211では縮約順序が変わる9 classでfirst-known raw値に丸め差がありましたが、全9 classで
符号は一致しました。いずれも非lowerなのでdeep gate対象外です。

## Evidence and integrity

- 全classのvalid-input actual memory/params/cost: `actual_profile_results.json`
- 140 concrete aliasのcost表: `concrete_costs.json`
- 軽量実行screen: `one_known_screen.json`
- 再生成コード: `build_and_profile.py`, `reprofile_known.py`
- self-check: `verify_results.py` → PASS
- authority/root SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Hardmax、lookup、private-zero、近似、新規nodeは使用していません。

