# high150 — task025 / task131 / task363 exact deep audit

## Outcome

採用候補は **0件**、projected gain は **+0.000000** です。immutable authority は
root `submission.zip`、SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
です。root ZIP、score CSV、docs、`others/71407` はこのレーンでは変更していません。

3件のauthorityはいずれも4設定（ORT disable/default × threads 1/4）で既知集合を
100%通過しruntime errorは0でした。しかし、全node出力のruntime shape監査では
3件すべてが非truthfulです。したがって既存shape cloakを保持した局所shaveは
SOUND候補になりません。

| task | authority cost | known ×4 | runtime-shape truth | current-only exact | decision |
|---:|---:|---:|:---:|---:|---|
| 025 | 474 | 266/266 | trace時buffer-shape contradiction | 1 unique variant、strict-lower 0 | reject |
| 131 | 691 | 266/266 | 18 mismatches | 3 unique variants、strict-lower 0 | reject |
| 363 | 512 | 265/265 | 7 mismatches | 1 unique variant、strict-lower 0 | reject |

## Exact and attribute/factor search

current memberから dead node / unused initializer / initializer dedupe / optional output /
Identity・no-op / CSE / constant fold / metadata normalization / combined pass を再生成しました。
3件ともdead node、unused initializer、initializer alias、duplicate producer、removable
optional outputはありません。task131の唯一のconstant-fold候補は`Shape`の固定化で
shape carrierを露出し、full checkerで4対1等のshape矛盾になりました。

`CastLike -> Cast(to=...)` はinitializer witnessを属性へ埋め込む全subsetを列挙しました。

- task025: 1 subset。cost `474 -> 72466`、strict-lowerなし。
- task131: 31 subsets。最小はcost 691のtie、次点703/714、strict-lowerなし。
- task363: 7 subsets。最小はcost 512のtie、他は9511、strict-lowerなし。

Slice/Reduce/Squeeze/Unsqueeze/Resizeの該当入力は現opsetではfree attributeへ移せません。
Shape/Attention/CenterCropPad/QLinearConvでschemaが許す設定は既に属性です。

initializer factorも全initializerについて下限を計算しました。N要素initializerを通常nodeで
再構成すると、最低でもN要素のintermediateが新たに課金されます。既存shape carrierを無料、
追加scalarを1 parameterと楽観しても差分下限は`N*itemsize + 1 - N`であり、最小値は
task025 `+8`、task131 `+1`、task363 `+1`です。よってmaterialized factorはstrict-lowerを
作れません。dtypeのみの変更もparameterが要素数課金なので削減になりません。

Sparse/Constant-sparseは親レーンの横断担当と重複するため、指示どおり除外しました。

## Strict-lower history disposition

現authorityより実コストが低い既知履歴5件も現行基準で再監査しました。以下が
**strict-lower候補の全件**です。

| task | SHA-256 | cost | evidence | decision |
|---:|---|---:|---|---|
| 025 | `0ff9ea73675b4c9566e0edd760a0f46d0475f882f7b66327064c23a8ca854e82` | 472 | 4設定すべて0/266、raw equality 0/266、25-input Einsum、shape trace error | reject |
| 025 | `9402de87c604dd69330d0122a24c4c25bbbc0ae66c61bad57fb351eb5103d847` | 472 | 4設定すべて0/266、raw equality 0/266、25-input Einsum、shape trace error | reject |
| 025 | `a8b2e0eecb42f3c2c7a6ea6b5878a54127537fa1270b0f165bd8ae20fe533f86` | 472 | 4設定すべて0/266、raw equality 0/266、25-input Einsum、shape trace error | reject |
| 131 | `5ac0fc168a52cb7f4d6fdee9b82aecc6346ba8c8ae0d60b13a72d3f03da5331c` | 627 | boolean 266/266 ×4だがraw equality 0/266、TfIdf lookup、11 shape mismatches | reject |
| 131 | `a13e6337acc30ddc9bc7f3276f3e464cc8144c12d40577bdd625d721ab1db182` | 596 | boolean 266/266 ×4だがraw equality 0/266、TfIdf lookup、11 shape mismatches | reject |

task363にはcost 512未満の実コスト候補はありませんでした。上記5件はいずれも
known/raw・policy・runtime-shapeのpre-fresh gateで落ちるため、2-seed freshを実施しても
採用判定は変わりません。freshはfail-closedで省略しました。

## True-rule rebuild controls

- **task025**: full guide lineを保存し、疎pixelを消し、同色guideの同じ側に投影する規則。
  truthful controlはknown 266/266ですがcost 370205で、474を下回りません。
- **task131**: green creatureをred lineの反対側隣接位置へ移し、その外側にcyanを置く規則。
  truthful controlはknown 266/266ですがcost 24927で、691を下回りません。
- **task363**: red exemplarをblackへ戻し、全ての合法translationを検出してredで塗る規則。
  truthful controlはlegal fresh 3000/3000、cost 12542ですが固定既知は263/265です。

task363の2件の固定fixtureはgenerator関係からinput-onlyで識別不能です。第2fixtureに
追加location `(1,3)` を加えると、同一inputのまま異なる合法outputが成立します。従って
deterministicなSOUNDモデルがfixture shimなしで固定既知100%と全合法relationを同時に満たす
ことはできません。

## Evidence

- `audit/inventory_exact.json`: authority SHA/cost、full/strict/UB0、runtime trace、exact scan。
- `audit/current_known_four_configs.json`: authorityの4設定既知監査。
- `audit/attribute_factor_scan.json`: CastLike全subsetとinitializer factor下限。
- `audit/strict_lower_history.json`: strict-lower 5件のSHA/cost/4設定/shape証拠。
- `audit/true_rule_controls.json`: truthful controlとtask363非同定可能性。
- `manifest.json` / `winner_manifest.json`: machine-readable disposition。

Final decision: **do not merge any model from this lane**.
