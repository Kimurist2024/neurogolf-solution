# tasks025 / 117 / 131 / 330 — high-memory POLICY90 residual audit

## Verdict

**採用候補は0件、projected gain は +0.000000 です。** immutable authority は
`submission_base_8009.46.zip`、SHA-256 は
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927` です。
現在グラフに対する safe exact optimizer と代数的残余探索を72 variant実行し、
37件はstructure gate、35件はstrict-lower gateで停止しました。strict-lower は0件でした。

| task | authority SHA-256 | memory + params = cost | authorityのtruth問題 | truthful rule control | final |
|---:|:---|---:|:---|---:|:---|
| 025 | `22a44063541e…` | 396 + 78 = **474** | traceがbuffer-shape矛盾で停止 | 370,205 | no candidate |
| 117 | `042e3ee0976a…` | 496 + 109 = **605** | strict失敗、10 shape mismatch、default ORT失敗 | 6,762 | no candidate |
| 131 | `d72faec06556…` | 572 + 119 = **691** | 18 shape mismatch | 24,927 | no candidate |
| 330 | `af2a81db8b4b…` | 730 + 166 = **896** | 38 shape mismatch、default ORT失敗 | 5,525 | no candidate |

authority costはこのレーンで再プロファイルした値です。task025/131 authority は既知集合を
ORT disable/default × threads 1/4 の4設定すべてで266/266通します。task117/330 は
`ORT_DISABLE_ALL` ではそれぞれ265/265、266/266ですが、default ORTは
`CenterCropPad` でsession作成に失敗します。これらは比較基準として保持するだけで、
新候補にshape cloakやdefault失敗を引き継ぐことは許容していません。

## Generator truth

- **task025 / global line-side projection**: 12..30の各軸上にあるfull guide lineを保存し、
  sparse probeを消し、同色guideのprobeと同じ側の隣接セルへ色を投影します。タスク全体の
  transposeが変動します。guide位置、色、probeの側を入力から求める必要があります。
- **task117 / global reflection**: 5-cell X bodyの中心を検出し、別色のleg spriteをその中心の
  上下・左右へ反射して4象限を完成します。size 12..15、sprite形状とflipが変動します。
- **task131 / global geometry**: green creatureをred lineの反対側隣接位置へ移し、その外側に
  full cyan separatorを置きます。creature extent、line位置、flip、transposeが入力依存です。
- **task330 / component topology**: separated 4-connected gray componentごとにsizeを数え、
  size=6だけred、それ以外をblueにします。10x10上に3..6 component、size 4..8です。

いずれも公開例ルックアップではなく generator rule を実装したtruthful controlが既知集合を
完全通過しています。task025/131 controlの独立referenceは過去fresh 3000/3000ですが、
control costはauthorityより桁違いに高く、strict-lower候補ではありません。task117/330の
controlも両ORTモードで全既知を通しruntime-shape mismatch 0ですが、同様に高コストです。

## Current residual search

各authority memberに対し、dead node / unused initializer / initializer dedupe / no-op /
CSE / unused optional output / constant fold / associative absorb / combined pass /
value-info normalizationを走査しました。加えてinitializer witnessを持つ全`CastLike`を
`Cast(to=...)`へ置き換える全subsetを列挙しました。

| task | variants | CastLike subsets | structure reject | valid costs | strict-lower |
|---:|---:|---:|---:|:---|---:|
| 025 | 2 | 1 | 1 | 72,466 | 0 |
| 117 | 34 | 31 | 32 | profile不能/非lower | 0 |
| 131 | 34 | 31 | 3 | 最小691 tie、以後703以上 | 0 |
| 330 | 2 | 1 | 1 | 9,895 | 0 |

全initializerについて「既存shape carrierは無料、追加scalarは1 parameter」という楽観条件で
factor/materialize下限も計算しました。`N*itemsize + 1 - N` は全tensorで非負のため、
通常nodeでinitializerを再構成してstrict-lowerにする余地はありません。

task330では既知の残余として6個の`ConstantOfShape([1])`をinitializerへfoldすると名目上
`-42`になります。しかし候補SHA
`53a38bc9b0678ebb40bbeb210e0038b513fb7b4689160a804ed9d1858c9b7338` はhidden shapeを露出し、
full checker、strict data propagation、両ORT sessionの全てで失敗します。

## Strict-lower history and POLICY90

- task025: cost472の3 SHAは4設定すべて0/266で、各設定2,394,000 nonfinite values、
  25-input `Einsum`、runtime-shape trace failureです。
- task117: cost605未満の履歴候補はありません。
- task131: cost627 / 596の2 SHAはthreshold上266/266ですが、`TfIdfVectorizer` lookup、
  11 runtime-shape mismatches、raw authority equality 0/266です。lookup/cloak禁止で棄却します。
- task330: cost807 / 808 / 817は既知166/266、162/266、210/266で、最良でも78.95%です。
  全て2 runtime-shape mismatchesを残します。

従ってnormal task向けの>=90% POLICY90 candidateも0件です。private-zero例外へ回す候補も
ありません。

## Fresh and four-config disposition

promising candidateが0件なので、候補に対するcomplete-known four-config runと2 disjoint
fresh seed runは0件です。これはfail-closed early stopです。actual strict-lower、checker、
strict/truthful shape、standard/no-lookup、known/POLICY90のpre-fresh gateを通らない候補は、
freshで採用可能にはなりません。

## Evidence and protection

- `audit/authority_costs.json`: authority再プロファイル。
- `audit/residual_scan.json`: 72 exact/algebraic variants、graph inventory、factor下限。
- `RESULTS.json`: task別SHA、cost、history、gate counts、verdict。
- `winner_manifest.json`: empty winner manifest。
- `scan_residual.py`: 再実行可能なfail-closed scan。

repo内に`AGENTS.md`ファイルは存在しなかったため、依頼文に埋め込まれたAGENTS指示を適用
しました。Kimiと`try_candidate.py`は使用していません。root `submission.zip`、
`submission_base_8009.46.zip`、`all_scores.csv`、`others/71407`、stagingはこのレーンでは
変更していません。終了時のroot submission SHAはauthority SHAと一致し、
`all_scores.csv` SHAは
`8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`です。
