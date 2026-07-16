# tasks182 / 204 / 208 / 284 high-memory・POLICY90 audit

## 結論

**採用候補は0件、projected gainは +0.0 です。** 現行 authority を維持します。

authority は LB 8009.46 の root `submission.zip` で、SHA-256 は
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`。
`submission_base_8009.46.zip` と byte-identical です。この lane は root ZIP、
`all_scores.csv`、`others/71407`、`artifacts/handcrafted` を変更していません。

| task | authority SHA prefix | cost (memory+params) | known/runtime | runtime shape | decision |
|---:|---|---:|---|---:|---|
| 182 | `625b31492d91` | 949 = 893+56 | disable-all 267/267、default load failure | 47 mismatch | keep |
| 204 | `312fa4435c54` | 2222 = 2076+146 | disable-all 268/268 × threads 1/4、default load failure | 53 mismatch | keep |
| 208 | `6c9bad970152` | 1422 = 1300+122 | disable-all 266/266 × threads 1/4、default load failure | 9 mismatch | keep |
| 284 | `0d03efd73a59` | 517 = 465+52 | 266/266 × 4 ORT configs | 11 mismatch | keep |

4 incumbents は full checker と strict shape inference/data propagation を通りますが、
実行 shape は宣言と一致しません。task182 はさらに dynamic QLinearConv bias の
channel 長を静的に証明できず、candidate 用の UB0 gate を満たす安全な継承元では
ありません。task204/208/284 の Conv-family UB finding は0です。

## generator truth

- **task182 / `776ffc46`**: gray 7x7 frame 内の sprite を取り出し、同じ
  translation-normalized shape の外部 sprite を frame 内 sprite の色へ変更する。
  独立 reference は known 267/267、fresh 2 seed 合計3000/3000。
- **task204 / `868de0fa`**: 非重複の青い正方形輪郭を、辺長が奇数なら7、偶数なら2で
  塗る。authority は fresh 2 seed 合計3000/3000。
- **task208 / `890034e9`**: 最初の box 内の黒い cutout と同寸法の2番目の黒領域を探し、
  その周囲へ box 色の輪郭を描く。authority は fresh 2952/3000 = **98.4%** で、
  現職自体は既存の normal-POLICY90 相当。ただし新しい改善候補ではない。
- **task284 / `b7249182`**: collinear な異色2 seed を内向きに延ばし、直交5-cell cap と
  serif を付ける。transpose も入力依存。独立 reference は known 266/266、fresh
  3000/3000、境界 parameter sweep 1960/1960。

## 全履歴の再走査

現行 authority に対し、repository 内の loose ONNX と全 ZIP member を再列挙し、
SHA-deduplicate しました。

- loose observations: 2,596
- ZIP files: 1,284、target member observations: 4,833
- authority duplicates: 263
- unique non-authority SHA: **265**
  - task182: 61
  - task204: 80
  - task208: 74
  - task284: 50
- inventory error: 0

checker/strict data_prop、standard ops、static floor、実 cost の順に絞ると43件が
実 cost screenへ進みました。task182/204/284 は authority より実際に安いものが0件。
task208 のみ5件が1417--1419で strict-lower でしたが、全件が default ORT で
`CenterCropPad` の shape 要素数1 / axes数2エラーとなりました。disable-all では
266/266でも、4設定必須・truthful-shape必須なので全件 reject です。

この5件中、SHA `2e2e6f...` と `3d6e01...` は `others/71405` の明示的 LB-black。
private-zero 用の all-support pass-through 証明もありません。残り3件も同じ
default-ORT/shape-cloak hard gate で落ちるため normal-POLICY90 候補にはなりません。

最終 pre-fresh survivor は **0**。指示どおり候補用の大規模 fresh は開始していません。

## optimizer cleanup と有限支持・代数監査

### task182

dead/unused/CSE/initializer alias はありません。Identity bypass と fixed-Shape fold は
既存 `CenterCropPad` の rank/shape 矛盾を顕在化させ、full checker または strict
data propagation を失います。memory-heavy `cnt6 = Mul(uint8, 6)` を `Selu` へ置換する案は、
Selu が uint8 を受けないため schema-invalid です。

### task204

主要 memory は BitwiseXor 504、BitwiseAnd 360、Mul 360、Concat 240 bytes。
XOR prefix の加算/乗算化は成立しません。合法 mask `0x00000005` で現行
`0x03000003` に対し因数化は `0x04fffffb` となります。固定 batch Shape fold は
nominal lower でも allocator 計画を変え、disable-all の Slice buffer reuse mismatchを
起こします。全15 lower-static history は実 cost 2222以上、profile failure、または
structure failureでした。

### task208

主要 memory は 19x19 border Less 361、Cast 158、Einsum 132、Slice 130 bytes。
uint16 ArgMax / BitShift は target CPU kernelなし。uint8 polynomial rewrite は
全256^3入力で同値でも uint8 Einsum kernelなし。`x/2 == x*0.5` は全65,536 float16
bit patternで同値ですが cost-neutral。隣接積まで再結合すると1,286反例があるため
採用不能です。

### task284

`Shape(input)[0:1]=[1]` と `Shape(x70)[0:1]=[56]` は整数範囲で厳密ですが、literal fold は
既存 rank/先頭次元矛盾を露出し strict structure を失います。dead node、unused/
duplicate initializer は0。非authority の最小実 cost は518で、現行517を下回りません。

## 最終判定

| gate | result |
|---|---:|
| unique history SHA | 265 |
| actual cost screen jobs | 43 |
| actual strict-lower | 5 (task208 only) |
| four-config + truthful-shape survivor | 0 |
| POLICY90 survivor | 0 |
| candidate fresh runs | 0 |
| winners | **0** |
| projected gain | **+0.0** |

機械可読な cost/SHA/count/verdict は `result.json`、空の採用一覧は
`winner_manifest.json`、全265 SHAの stage は `rescreen.json` にあります。
