# task257 / task310 mask absorption audit (lane 174)

## Decision

**NO APPLY. Exactかつstrictly lowerな候補は0件。**

authorityは`submission_base_8009.46.zip`（SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`）
に固定した。全候補はfull checker、strict shape inference + data propagation、
公式互換profiler、ORT default/disabledで検証した。root/submissionは変更していない。

## task257: `mask[r] mask[c]` → `feat` absorption

baselineはmemory 0 + params 114 = **114**。内訳は`feat[2,30]` 60、
`proj_a[2,2]` 4、`mask[30]` 30、`color[2,10]` 20。

`feat`のindex roleは次の通り。

- source row `h`: positions 1=`ph`, 3=`qh`, 6=`th`
- output row `r`: positions 2=`pr`, 4=`sr`, 7=`ur`
- source col `w`: positions 9=`vw`, 11=`jw`, 14=`lw`
- output col `c`: positions 10=`vc`, 12=`kc`, 15=`mc`
- cap: positions 17=`r`, 18=`c`

出力側の任意のr-factor 1本とc-factor 1本にmaskを掛ければ、mask operandsは
代数的に除去できる。しかし同じ`feat` initializerはsource側の`h,w`にも共有
されるため、initializer全体を変更するとsource係数も変わる。

さらに`feat[:,0:4]`の各列パターンは`feat[:,5:9]`に反復している。つまり
maskなしのshared factorだけでは、出力index 0..3と5..8を区別できない。
baselineはmaskで前者だけを残す一方、source側では後者も必要である。この
非対称性を同じshared tensor 1本で同時に表すことはできない。

実測:

| candidate | rewrite | cost | official correctness | all-input判定 |
|---|---|---:|---:|---|
| `feat_global` | 全12個のfeat occurrenceをmasked featへ変更 | **84** | false | reject。最初の合成probeでraw 5セル・threshold 1セル差、max delta 9219.13（両ORT同一） |
| `feat_clone` | `pr`,`vc`だけ`feat_cap=feat*mask`へ変更 | **144** | true | exact。既知全例raw bit一致、合成8/8×両ORT raw bit一致 |

exact cloneはmask 30 paramsを消す代わりに`feat_cap[2,30]` 60 paramsを追加する。
したがって114→144（score差 -0.233615）で、30 params悪化する。

## task310: `cap[R] cap[S]` → `P0/P1/P2` absorption

baselineはmemory 194 + params 307 = **501**。各`Pk[30,2]`は60 params、
`cap[30]`は30 params。

final Einsumでのshared role:

- `P0`: source positions 0=`ha`, 18=`wd`; output positions 3=`RA`, 21=`SE`
- `P1`: source positions 1=`hb`, 19=`we`; output positions 4=`RB`, 22=`SF`
- `P2`: source positions 2=`hc`, 20=`wf`; output positions 5=`RD`, 23=`SG`
- cap: positions 14=`R`, 30=`S`

`P0/P1/P2`の3本はR/Sを3-bit（mod 8）で表し、行パターンが8周期で反復する。
したがってshared P factorsだけではR=0と8、1と9、…を区別できない。`cap`
は最初の周期0..7だけを残すが、source側のh/wでは8以降のP表現も必要である。
出力側だけを変える独立factorまたは位置依存vectorなしにcapを除去することは
できない。

P0/P1/P2それぞれでglobal版とselective clone版を実測した。

| candidates | cost | official correctness | all-input判定 |
|---|---:|---:|---|
| `p{0,1,2}_global` | **471** | false | reject。probe 0でraw/threshold各10セル差、max delta 6（両ORT同一） |
| `p{0,1,2}_clone` | **531** | true | exact。3案とも既知全例raw bit一致、合成8/8×両ORT raw bit一致 |

cloneは出力occurrence 2箇所だけを`Pk_cap=Pk*cap`へ差し替え、source occurrence
2箇所は元のPkを保持する。これは全入力に対して代数等価だが、cap 30 paramsを
消してclone 60 paramsを追加するため501→531（score差 -0.058156）となる。
memoryは194のままで、実測でも30 params悪化した。

## Gate conclusion

- 安価なglobal案: task257 84、task310 471だが、既知goldと合成反例の両方で失敗。
- 全入力exactなselective clone案: task257 144、task310 531でコスト悪化。
- exactかつ`<114` / `<501`の候補がないため、known4、fresh10000、truthful
  shape、UB/no-lookup gateは仕様どおり未実行。誤答候補に深い汎化試験を行っても
  採用可能にはならない。

全index use、各候補のhash、checker/profiler/ORT結果、合成probeのセル差は
`results.json`に保存した。

## Integrity

- `submission.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- task257 source SHA-256:
  `32e5452b9089ab217e13dc4aac064b7807f9397603a66e5a5a945ffa3b0f5ef6`
- task310 source SHA-256:
  `4eed21efedf2b44e11d2bb748d383275d193144c3c0f8f9f55265c8639e6fdec`
- 採用・merge・root変更: 0件

