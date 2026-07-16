# 71407 — LB 8009.46 cheaper candidates

基準は `submission_base_8009.46.zip`、SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`、
MD5 `2dc6d412ddd8bd3102f42775155e4a38` です。

直下の `*.onnx` は、8009.46の同一タスクよりstrictに安い検証済み候補です。
task007/012/161/175/192/344/355以外は真ルールまたは全入力等価性まで証明しています。task012は
全196生成幾何94.90%・known95.09%・独立fresh最低94.72%のcleanな単一Convです。task175は
ランダム生成の受理support内では厳密ですが固定validate 4件だけを外すため、
task192/344と同様にユーザー指定の90%採用基準による `POLICY90` として明示しています。

| file | cost | projected gain | evidence |
|---|---:|---:|---|
| `task007.onnx` | 70 → 68 | +0.0289875369 | POLICY90: cleanなoutput-only Einsum。known260/266、一次fresh19501/20000＋独立fresh19497/20000、4 ORTでsign安定・error/nonfinite/shape差/small-positive 0 |
| `task012.onnx` | 710 → 650 | +0.0882926071 | POLICY90: cleanなoutput-only depthwise Conv。known252/265、全196幾何186/196、一次fresh18977/20000＋独立fresh18974/20000、4 ORTでsign/raw安定・error/shape差0 |
| `task013.onnx` | 357 → 356 | +0.0028050509 | 有効入力代数証明＋非負float16全31744値bitwise総当たり＋known/fresh raw一致 |
| `task066.onnx` | 562 → 551 | +0.0197670407 | 全1861056生成幾何＋全uint8 carrier証明、green selectorを既存factorから厳密再構成しraw一致 |
| `task090.onnx` | 1050 → 1049 | +0.0009528348 | 正のbitset値に対するDiv→Selu、全31744値bitwise総当たり＋known/fresh dual-ORT raw一致 |
| `task101.onnx` | 5655 → 5641 | +0.0024787548 | And(all-true,b)→shape-preserving Expand、known全件＋fresh 10000×両ORT raw一致 |
| `task134.onnx` | 423 → 422 | +0.0023668650 | 非負float16のMul→Selu、全31744値bitwise総当たり＋known/fresh dual-ORT raw一致 |
| `task158.onnx` | 7578 → 7498 | +0.0106129943 | 48局所構成の完全supportとTopK zero下限を証明し、anchor role/phaseをuint8 bitmaskへ厳密化。known1064＋独立fresh12000 raw一致 |
| `task161.onnx` | 190 → 186 | +0.0212773984 | POLICY90: cleanな3-node graph。known265/266、一次fresh19872/20000＋独立fresh19859/20000。terminal `poly` のfloat32×8でrawを厳密一様scaleし、4 ORTでerror/nonfinite/shape差/small-positive 0 |
| `task175.onnx` | 166 → 145 | +0.1352540459 | POLICY90: known262/266。ランダム生成がrejectする対称同時欠損だけ失敗し、一次fresh24000＋独立fresh16000実行は全正解・error/shape差0 |
| `task192.onnx` | 1609 → 1134 | +0.3498616627 | POLICY90継承: 選択vector/adj不変の代数shave、親と全13060 raw一致。到達可能反例は親と同じ |
| `task205.onnx` | 1042 → 1038 | +0.0038461586 | `ReduceSum(row_mask)+Mul`→`Einsum("ri,->")`。ORTがr/iを先に縮約してK=1積を行う実装順序とbinary `[30,1]` 全supportでbitwise同値、known 4設定＋独立監査＋generator2000 raw一致 |
| `task209.onnx` | 2087 → 2085 | +0.0009587728 | Mul/Div→Seluで2定数削減、各全31744値bitwise総当たり＋known/fresh dual-ORT raw一致 |
| `task226.onnx` | 372 → 370 | +0.0053908486 | private-zero由来だが全136生成状態×4 ORTを完全証明、known/fresh raw一致・error 0 |
| `task245.onnx` | 385 → 384 | +0.0026007817 | 4つの正値LogのDiv2→Selu(.5)、非負float16全31744値＋known/fresh 4 ORT raw-bitwise一致 |
| `task310.onnx` | 501 → 491 | +0.0201619733 | parity tensorを共有Hadamard因子へ厳密分解。known1064＋fresh40000でauthorityとraw bit一致、delta/error/shape差0 |
| `task319.onnx` | 1003 → 975 | +0.0283133170 | private-zero authorityの完全support pass-through＋3入力恒等変換、known/fresh 4 ORT raw一致・新規shape差0 |
| `task328.onnx` | 558 → 553 | +0.0090009609 | private-zero由来だが全71136生成状態を143 orbitへ厳密縮約し4 ORT構成で完全証明 |
| `task333.onnx` | 423 → 421 | +0.0047393454 | GE=[1,-1]の共有factor吸収、全モノミア等価証明＋変更factor全80値＋fresh 8000件raw一致 |
| `task344.onnx` | 137 → 132 | +0.0371790032 | POLICY90: known266/266、新規fresh19962/20000(99.81%)×4 ORT、candidate-authority符号差0・error/shape差0 |
| `task349.onnx` | 3564 → 3532 | +0.0090192269 | affine/tableと5残差変換を完全generator support証明、known267×4＋fresh累計40000 raw一致 |
| `task355.onnx` | 250 → 249 | +0.0040080214 | POLICY90: known264/267、一次fresh19724/20000＋独立fresh19731/20000、4 ORTでerror/nonfinite/shape差/small-positive 0。public overfit-riskのためexactとは分類しない |
| `task366.onnx` | 7987 → 7985 | +0.0002504383 | 2定数の非負float16 Mul→Selu、全31744値bitwise総当たり＋known/fresh dual-ORT raw一致 |

23件合計のローカル推定改善は `+0.7881256397`、推定値は
`8010.2481256397` です。exact SHAのLB確認前なので、LB確定スコアにはまだ
加算しません。

task192の完全保証版(cost1143, SHA `5c5eaefa...`)は
`FALLBACK_EXACT_DO_NOT_AUTO_MERGE/task192_exact1143.onnx.fallback` に保存済みです。
POLICY90を使わない運用ではこのフォールバックに戻せます。

8008.14 → 8009.46で固定された24タスクに012/013/066/090/101/134/158/175/192/205/209/226/245/310/319/328/333/344/349/366は含まれません。
8009.46内の20タスクは8008.14時点と同じSHA・同じ公式costなので、候補の
安全性証明とstrict-lower差分はそのまま維持されています。詳細は
`REBASE_8009_46.json` を参照してください。

`PROBE_ONLY_DO_NOT_MERGE/` は90%台の近似候補です。拡張子を
`.onnx.quarantine` にしてあり、通常の `*.onnx` 一括マージから除外されます。
過去にpolicy90/95候補が私的0点になったため、isolated LBで白と判明するまで
拡張子を変更したり直下へ移動したりしないでください。task344だけは独立
normal-POLICY90再監査とユーザーの90%採用指示により、同SHAのactive copyを直下に置いています。

`task344_cost132.onnx.quarantine` はactiveな `task344.onnx` と同一SHAの証拠保存
copyです。generator到達可能な10x10反例でauthorityとの差が1セルあるためexact
分類にはしていませんが、既知266/266・独立fresh19962/20000（99.81%）・error 0
を確認したnormal `POLICY90` としてactive採用しています。この
`+0.0371790032` は上記23件の候補合計に含まれます。

`task205_cost1038.onnx.quarantine` は全support証明完了後、同一SHA
`43c963c4...` をactiveな `task205.onnx` へ昇格しました。`row_mask` は厳密な
binary `[30,1]` で、対象ORTは先にReduceSumし、その後K=1のscalar積を行うため、
authorityのReduceSum→Mulと全 `2^30` maskでbitwise同値です。quarantine copyは
証拠保存用で、active合計には1回だけ計上しています。

task396のcost1017/cost961 quarantineもgenerator到達可能な反例を4 ORT構成で
再現済みです。1017はuint8減算のunderflow、961はTopK行を3に減らしたことで
決定的な4行目を落とします。どちらも `REJECTED_DO_NOT_MERGE` です。

このループは`submission.zip`、`submission_base_8009.46.zip`、スコア台帳を変更していません。
ただしtask012昇格直前の監査で、root `submission.zip` と`all_scores.csv`が別の
同時処理によりauthorityから既に分岐していることを検出しました。ユーザー/外部状態として
上書きせず保持し、比較authorityは不変の`submission_base_8009.46.zip`に固定しています。
