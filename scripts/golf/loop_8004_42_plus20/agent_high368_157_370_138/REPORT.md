# Lane 138 — task368 / task157 / task370 strict-lower exact re-golf

## 結論

**昇格候補は0件、projected gainは0.0です。**

8009.46 authorityのroot `submission.zip`を開始から終了まで読み取り専用とし、
SHA-256 `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
を維持しました。`all_scores.csv`と`others/`もこのlaneから変更していません。

| task | member SHA-256 prefix | authority cost | memory + params | known × 4 | runtime shape | decision |
|---:|---|---:|---:|---|---|---|
| 157 | `a1254f261940` | 849 | 746 + 103 | 265/265 × 4 | mismatch 0 | keep |
| 368 | `0d950f5053aa` | 521 | 481 + 40 | 265/265 × 4 | mismatch 2 | keep |
| 370 | `513c0b40056f` | 944 | 824 + 120 | disable 266/266 × 2、default load失敗 | mismatch 50 | keep |

25候補の最終分類は次の通りです。

- `REJECT_FULL_STRICT_SCHEMA_UB`: 20
- `REJECT_UNSCORABLE_OR_ORT_KERNEL`: 2
- `REJECT_NOT_STRICTLY_LOWER`: 2
- `REJECT_KNOWN4_OR_RUNTIME`: 1

full checker、strict shape inference (`data_prop=True`)、standard domain、banned op、
Conv/ConvTranspose/QLinearConv bias UB0を構造gateに含めています。既存authorityのraw出力に
bitwise同値であることも、strict-lower候補に対する独立gateとしました。

## task157

authorityは既に以前の4-byte因数移動を取り込み、103 parameter全てが使用中です。
initializer重複、unused initializer、同一node CSEは0件です。

generatorは3または4個の一意footprintを生成し、青・灰配置を非重複にします。この不変量から
4体目のblue factor抽出を
`OR -> XOR -> lowbit`から3段減算へ置換する候補を作りました。

- apparent cost: **849 -> 845** (`742 + 103`)
- full checker / strict data propagation / UB0: pass
- official-like profile: cost 845だが`correct=false`
- known: **237/265**, wrong 28、runtime error 0（4設定すべて同じ）
- authority raw: bitwise **218/265**、threshold同値237/265（4設定すべて同じ）

失敗原因はauthorityが常に`sentinel16=32768`も`bstarts_s`へ加えることです。3個の選択factorを
減算しても、4体構成では「未使用factor + sentinel」が残ります。現行lowbit列はその下位factorだけを
選びますが、減算候補は2bitを残すためraw同値ではありません。known gateでfail-closedにしました。

他の試行:

- 非重複行bitfieldの3段`BitwiseOr`を可変長`Sum`へ畳む案は、ONNX schemaが
  `tensor(uint16)` Sumを許可せずfull/strictで拒否。
- 3段補正Whereをfloat16からuint16へ型付けし末尾Castを除く案は、数値としては全uint16-domain
  exactですが、ORTに`Where(uint16)` kernelがなくload不可。
- 現行のArgMax/Gather、lowbit抽出、uint16除算・cast列には、対応ORT kernel内で中間tensorを
  減らす同値演算がありませんでした。

このtaskで新しいfixture lookup、private-zero近似、shape cloakは導入していません。

## task368

authorityは40 parameter、unused/duplicate initializer/CSEはいずれも0です。主なstatic memoryは
QLinearConv 444、Mul 32、GroupNormalization 4、CastLike 1 bytesです。

- `QLinearConv`の数値0 zero-pointをoptional inputとして省略する案は、このopset/schemaでは
  当該inputがsingle-requiredでありfull checkerが拒否。
- `CastLike(gn, zero_u8)`を`Cast(to=UINT8)`へ属性化すると値変換自体は同じですが、
  隠れていた`gn/qi=[1,10,30,30]`をshape inferenceが露出し、actual costは
  **521 -> 9520**。strict-lowerではありません。
- `value_info`のみのnormalizeは実行profileを完了できずcost `-1`。

runtime traceはauthorityの`gn`と`qi`で2 mismatch、観測中間bytesは45,476です。
この既存shape cloakを残す、または新しい宣言で隠す候補はstage対象外としました。

## task370

authorityは120 parameter全て使用中で、initializer重複とCSEは0です。公式相当costは944ですが、
strict static memoryは333 bytesに対して、全node-output traceの観測中間bytesは195,887、
declared/actual mismatchは50です。default ORTはConcatの`1 != 20` shape conflictでsessionを
作れません。

探索したexact局所変換:

- 固定canonical shapeから`Shape/Sub`列をinitializerへfold
- `pow_f_hid`、`base_pair_hid`、`selector3_b`の恒等`CenterCropPad` bypass
- `PRelu(-1,slope) -> Neg(slope)`
- strictly-negative constant base上の`PRelu -> Mul`
- downstream `ReduceL1`を利用したnegative-mask/PReluの積への畳み込み
- 9個の`CastLike`を`Cast(to=...)`へ属性化

Shape/CenterCropPad/PRelu候補は、入力channel 10、height 20..30、broadcast channel 2/3/10などの
実shapeを露出し、既存1x1宣言と衝突してfull/strict gateで停止しました。Cast属性化だけは構造pass
しましたが、actual costは **944 -> 19290** で大幅増です。shape cloakを維持するmetadata shaveは
採用していません。

## fresh gate

採用条件は「known全件 × disable/default × threads 1/4、runtime error 0、authority raw bitwise同値、
truthful runtime shape」の後に、独立2seed × 各1500以上、各ORT modeで正答率90%以上です。

今回はこの前段を通った候補が**0件**だったため、freshを実行して後段だけで救済することはせず、
fail-closedで終了しました。`winner_manifest.json`は意図的に空です。

## 証拠

- `audit/model_anatomy.json`: initializer/CSE/static memory内訳
- `audit/build_manifest.json`: 25生成候補と変換根拠
- `audit/screen_results.json`: authority auditと全候補stage
- `candidate_manifest.json`: 最終候補manifest
- `winner_manifest.json`: 空の昇格manifest
- `evidence_sha256.json`: 証跡SHA-256

最終判断は **NO_PROMOTION** です。
