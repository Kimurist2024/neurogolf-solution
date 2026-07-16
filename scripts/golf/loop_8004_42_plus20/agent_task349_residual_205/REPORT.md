# task349 residual regolf 205

## Outcome

**ACCEPT_GENERATOR_SUPPORT_EXACT**。stage済みauthority SHA
`f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2`
から、次をwinnerとします。

- path: `candidates/task349_residual_patch_final.onnx`
- SHA-256: `8ab46bc1217c80c1d15c6064ea12a502c15274e12f79d9546f3d4620b76b72a3`
- official cost: **3548 -> 3532 (-16)**
- profile: `3233 + 315 -> 3239 + 293`
- projected log-score gain: **+0.004519781705619557**

root、`others/71407`、stageには書き込んでいません。

## Cost decomposition

| variant | memory + params = cost | authority差 |
|---|---:|---:|
| authority | 3233 + 315 = **3548** | 0 |
| valid-cols affine reuse only | 3237 + 307 = **3544** | -4 |
| shift uint8 only | 3241 + 305 = **3546** | -2 |
| table rewrites combined | 3245 + 297 = **3542** | -6 |
| combined + narrow side | 3242 + 297 = **3539** | -9 |
| combined + narrow side + beam rank | 3241 + 294 = **3535** | -13 |
| final + split special h-patch | 3239 + 293 = **3532** | **-16** |

## Complete semantic proof

### 1. `valid_cols_table` elimination — generator-support exact

generator ASTは`factor = common.randint(2,6)`、`size = 5*factor`であり、support上の
sideは`{10,15,20,25,30}`です。

既存`affine_width_factor[j] = -2^(30-j)`なので、ONNX Gatherのnegative indexを使うと

```text
affine_width_factor[-side] = -2^side
BitwiseNot(-2^side)        =  2^side - 1
```

となります。5 sideすべてを列挙し、authorityの`valid_cols_table[side/5]`と完全一致を
確認しました。これにより7要素tableと`five_i32`を除去しました。

### 2. `shift_by_mod` elimination — all legal table rows exact

authorityの全11行で、`shift_by_mod[i] = 1 << hend_offset_by_mod_i8[i]`です。
radiusは0..5、shiftは1..32。candidateはradiusをuint8へCastし、uint8 BitShift後に
int32へ戻します。全11 codeを列挙して一致し、overflowはありません。

### 3. side/coords narrowing — generator-support exact

authorityの`Sqrt(area) -> Cast(int32) -> Cast(int8)`を直接`Cast(int8)`へ変更しました。
support上ではareaがsideの完全平方でsideは10..30なので結果は同じです。

negative Gather indexのみ`Neg(int8) -> Cast(int32)`とし、値域は-30..-10。
`coords4`は0..29をint32からint8へ変更し、row比較もint8化しました。全5 side x
30 coordinateを列挙し、authorityの比較結果と一致しました。

### 4. Beam end Unsqueeze elimination — generator-support exact

authorityはpositive sideに対して
`Clip(side_i8, 0, 29)`をscalarで計算し、axes `[0,1,2,3]`でUnsqueezeします。
candidateは1要素rank-4の29を使い、`Min(side_i8, max29_rank4_i8)`で同じrank-4値を
直接生成します。全5 sideで一致。4要素axes initializerとUnsqueeze outputを除去しました。

### 5. Duplicate h-patch split — all-input exact

authorityの6行は次です。

- main 4行: signature `495564` x2、`133111344` x2
- special 2行: signature `214431744` x2、indices `[19,27]`、values `[63,-63]`

special signatureをscalar条件1個に分離し、そのbooleanをhalo 2更新と既存beam 1更新で
共有しました。main+specialを再結合するとindices、signatures、valuesの全6行が
authorityと同一です。これはsample依存ではなく全入力代数同値です。

変更対象外の全nodeはbyte-identicalかつ同順序、変更対象外initializerも
byte-identical、その他model fieldもbyte-identicalです。機械証明は
`audit_final.json`の`semantic_and_mechanical_proof`にあります。

## Static verification

- full ONNX checker: pass
- strict shape inference: pass
- strict shape inference + data propagation: pass
- runtime typed trace: 123 tensors、shape mismatch 0、nonfinite 0、truthful
- standard domain opset18
- functions 0、sparse initializers 0、nested graphs 0
- Conv / ConvTranspose / QLinearConv 0、従ってshort-bias UB0
- nonfinite initializers 0
- unused initializer 0

## Runtime equivalence

### Known

known 267件をdisable-all/default x threads 1/4の4 ORT構成で実行しました。

- authority right: 267/267 x4
- candidate right: 267/267 x4
- raw-bitwise equality: 267/267 x4
- threshold equality: 267/267 x4
- runtime error / nonfinite: 0 / 0

### Independent fresh

親や前reviewと異なる2 seedを使い、各2,500件を4構成で比較しました。

| seed | cases | authority/candidate right | raw-bitwise equality x4 |
|---:|---:|---:|---:|
| 20534917 | 2500 | 2485/2500 (99.40%) | 2500/2500 |
| 20534931 | 2500 | 2487/2500 (99.48%) | 2500/2500 |

合計 **20,000/20,000 case-config comparisons**でraw-bitwise一致、threshold一致、
runtime error 0、nonfinite 0です。generator誤答はauthorityとcandidateで完全に同じで、
候補固有の退行はありません。

## Rejected searches

- `Pow(int32)`によるtable除去: ONNX schema上は合法だが、このORTにkernelがなく
  `NOT_IMPLEMENTED`。候補化せず。
- direct int8 TopK: schema上は合法だが、このORTはint8/uint8/int16/uint16 TopKを
  実装していない。現行float16 Castを維持。
- h-patch pair broadcast、zero/ones/coordsのruntime生成、beam recurrence展開は、
  initializer削減よりruntime memory増加が大きくstrict-lowerにならない。
- 残存initializerはunused 0、byte-identical alias 0。追加の安全なscalar aliasは
  costを下げませんでした。

## Evidence and integrity

- final audit: `audit_final.json`
- reproducible audit: `audit_final.py`
- reproducible candidates and profiles: `build_candidates.py`, `build.json`
- protected `submission.zip` / `submission_base_8009.46.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- protected `all_scores.csv` SHA-256:
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- authority remained SHA `f753...` throughout the audit.

Final decision: **winner is safe to stage as a generator-support exact pass-through.**
