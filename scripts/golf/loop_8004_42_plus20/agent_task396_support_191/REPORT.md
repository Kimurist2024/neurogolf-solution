# task396 full-support audit — 71407 cost1017 / cost961

## Outcome

**71407のtask396候補2件は両方REJECTです。** 既知例では4構成すべて
266/266ですが、公開generator `task_fcb5c309.py` の明示parameterから再現できる
反例を各候補に確認しました。root `submission.zip`、`all_scores.csv`、71407の
task396 quarantine 2件は変更していません。

| model | SHA-256 | memory + params = cost | full/strict | known ×4 | decision |
|---|---|---:|---|---:|---|
| authority | `ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94` | 924 + 95 = **1019** | pass | 266/266 | 現LB authorityのまま |
| cost1017 | `83f7ef034709949a7d743fd8909944c11fd7b65b2cf097874b8a2957abe1d6bf` | 922 + 95 = **1017** | pass | 266/266 | **REJECT** |
| cost961 | `1806e29dbd9f6cf7e21b2bb7dcf49f02ea1f613d2f12cfbfdad04f256ab99073` | 872 + 89 = **961** | pass | 266/266 | **REJECT** |
| authority-equivalent repair | `6cdd850fafd26068e49bb828171492bec767849b2c3676946f8e17e03f879045` | 923 + 95 = **1018** | pass | 266/266 | generator-unsoundのためREJECT |

全モデルでstandard domain、full checker、strict shape inference + data propagation、
static shapes、finite initializer、Conv bias UBなしを確認しました。既知266件と下記
2反例の全runでruntime error **0**、raw nonfinite **0**です。従って失敗はORT差や
NaNではなく、graphの決定論的な規則誤りです。

## Counterexample 1 — cost1017 uint8 underflow

seed `92000396` のvalid case 23を、次の公開generator parameterだけで完全再現
しました。

```text
width=12, height=13
brows=[2,3], bcols=[3,9]
wides=[4,3], talls=[6,4], colors=[8,6]
rows=[1,3,4,4,4,5,5,6,7,7,8,10,11,11]
cols=[9,5,0,4,10,5,10,5,7,8,2,4,2,6]
```

この入力では `s8=maxw=0` です。authorityは
`Gather(s8,bestj1)->wu; Max(wu,1)->wu_safe` で1へclampします。一方cost1017は
この2 nodeを削除して `Sub(maxw,1)` へ直結したため、uint8の `0-1` が **255** に
underflowします。

| ORT mode | authority mismatch | cost1017 mismatch | cost961 mismatch | error / nonfinite |
|---|---:|---:|---:|---:|
| disable-all, threads=1 | 16 | **48** | 16 | 0 / 0 |
| disable-all, threads=4 | 16 | **48** | 16 | 0 / 0 |
| default, threads=1 | 16 | **48** | 16 | 0 / 0 |
| default, threads=4 | 16 | **48** | 16 | 0 / 0 |

authority自体もこのgenerator caseでは16セル誤りますが、cost1017はauthorityとraw
非同一で、underflowにより48セルへ悪化します。したがってcost1017のfull-support
correctnessは明示反例で否定されます。

## Counterexample 2 — cost961 drops the decisive fourth row

seed `94000396` のvalid case 93を、次のparameterで完全再現しました。

```text
width=14, height=12
brows=[4,5], bcols=[2,10]
wides=[4,3], talls=[5,3], colors=[2,5]
rows=[0,1,5,6,6,6,7,7,9]
cols=[11,5,4,3,8,11,3,4,1]
```

authorityの4-row TopKは `ri=[5,7,6,4]`, `s8=[0,0,0,1]` となり、決定的な4番目
のrow `ri=4` を選び正解します。cost961は `krow=3` に削減され、
`ri=[5,7,6]`, `s8=[0,0,0]` しか見ません。その結果 `r0=5`, `powshift=0` を選び、
内部maskが飽和して4セルを誤ります。追加されたstart-row correction branchはこの
caseでは `shift_cond=False` であり、原因ではありません。

| ORT mode | authority mismatch | cost1017 mismatch | cost961 mismatch | error / nonfinite |
|---|---:|---:|---:|---:|
| disable-all, threads=1 | 0 | 0 | **4** | 0 / 0 |
| disable-all, threads=4 | 0 | 0 | **4** | 0 / 0 |
| default, threads=1 | 0 | 0 | **4** | 0 / 0 |
| default, threads=4 | 0 | 0 | **4** | 0 / 0 |

これはauthorityが正解するgenerator到達入力でのcandidate-only failureなので、
cost961は明確に採用不能です。

## Repair disposition

cost1017に `Max(maxw,1)` 1 byteを戻すとcost **1018**になります。このrepairは
既知266件×4構成でauthorityとraw bit-identical（different 0、error 0）で、上の
2反例でもauthorityとraw bit-identicalです。ただしCounterexample 1ではauthority
自身が誤るため、これは**authority-equivalent repairであってtrue-rule repairでは
ありません**。候補としては隔離・REJECTしました。

既存のgenerator-derived corner parser
`f1bddd36f0c0b943fe84d500bb629159b3639997bf7ea4b2e39eb2aa2bc9da2b`
はfresh 5000/5000×2 ORTのsound controlですが、cost **1245**（authorityより+226）
です。今回、authority 1019未満のtrue-rule exact repairは得られませんでした。

## Integrity and evidence

- `submission.zip`: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv`: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- task396 quarantine 2件のSHAは上表から不変。
- 監査中の71407 `MANIFEST.json` 変更
  `6f22cc20... -> b57f95fd...` はrootによるtask192 POLICY90 stageで、task396 payload
  には変更なし。このlaneはMANIFESTを書いていません。
- Machine-readable evidence: `audit/result.json`
- Explicit generator witnesses: `counterexamples/*.json`
- Reproducible audit: `audit_task396.py`

Final decision: **do not merge either task396 model from 71407.**
