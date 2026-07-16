# task398 / task324 selector absorption audit (lane 176)

## Decision

**NO APPLY.** task398には全入力exactかつstrictly lowerな候補がなく、task324の
2-param削減候補は代数的にはexactだが、authorityから継承したshape cloak、
default ORT起動失敗、大量の非有限値によりfail-closedで棄却した。

Authorityは`submission_base_8009.46.zip` SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
に固定。root/submission/others/score ledgerは変更していない。

## task398: Q4 selector

baselineはmemory 144 + params 202 = **346**。

final Einsumで`Q4=[1,-1,-1]`はpositions 7, 20, 33, 46, 52, 58, 59に現れる。

- positions 7/20/33/46は、それぞれQ0/Q1/Q2/Q3の「2回目」のindex
  `g/s/B/J`だけに掛かる。
- position 52はKの`N` occurrenceだけに掛かる。
- positions 58/59は同じ`R` indexで積になるため、`Q4[R]^2=1`として無条件に
  除去できる。

Q0..Q3とKはselectorなしの1回目とselectorありの2回目に共有される。QkとKへ
Q4をglobalに掛ける補償では、1回目に必要なQ4指数0と2回目に必要な指数1を
同時に満たせない。

| candidate | cost | correctness | result |
|---|---:|---:|---|
| `q4_global` | **343** | false | probe 0でraw 300セル・threshold 264セル差、max delta `1.0586921893888e13`。棄却 |
| `q4_k_clone` | **367** | true | `K_Q4=Q4*K`を選択側K positions 9/22/35/48/54だけに使用。合成dual ORT raw一致だが+21 params |

K clone 24 paramsを追加してQ4 3 paramsを消すため、最小exact案でも346→367。
したがってtask398はdeep gate対象なし。

## task324: onehot_values selector

baselineはmemory 227 + params 212 = **439**。`onehot_values=[0,1]`はnode10に
4回、node11に5回、node21に2回使われる。

### Shared initializer absorption

`base0/refdiff/signpow`をglobalにmaskしてselectorを消す案は437になるが、
それぞれにselector外usageがあるため誤答した。公式correct=false、disabled
probe 0でraw 454セル・threshold 228セル差。

usageを分離したexact clone案は`base0_onehot[1,2,1,2]`、
`refdiff_onehot[2,2]`、`signpow_onehot[2,3]`の14 paramsを追加する。selector
2 paramsを消しても439→451となり棄却。

### 無追加parameterのexact selector synthesis

既存initializerだけで次の恒等式が成立する。

```text
e[a] = Σ(u,v,w,x) seedsel[a,u] * bgsel[v,w]
                    * Emap[x,u] * Emap[x,w]
     = [0,1]
```

node10/11の各selectorをこのcontractionへ置換した。node21は利用可能なEinsum
labelが6個しかないため、2個のselectorの積を次の形で共有した。

```text
e[A]e[h] = e[A] δ[A,h]
δ[A,h]   = Σd refdiff[d,A] * refdiff[d,h]
```

有限tensorを直接全要素検査し、single identityは`[0,1]`、paired identityは
`[[0,0],[0,1]]`とbit一致。新node、新initializerは0、元の他initializerと
node 0..9/12..20はbyte同一で、変更nodeは10/11/21だけ。これは入力値に依存
しないtensor恒等式なので全入力で代数等価である。

実測ではmemory 227 + params 210 = **437**、公式correct=true、理論gain
`ln(439/437)=+0.004566218`。

## Required deep gates for the 437 candidate

### Known4

- DISABLE_ALL threads1/4: authority/candidateとも266/266 gold正解、raw一致
  266/266、threshold一致266/266、runtime error 0。
- ただしcandidate output内の非有限値が各configで95,932個あり、両configを棄却。
- default threads1/4: authorityとcandidateの両方がsession生成時に
  `TopK: Axis has less than the requested k elements`で失敗。

したがってknown4は0/4 pass。

### Fresh10000

`task_d07ae81c.generate()`を2 seed×5,000有効ケースで検証した。generatorの
非有界retryにはケース単位0.25秒timeoutを設定し、各seed 1件のtimeoutを記録
した上で5,000件ずつ生成完了した。

- authority gold: 10,000/10,000
- candidate gold: 10,000/10,000
- authority/candidate raw一致: 10,000/10,000
- threshold一致: 10,000/10,000
- runtime error: 0
- candidate非有限値: seed1 1,814,613 + seed2 1,821,540 = **3,636,153**

正答maskは完全一致するがnonfinite gateは失敗。

### Truthful shape / UB / lookup

runtime traceは6個のshape mismatchを検出した。

- `__f16hid`: declared `[1,1,1,1]`, actual `[1,10,30,30]`
- `graph_input_cast_0`: declared `[1,1,1,1]`, actual `[1,10,30,30]`
- `counts`: declared `[1]`, actual `[10]`
- `active_oh_hid`: declared `[1,1]`, actual `[10,4]`
- `active_oh`: declared `[1,1]`, actual `[10,4]`
- `output`: declared `[1,1,30,30]`, actual `[1,10,30,30]`

さらに最初のtraceだけで392個の非有限値を検出。lookup op、sparse initializer、
external data、nested graph、禁止opはないが、shape/nonfinite UBのため総合gateは
失敗する。

shapeを正直に直す場合、少なくとも`__f16hid`のf32 36,000 bytesと
`graph_input_cast_0`のf16 18,000 bytesが計上される。現在227のmemory floorを
5万bytes以上押し上げるため、truthful repair後に439未満となる余地はない。

## Final outcome

- task398: exact案367 > 346、安価案343は誤答。
- task324: algebraic exact案437 < 439だが、known4、truthful、nonfinite、UBを失敗。
- 採用、merge、root変更: **0件**。

完全なchecker/profiler/ORT probeは`screen_results.json`、known4/fresh10000/
runtime trace/formal proofは`deep_audit_324.json`に保存した。

## Integrity

- root `submission.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- task398 source SHA-256:
  `339e0b25b3f45862f51b98c239755e597aca040b951acec5065b380a753d2513`
- task324 source SHA-256:
  `894be3a6ae4d93ce52a5bb0ec8b03fe3443de74b746cff7529dc4989cd73ac08`

