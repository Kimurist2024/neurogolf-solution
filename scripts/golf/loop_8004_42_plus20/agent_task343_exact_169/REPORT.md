# task343 exact cost-172 search (authority 8009.46)

## 結論

採用可能な strict-lower 候補はありません。authority の task343 は cost
173 ですが fresh で `4975/5000`、既知の cost-172 二案もそれぞれ
`4960/5000` と `4976/5000` で反例を再現しました。真ルール exact の
control は cost 178 のため、スコアを悪化させる候補として不採用です。

- Authority ZIP: `submission_base_8009.46.zip`
- ZIP SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Authority task343 SHA-256: `7d64c3eda1167f322d8981531e433e7195e54d48e16e29c771b52a379af17ab1`
- Required exact cost: `<=172`
- Winner: `null`
- ZIP / CSV / score ledger write: none in this lane

## 真ルール

`inputs/arc-gen-repo/tasks/task_d8c310e9.py` では、基本ブロック幅 `P` は
3 または 4 です。通常は周期 `P`、交互反転時は周期 `2P` なので、出力は
必ず入力左端から周期 6 または 8 で復元できます。縮退して両周期が正しい
ケースは classifier の don't-care として扱えます。

exact178 は以下の self-correlation だけで周期を選びます。

```text
use period 6 iff q11 + 2*z7 + 35*z11 > q4
otherwise use period 8
```

出力は選択周期で `Gather(input, arange(30) mod period)` するため、色 ID を
推定も変換もしません。lookup、private-zero、巨大入力、custom domain、nested
graph、function、sparse initializer は使用していません。

## exact178 の完全性証拠

`finite_rule_proof.py` は色の数値 ID を equality partition に正規化し、次を
全列挙しました。

- `P in {3,4}`
- 各長さ `1..3`
- 使われる非ゼロ色の同値関係を最大 3 ラベルの restricted-growth string で全列挙
- `flip in {0,1}`
- generator が許す全 visible width

全 3,144,312 parameter states で不一致 0 です。このうち通常の
`generate()` が `input == output` のため再抽選する 385,614 状態を除くと、
実 generator support は 2,758,698 状態で、ここでも不一致 0 です。

さらに実 ONNX を次の gate に通しました。

| gate | result |
|---|---:|
| known, ORT disable/default x thread 1/4 | 各 266/266 |
| fresh seed 343169101, 4 modes | 各 5000/5000 |
| fresh seed 343169102, 4 modes | 各 5000/5000 |
| checker / strict data propagation | pass / pass |
| runtime shape mismatch | 0 |
| nonfinite / positive margin below 0.25 | 0 / 0 |
| actual cost | 178 = memory 146 + params 32 |

exact178 SHA-256:
`b47938285ea00b04aebea8709dd448c9983f0e3c8c6284050314097af0525c1b`。

## cost 172 探索

既存 SOUND 探索で全 affine scalar dynamic-Conv 56,074 特徴、共有定数、
anchor relation、Cast mask の主要な 15-byte decision family は走査済みです。
今回さらに未探索だった二領域を閉じました。

1. `search_threshold_pairs.py`: 56,074 特徴それぞれについて arbitrary scalar
   initializer を許す非劣位 `Equal/NotEqual/Greater/Less` threshold を生成し、
   exact `AND/OR` pair を bitset で全走査。universal mask 13,268 種、zero-8
   mask 1,374 種、hit なし。
2. `search_z11_relation.py`: visible `<=11` を exact に示す `z11` gate と、残り
   二つの affine probe 間の `Equal/Greater/Less` を組み合わせる cost-172
   family を走査。hard-case value vector 13,504 種、hit なし。

固定 output route だけで cost 157（params 32、runtime memory 125）を消費し、
cost 172 には decision 部が 15 byte しか残りません。exact178 の decision は
4 scalar Conv と比較に 21 byte 必要であり、actual cost 178 です。

## 採用判断

`winner = null`。cost-172 二案は visible gold を通っても fresh 反例があるため、
90% gate や private-zero 許容ではなく「通過保証あり」の条件を満たしません。
exact178 は generator support に対し完全ですが strict-lower でないため、root
submission へは反映しません。
