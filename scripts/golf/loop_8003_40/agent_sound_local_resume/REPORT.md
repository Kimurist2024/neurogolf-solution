# SOUND local/bounded resume — NO_CANDIDATE

## 結論

`task168 / 192 / 343 / 344` を、LB 基準 `8003.40` と Wave 1 の
`task344` に対して再監査した。**採用可能な新候補は 0 件
（NO_CANDIDATE）**。基準 ZIP、Wave 1 ZIP、`LOOP_STATUS.md`、root の
score/CSV ファイルは変更しておらず、候補を ZIP にマージしていない。

採否は「真ルール reference が known と generator-fresh 5000 件で完全」
かつ「lookup/TfIdf/Hardmax/giant Einsum/shape cloak なしの健全 control が
現行より安い」という順で判定した。4件とも reference 証明は完全だが、
健全 control がコスト門を超えたため早期棄却した。

| task | 現行/比較対象 cost | lookup-free・非giant control cost | 差 | 判定 |
|---:|---:|---:|---:|---|
| 168 | 416 | 20,403 | +19,987 | REJECT_COST |
| 192 | 1,609 | 18,973 | +17,364 | REJECT_COST |
| 343 | 173 | 178 | +5 | REJECT_COST |
| 344 | baseline 197 / Wave 1 191 | 910 | +713 / +719 | REJECT_COST、Wave 1維持 |

## 真ルールと reference 証明

証拠: `reference_audit.json`。各 readable rule は raw solver と known 全件で
一致し、generator が新規生成した5000件にも完全一致した。generation error
は全 task で0件。

### task168 — Type B（有界対角レイ）

```text
output = input のコピー
各 2x2 窓について:
  非ゼロ同色3セル + ゼロ1セルなら L triomino
  missing corner から 2x2 の外向き対角方向へ進み、境界まで同色を描く
```

- known/raw/readable: `265/265`
- fresh reference: `5000/5000`
- 現行 SHA-256:
  `dcf6a0cc845c4363197195dcf72e64f89e45116eacdc08ac767cfc0076f845f4`
- 現行 cost: `416`。52-input floating `Einsum` を最終展開に使うため、
  今回の新候補方針では再利用不可。
- 健全 control SHA-256:
  `24e27e72411c8561da75b446c8d7253b7581784b14d9d8063097e06ac83fac71`
- control: known `265/265`, cost `20,403`、giant Einsum なし、Conv bias UB 0。
  真の L 検出と方向別伝播を通常演算で物質化する中間メモリが構造床となり、
  cost 416 に勝てない。

### task192 — Type A（動的色 + 厳密 cross predicate、最高リスク）

```text
A = 1..9 のうち出現回数最大の色（同数なら小さい色）
各セル (r,c):
  H = A が同じ行の c-1..c+1 に存在
  V = A が同じ列の r-1..r+1 に存在
  output[r,c] = A if input[r,c] != 0 and H and V else 0
```

- known/raw/readable: `265/265`
- fresh reference: `5000/5000`
- 現行 SHA-256:
  `e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c`
- 現行 cost: `1,609`。
- 真ルール exact control SHA-256:
  `a348c9f8751043a0414aa91722710ed22b8e9627a1af997dec0ee691bf0792fe`
- control: known `265/265`, cost `18,973`、Einsum input数は最大3、
  lookup/TfIdf/Hardmax/giant Einsumなし、Conv bias UB 0。

単一線形 Conv では「横1 + 縦1」と「同じ軸2」を一般に分離できない。
厳密な H/V の非線形中間面を物質化する control は現行を17,364上回る。
したがって、真ルール100%を必須とする今回の task192 には安い候補がない。

### task343 — Type B（有界周期展開）

```text
各行 row:
  period = 8 if row[:4] == row[4:8] or row[:4] == row[8:12] else 6
  output_row = (row[:period] を3回反復)[:15]
```

- known/raw/readable: `266/266`
- fresh reference: `5000/5000`
- 現行 SHA-256:
  `7d64c3eda1167f322d8981531e433e7195e54d48e16e29c771b52a379af17ab1`
- 現行 cost: `173`。
- exact control SHA-256:
  `b47938285ea00b04aebea8709dd448c9983f0e3c8c6284050314097af0525c1b`
- control: known `266/266`, cost `178`、giant Einsumなし、Conv bias UB 0。
  厳密な周期判定と動的 Gather の既知健全床178が現行173より5高いため棄却。

### task344 — Type A（同時局所再彩色）

```text
元入力を参照して全セルを同時更新:
  center==2 and orthogonal neighbor contains 3 -> 0
  center==3 and orthogonal neighbor contains 2 -> 8
  otherwise -> center
```

- known/raw/readable: `266/266`
- fresh reference: `5000/5000`
- baseline SHA-256:
  `d0902dc6498525c5f62f12fc02e25fe7914afbae4a583fd77b71f8f05f08019f`
  (`cost=197`, 25-input giant Einsum)
- Wave 1 SHA-256:
  `e980db3f1083e1edecd9b10ca19b7483103db0fc3b8d535a1c1e2248d57672b0`
  (`cost=191`, 24-input giant Einsum)
- 非giant direct-Conv control SHA-256:
  `b1fa63d4aaa32fceb4f08ac425c2618ee83624910540d51ff7cfd52ad744d78e`
- control: known `266/266`, cost `910`、Conv bias UB 0。

健全な direct output Conv は厳密だが、Wave 1 より719高い。既存 Wave 1
より安全かつ安いという両条件を満たさないため、新規置換はせず Wave 1 を
維持する。

## 共通構造監査

証拠: `model_audit.json`。

4つの非giant control はすべて次を通過した。

- `onnx.checker.check_model(full_check=True)`
- `infer_shapes(strict_mode=True, data_prop=True)`
- 全推論次元 static/positive
- known gold 完全一致、runtime probe は ORT default / DISABLE_ALL とも error 0
- banned op / nested graph / function / sparse initializer なし
- Conv-family bias length UB 0
- input/output は `[1,10,30,30]`

ただし全 control は検証候補になる前の必須条件「truthful cost が比較対象より
小さい」を満たさない。よって追加の候補ZIP作成・マージは行わない。

## 最終状態

- Decision: `NO_CANDIDATE`
- Adopted tasks: `[]`
- Projected gain: `+0.0`
- Protected files changed: `none`
- Submission/Wave 1 changed: `no`

