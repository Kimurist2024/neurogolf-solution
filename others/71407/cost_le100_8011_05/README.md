# cost≤100 optimization bundle (8011.05 authority)

基準は `submission_base_8011.05.zip`（LB 8011.05、SHA256
`ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56`）です。
ルートの `submission.zip` / `all_scores.csv` にはマージしていません。

## POLICY95 採用候補

| task | cost | fresh (2 seeds) | score gain | 判定 |
|---:|---:|---:|---:|---|
| 070 | 66 → 52 | 99.00%, 98.45% | +0.238411 | POLICY95 / known private-zero risk |
| 202 | 48 → 20 | 97.40%, 96.65% | +0.875469 | POLICY95 / known private-zero lineage; 14-input non-giant |

両候補とも可視gold 100%、4つのORT構成で結果一致、実行エラー0、非有限値0、
shape mismatch 0、`(0, 0.25)` の不安定な正値0です。task202は半減目標を達成しています。

合計の理論改善は **+1.113880**、両方がLBで得点した場合の投影値は
**8012.163880** です。ただし競技スコアはタスク単位でall-or-nothingのため、
95% freshはLB得点を保証しません。この2件はユーザー指定のPOLICY95枠であり、
保証安全枠ではありません。

既知のprivate-zero履歴をそのまま再現した場合は「改善が0」ではなく両タスクの現行得点を
失うため、最悪ケースの差分は **-41.939144**（約7969.110856）です。提出用zip名にも
`NOT_LB_GUARANTEED` を付け、現行championと明確に分離しています。

## ファイル

- `POLICY95_PRIVATE_ZERO_RISK/task070.onnx`
- `POLICY95_PRIVATE_ZERO_RISK/task202.onnx`
- `submission_POLICY95_NOT_LB_GUARANTEED.zip` — 上記2件だけを置換した400件zip
- `MANIFEST.json` — SHA、差分、投影値、除外理由
- `all_scores_POLICY95_projection.csv` — 候補適用時の投影表
- `evidence/` — 2 seed × 2 optimization × 2 thread設定の監査証跡
- `search_reports/` — cost帯別探索とscore25類似探索（合計37,000件超）の報告

`submission_POLICY95_NOT_LB_GUARANTEED.zip` のSHA256は
`47ce53fc27bc95f43f0ed20bae2baff48866355d5e575964a3519cece4b73799`、
MD5は `c453e291eb52933d7cca9955dde1f3a7` です。

## 主な除外

- task070 cost50: fresh正答率は95%以上だが、fresh rawに `(0, 0.25)` の値が出るため、
  margin安定なcost52を選択。
- task322 20→19 / task372 13→12: short ConvTranspose biasによる未定義動作。
- task135 cost1試作: finiteモデルではgold/freshとも不一致。

依頼の出力先が同じ絶対パスを二重に連結した文字列だったため、貼り付け重複と判断し、
既存の `others/71407` 配下に本ディレクトリを作成しました。
