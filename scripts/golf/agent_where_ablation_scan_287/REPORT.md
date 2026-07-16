# Where branch ablation scan 287

## 結論

`PASS_CANDIDATES_FOUND`。ただし、これは exact correctness ではなく、指定された
POLICY90 の予備通過である。報告対象は task 046 の最安コスト帯から得た1候補だけで、
既存 submission への昇格・差し替えは行っていない。

| task | authority cost | candidate cost | reduction | known | fresh seed 1 | fresh seed 2 |
|---:|---:|---:|---:|---:|---:|---:|
| 046 | 627 | 626 | 1 | 250/267 = 93.633% | 1857/2000 = 92.850% | 1879/2000 = 93.950% |

- Candidate: `candidates/task046_where098_false_cost626.onnx`
- Candidate SHA-256: `0494438c3f9bc1c38db1d65a01417fd53bec2f73c8a7a5e7cfad0c2b1875222d`
- Authority task046 SHA-256: `fb649383229d5cdcb562b8c1ce52256ff344193810888b795c20ac0aa0660d77`
- Projected score gain: `0.0015961695328221347`
- Official profile: memory `396`, params `230`, actual cost `626`, score `18.560649628899903`
- Official exact checker result: `false`（POLICY90 候補なので昇格不可）

## 変換内容

Authority の node index 98、`Where` node `ost7` を false branch に固定した。

- condition: `g367_1`
- Where output: `ost7`
- selected branch: `ost7_tmp`
- selected branch / Where output の inferred signature: ともに `UINT8 [1,1,1,1]`
- consumer `sm7` の入力: `ost7` -> `ost7_tmp`
- `onnxoptimizer`: `eliminate_deadend`, `eliminate_unused_initializer`
- node count: `172 -> 171`
- initializer count: `13 -> 13`（initializer 内容差分なし）

独立再生成は保存候補と byte-identical で、SHA-256 も一致した。グラフ差分は
`ost7` の削除と `sm7` の上記 rewire だけで、追加 node はない。

## 探索範囲

Authority は `submission_base_8009.46.zip`、SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`。

実行直前に `others/71407/MANIFEST.json` が別 lane により更新され、依頼時の active 22件から
23件になっていた。この run は fail-closed で新規 task355 を含む23件すべてを除外した。
manifest の pinned SHA-256 は
`73c6861e01332a72a4502459c736d994e4cc3f55b5ec8057d87f126ad52a96db`。
private-zero / unsound catalog との union は71 task、eligible authority は329 taskだった。

| 項目 | 件数 |
|---|---:|
| authority tasks | 400 |
| eligible authority tasks | 329 |
| cost 150--500 priority authority tasks | 90 |
| eligible authorities with `Where` | 56 |
| `Where` nodes | 271 |
| exact shape/dtype branch variants | 409 |
| priority variants | 160 |
| strict-lower + structure pass | 189 |
| known 4-config pass | 24（task 023 / 046） |
| fresh-audited candidates | 10 |
| final reported tasks | 1（task 046） |

cost 150--500 を優先したが、最終通過 task046 の authority cost は627である。
task023 の8候補は known では通過したが、fresh accuracy が84.75--86.30%に留まり全件棄却した。
task046 は同じ最小 cost 626 の別候補1件が fresh で88.65--89.20%となり棄却、上記候補が通過した。
通過後の task046 同額 known-pass 14件は、同taskから1候補だけを報告するため fresh 未実施とした。

分類内訳は `REJECT_STRUCTURE=220`、`REJECT_KNOWN_FOUR=165`、
`REJECT_FRESH_FOUR=9`、`PASS_FRESH_FOUR=1`、通過後未実施 `14`。
known は200,464 case-config executions、fresh は160,000 case-config executions。

## POLICY90 と安全ゲート

Known 267 cases と fresh generator の独立2 seed（`287000046`, `287100046`）、
各2,000 unique casesを、次の4 configすべてで評価した。

- `ORT_DISABLE_ALL`, threads 1 / 4
- `ORT_ENABLE_ALL`, threads 1 / 4

各 dataset で4 configの accuracy、sign hash、raw hashは一致した。全 run で以下は0だった。

- runtime errors
- nonfinite cases / elements
- runtime output shape mismatches
- `(0, 0.25)` small-positive elements
- config間 sign mismatch cases / cells

最小正値は1、最大非正値は0。Known / fresh の全configが90%以上だった。

構造監査は full ONNX checker と strict shape inference (`data_prop=True`) を通過した。
Authority の canonical I/O name/shapeを維持し、input は `FLOAT [1,10,30,30]`、output は
authority と同じ `UINT8 [1,10,30,30]`。standard domainのみで、lookup、banned op、nested graph、
function、sparse/external/nonfinite/giant initializer、giant Einsumはない。Conv bias UB findingsは0。
runtime intermediate trace は189 tensors、declared/actual mismatch 0、shape-cloak findings 0、
single-example intermediate bytes 396だった。

## 成果物と再現

- `scan.py`: exhaustive scanner / auditor
- `evidence.json`: authority inventory、全candidate classification、known / freshのcase-level集計
- `candidates/task046_where098_false_cost626.onnx`: 唯一の報告候補

再実行:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/golf/agent_where_ablation_scan_287/scan.py
```

`root/`、`submission/`、`all_scores/`、`others/71407/` は変更していない。Kimi は使用していない。

