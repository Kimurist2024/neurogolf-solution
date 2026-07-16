# Transpose–unary–inverse-Transpose cancellation scan 268

## 結論

authority 400件と、`others/71407` のactive root 19件を上書きした staged-best
400件を全走査した。`Transpose → 1入力1出力ノード → Transpose` というデータ
フロー自体が **0件** であり、安全な単項elementwise、逆perm、single-use、型・shape
保持の条件まで到達する候補はなかった。strict-lower候補は **0件**、winnerはnull。

authorityは `submission.zip` SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`。
staged snapshot digestは
`9aa6726514a3c19410dc2a2199ab320625c60ecb3a221e6f7ae889ef52a4f5ac`。

## Census

| collection | models | Transpose | safe unary | T→safe unary | safe unary→T | 任意1-in/1-out sandwich | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| authority | 400 | 24 | 1,817 | 1 | 0 | 0 | 0 |
| staged root | 19 | 2 | 347 | 0 | 0 | 0 | 0 |
| composite staged-best | 400 | 26 | 1,837 | 1 | 0 | 0 | 0 |

唯一のnear edgeはauthority/compositeのtask285にある `Transpose → Cast`
(node 75→78) で、Cast出力のconsumerにTransposeはない。許可op集合を外した広義
スキャンでもsandwichが0件なので、許可リストの不足による取りこぼしではない。

全sourceはONNX `checker(full_check=True)` を通過した。strict shape inferenceは
authorityで7件、staged rootで1件失敗したが、今回のゼロ判定はraw graphの接続関係
だけで確定するため影響しない。もし構造候補が存在した場合だけ、perm合成identity、
両中間値single-useかつgraph outputでないこと、最終shape一致、Transpose前後のdtype
保持を追加で要求するfail-closed実装である。

## Candidate gate / policy

構造候補が空なので、official known-4 raw exact、fresh独立2000相当、
full/strict/profile/UB0は実行していない。これは未検証候補の採用ではなく、検証対象が
存在しないための明示的skipである。candidate ledgerは空で、private-zero、lookup、
runtime shape cloak由来の候補は一切採用・昇格していない。

SOUND_REBUILD_PROMPTのゲートに従い、`submission.zip`、`others/71407`、score ledgerは
変更していない。再現・証跡は `scan.py` / `scan.json`、独立監査は `audit.py` /
`audit.json`、空候補台帳は `candidates.json` に保存した。
