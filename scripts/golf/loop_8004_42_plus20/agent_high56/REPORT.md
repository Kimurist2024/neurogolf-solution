# agent_high56 — 追加8タスク全履歴監査

## 結論

- 対象を `task348,369,306,106,091,121,108,265` の8件へ拡張した。
- immutable baseline は `submission_base_8005.16.zip`、SHA-256 は
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`。
- archive retained 候補に加え、`scripts/` 配下の全ONNX履歴をSHA-256で
  重複除去して247モデルを actual cost で再計測した。
- 正の actual cost が baseline より小さい候補は31件、known full を
  `ORT_DISABLE_ALL` / default の両方で100%通過した候補は3件だった。
- 3件はすべて task091 の shape-cloak + `ScatterElements` モデルであり、
  runtime shape mismatch が8件または14件ある。安全ゲート不通過のため棄却した。
- **safe winner は0件**。fresh は pre-fresh 構造ゲートを通る候補がないため
  実行していない。ZIP統合・保護ファイル変更は行っていない。

## タスク別結果

| task | baseline actual | unique history | valid lower | known dual 100% | verdict |
|---:|---:|---:|---:|---:|---|
| 348 | 130 | 9 | 1 | 0 | cost 73 は known 0/265 |
| 369 | 130 | 19 | 0 | 0 | `cand_rebuild_v2` は scorer sentinel `-1` で採点不能 |
| 306 | 128 | 21 | 17 | 0 | lower 全件 known 0/265 |
| 106 | 127 | 18 | 0 | 0 | cheaper history なし |
| 091 | 126 | 130 | 12 | 3 | 3件とも shape cloak + scatter |
| 121 | 125 | 20 | 1 | 0 | cost 124 は known 0/266 |
| 108 | 124 | 0 | 0 | 0 | baseline 外の固有履歴なし |
| 265 | 121 | 30 | 0 | 0 | `debug_outputs` は scorer sentinel `-1` で採点不能 |

## task091 の known-dual 通過候補

| actual | runtime mismatches | candidate | rejection |
|---:|---:|---|---|
| 122 | 8 | `lane_archive_top200/task091_r08_static122.onnx` | shape cloak / ScatterElements |
| 117 | 14 | `task091/agent_terminal/pad_plain.onnx` | shape cloak / ScatterElements |
| 124 | 14 | `task091/micro_pad_crop.onnx` | shape cloak / ScatterElements |

known examples に一致しても、宣言形状と実行形状が一致しないため official actual
cost として安全に扱えない。private-zero 保証条件（truthful runtime shapes、strict
data propagation、fresh true-rule dual 100%）を満たさないので fresh へ進めなかった。

## 監査上の修正

`rank_dir.cost_of()` は sanitizer/profiler が採点不能の場合に
`(-1, -1, -1)` を返す。最初の素朴な `cost < baseline` 判定ではこれが cheaper と
見えるため、`audit_all_history.py` を `cost > 0` 必須の fail-closed 判定へ修正した。

## 成果物

- `history_lead_audit.json`: archive retained の一次監査
- `all_history_audit.json`: 247ユニークモデルの actual/known/structure 証跡
- `audit_all_history.py`: 全履歴再監査スクリプト
- `result.json`: 最終集計
- `winner_manifest.json`: safe winner 0件の明示

