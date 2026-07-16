# agent_target_mid13 最終報告

## 結論

対象 `task237 / task238 / task354 / task378` について、固定基準
`submission_base_8004.50.zip`（SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`）を監査し、
真ルール由来の既存 control と新規削減案を比較した。

**安全かつ基準より厳密に安い候補は 0 件。採用候補なし、gain = 0.0。**

候補を無理に残すと、task238/354/378 では runtime shape cloak を受け入れることに
なり、今回の必須条件に反する。task237 は唯一 shape が健全だが、基準 cost 529 が
既に最安で、新規の健全な幅9 packing は cost 532、既存の独立再構築は cost 542
だった。したがって ONNX candidate は保存していない。

機械可読の全結果は [result.json](result.json)、基準監査の生データは
[baseline_audit.json](baseline_audit.json)、sound control は
[sound_controls_audit.json](sound_controls_audit.json) にある。

## 真ルール

| task | generator | 解読したルール | 分類 |
|---:|---|---|---|
| 237 | `task_99fa7670.py` | 各非零 seed 行を seed 列から右端まで塗り、同色を右端列で下方向へ carry。後続 seed が上書きする。 | Type B、有界伝播 |
| 238 | `task_9aec4887.py` | 4色の中空 frame と cyan sprite を正規化し、2本の対角線で分けた4三角領域の辺色で occupied cell を塗る。対角線上は cyan。 | Type D、入力依存正規化 |
| 354 | `task_ddf7fa4f.py` | row 0 の3色 light の直下にある各 gray rectangle を、対応 light 色へ水平 bounded flood で塗り替える。 | Type B、有界水平伝播 |
| 378 | `task_ec883f72.py` | solid inner body の bbox を求め、各 corner の3セル外側から外向き対角 ray を body 色で描く。真の正方形 grid で clip。 | Type D/B、bbox + 有界 ray |

raw の短い正解関数、ARC generator、visible pair、過去の fresh log を相互に照合した。
例の座標・色・個数を lookup する規則は採用していない。

## 基準モデル監査

実測 cost は zero input のみの rank ではなく、gold を通す official-like profiler の値を
採用した。task238 は zero input だと 559 だが、実例では 562 になるため、比較基準は
**562** である。

| task | baseline SHA-256 | memory + params = cost | known disable-all | known default | strict+data_prop | static=runtime shape | 判定 |
|---:|---|---:|---:|---:|---|---|---|
| 237 | `d8ef7011…58ff4` | 413 + 116 = **529** | 266/266, err 0 | 266/266, err 0 | PASS | PASS、全22 tensor trace一致 | 健全 incumbent |
| 238 | `02489424…23cc` | 418 + 144 = **562** | 266/266, err 0 | session生成失敗 | PASS | **FAIL、6 tensor不一致** | cloak reject |
| 354 | `c86ec60a…efe4` | 461 + 76 = **537** | 266/266, err 0 | 266/266, err 0 | PASS | **FAIL、7 tensor不一致** | cloak reject |
| 378 | `3e66557d…416` | 468 + 57 = **525** | 267/267, err 0 | 267/267, err 0 | PASS | **FAIL、全中間 trace が buffer shape conflict** | cloak reject |

重要な shape 証拠:

- task238: `GroupNormalization(input, ...)` の `gn` は runtime
  `[1,10,30,30]` だが `[1,1,1,1]` と宣言。`route_shift` も runtime 7x7、
  `base_route8_pad` は runtime 8x8。default ORT は `CenterCropPad` の shape/axes
  不整合で session を作れない。
- task354: 2個の `GroupNormalization` 出力は双方 runtime
  `[1,10,30,30]` なのに `[1,1,1,1]` 宣言。graph output も宣言1x1に対し runtime
  `[1,10,30,30]`。
- task378: 全中間を expose すると `Mul` で `{1,10,1,1}` と
  `{1,10,30,30}` の buffer reuse conflict が発生。

4基準とも standard domain のみ、禁止 op / nested graph / giant Einsum /
TfIdfVectorizer / Hardmax / Conv-family bias UB は検出されなかった。ただし shape
一致は独立した必須条件なので、普通の gold 実行だけ通る task238/354/378 を候補の
土台にはできない。

## task237 の完全 fresh 再検証

唯一構造ゲートを通った基準 task237 を、generator の独立2 seed、各5000件、
ORT default と `ORT_DISABLE_ALL` の両方で再検証した。

| seed | disable-all | default | runtime error | shape error | min positive | max off | `(0,0.25)` cell |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,004,237 | 5000/5000 | 5000/5000 | 0 | 0 | 1.0 | 0.0 | 0 |
| 8,104,237 | 5000/5000 | 5000/5000 | 0 | 0 | 1.0 | 0.0 | 0 |

詳細は [task237_fresh_2seed.json](task237_fresh_2seed.json)。known 266/266 と合わせ、
task237 incumbent は真ルールを実装していると判断できる。

## 新規 task237 探索

既存 packing は幅9だけを別 scalar gather で判別する。ここを9列
`ConvTranspose` kernel に吸収し、`GatherElements + Cast + Add` を消す案を新規に実装した。

| attempt | SHA-256 | memory + params = cost | full/strict | 結果 |
|---|---|---:|---|---|
| dense width9 pack | `36fbbbec…bb9c` | 407 + 125 = **532** | PASS/PASS | 基準529より3悪化、reject |
| sparse width9 pack | `082c6fa8…53f4f` | 理論 407 + 116 = **523** | **FAIL/FAIL** | sparse ConvTranspose weight で checker/shape inference subprocess exit 139、reject |

sparse 版は cost だけなら6下がるが、strict/full check を通らず安全候補ではない。
失敗 ONNX は `/tmp` 内だけで作り、終了時に破棄した。根拠は
[task237_attempts.json](task237_attempts.json)。過去の `Min` 削除（fresh 3/20）、
`Shrink` shift（fresh 0/20）、両方削除（fresh 0/100）も再監査し、採用していない。

## 最寄りの shape-truthful sound control

各タスクで、lookup/cloak を使わず full checker、strict+data_prop、両 ORT known、
runtime trace を通る既存 true-rule control を実測した。

| task | control path | SHA-256 | memory + params = cost | baselineとの差 | known両mode | runtime shape |
|---:|---|---|---:|---:|---:|---|
| 237 | `lane_c23/candidates/task237_rebuild_542.onnx` | `435d2ba6…868c` | 407 + 135 = **542** | +13 | 266/266 | 一致 |
| 238 | `scratch_claude/task238/cand.onnx` | `372b3c2a…60bd` | 7209 + 473 = **7682** | +7120 | 266/266 | 一致 |
| 354 | `scratch/task354/task354.onnx` | `1e4b4314…2c46` | 6298 + 39 = **6337** | +5800 | 266/266 | 一致 |
| 378 | `lane_c23/candidates/task378_sound_k12_scaled.onnx` | `2fa9656d…cd0` | 1540 + 111 = **1651** | +1126 | 267/267 | 一致 |

control は全て banned op 0、nonstandard domain 0、Conv-family bias UB 0。
task238 control は archived fresh 10,000件超で mismatch 0、task378 reference は
fresh 5000/5000 と bounded geometry 13,712/13,712 の既存独立証拠がある。しかし全て
固定基準より高コストなので、今回の candidate gate には進めない。

## Fresh gate を打ち切った理由

採用順は「strictly cheaper actual cost → 構造/shape → known両mode → fresh 2seed」の順。

- task237 の新規 dense 案は cost 532 > 529。
- task237 の sparse 案は checker/strict 失敗。
- task238/354/378 の超安 baseline lineage は runtime shape 不一致。
- shape-truthful control は全て基準より大幅に高い。

したがって candidate として fresh 2seed を実行すべきモデルは 0 件だった。
不合格モデルへ fresh だけを大量実行しても安全性条件は回復しない。task237 incumbent
については独立2 seedの完全 fresh を実施済み。

## Private status

`docs/golf/private_zero_tasks.md` の private-zero catalog に4タスクはいずれも載っていない。
task354 は過去に isolated tail で white-probe 済みと記録されている。ただし今回 private
提出は行っておらず、これは既存 provenance の記録であって新しい private 判定ではない。

## 保護対象と出力範囲

ZIP 統合は行っていない。新規・更新ファイルはすべて
`scripts/golf/loop_8004_42_plus20/agent_target_mid13/` 内に限定した。

監査終了時の保護対象 SHA-256:

- `submission.zip`: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`
- `best_score.json`: `63b472d7cfc20b872371b72a0bfd90d25494a4aec32a0a6aac0833b899f34ebb`
- `all_scores.csv`: `05c1b9810e79ee896aabb24f15aa5cd1f72479f1d12ce8d78b550459e5ade7b1`
- `a.csv`: `16570b4c751641aaab1cd0216778e6d903d56729ee04844625f1eca4d28bf156`
- `artifacts/handcrafted` aggregate: `39f5ceb7cca52e61ad05418ff8a2b9d275026daa9a203df8a5f636ce9133d49c`

最終 verdict: **NO_SAFE_STRICTLY_CHEAPER_CANDIDATE**。
