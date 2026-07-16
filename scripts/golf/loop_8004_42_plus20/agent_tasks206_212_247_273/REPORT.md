# SOUND exact-regolf: task206 / 212 / 247 / 273

## 結論

**winner: null / 採用候補0件 / 推定増分 +0.0**。

`submission_base_8009.46.zip` から4ファイルを直接抽出して再監査したが、
全入力で成り立つ dead/CSE/initializer alias/no-op は4件とも0件だった。
generator-support のみに依存する変更も、完全証明できる低コスト化は得られなかった。
したがって `candidates/` は空のままである。

| task | SHA-256 | memory | params | cost | SOUND判定 | 結論 |
|---:|---|---:|---:|---:|---|---|
| 206 | `cf0011a5073a7e99270c7abd5be6382f2dfbd370ca820ae0b9719311b498a8f9` | 126 | 68 | 194 | 不適格 | false output shape + `CenterCropPad`/`ScatterElements` |
| 212 | `e3f20fe069499de6c8ab36eadb10e69802ab61c4c87eb98d9843c7a87869ad42` | 0 | 240 | 240 | 不適格 | 48-input giant `Einsum` |
| 247 | `62a535f96689b455894f3bd34dffebf78406c2cde6bd72d6ceb1362c19f5e37c` | 165 | 47 | 212 | 適格 | 既知のspec-derived exact floor、shaveなし |
| 273 | `bd5484be154f5303a19893499fb9a33698bef35a5a2a126ec5abe6357b5509e1` | 88 | 105 | 193 | 不適格 | `TfIdfVectorizer`×8 + 53-input giant `Einsum` |

全costは repository scorer の再計測値で、依頼時のauthority costと一致した。

## generator真ルール

`inputs/sakana-gcg-2025/raw/taskNNN.py` の `p` を独立importし、
`inputs/neurogolf-2026/taskNNN.json` の train/test/arc-gen 全ペアで再実行した。
4タスクとも全件一致した。

- task206: 色5のマーカーを消し、入力内の3x3以下の非空spriteを
  マーカー中心へ複写する。
- task212: 色5の水平horizonを基準に、各色1/2の点を列内で規則の向きへ
  horizonまたは端まで伸ばす。
- task247: 各非背景objectのセル数を数え、最大セル数のobject色を
  左から右へ並べ、最大セル数の行数だけ反復する。
- task273: 各4隅を色4で示した矩形について、内部を色2で塗る。

## 検証

各baselineを次の4設定で検証した。

1. `ORT_DISABLE_ALL`, threads=1
2. `ORT_DISABLE_ALL`, threads=4
3. default ORT, threads=1
4. default ORT, threads=4

結果:

- known: task206=266、task212=265、task247=269、task273=266 の全例で、
  4設定すべて正答・runtime error 0・nonfinite 0・small positive 0・
  設定間raw bitwise一致。
- fresh: seed `71407261` / `71407262` の各100例、計200例/タスクで、
  generator出力と raw `p` が全件一致し、4設定すべて正答・runtime error 0・
  nonfinite 0・small positive 0・設定間raw bitwise一致。
- task206/212 は既存の独立監査
  `agent_new_mid22/baseline_fresh_dual500.json` でも各500 freshを通過済み。
- task247 のspec-derived系は `scripts/golf/scratch/task247/REPORT.md` で
  fresh 2000/2000 の履歴があり、今回の結果と整合する。

baselineのfresh通過だけではSOUND適格性を上書きしない。task206/212/273は
今回の禁止構造を保持するため、候補として再発行していない。

## exact shave再走査

各graphに対して次を再走査した。

- unused initializer
- 同一dtype/shape/value initializer alias
- Identity / single-input variadic / identity Transpose
- 同一node signature CSE
- 反復initializer軸（寸法縮約の候補）

全4件で unused=0、alias=0、no-op=0、CSE=0、反復軸=0。

履歴証拠とも整合する。

- task206: `agent_new_mid22/REPORT.md`。cost194はcost196系の終端。
  truthful spec control `scripts/golf/scratch/task206/cand_v8.onnx` はcost7753。
- task212: 同reportで単軸縮約123件とcoupled rank縮約6件が既に失敗。
  truthful spec control `scripts/golf/scratch/task212/candidate_v13.onnx` はcost4398。
- task247: `root_high51/REPORT.md` の全履歴scanにlower leadなし。
- task273: `root_high52/REPORT.md` の唯一のactual cost192 leadはknown 0/266。
  separator除去再コンパイルも過去に最初のknown例で失敗済み。

## 非変更保証

root artifactには触れていない。

- `submission.zip`: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `submission_base_8009.46.zip`: 同一SHA
- `all_scores.csv`: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`

機械可読証拠は `evidence/audit.json`、再実行コードは `audit.py`。
