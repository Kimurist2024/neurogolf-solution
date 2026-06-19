# オーケストレーション構造: Fable 提案 → Opus 実装

このリポジトリの ONNX コスト最適化は、役割を分離した 2 段構造で回す。

## 2 レーン構成(2026-06-12 ユーザー指示で導入)

最適化エージェントは目的別に 2 レーンに分ける:

| レーン | 目的 | 進め方 | 成果物 |
|---|---|---|---|
| **Lane G(全体)** | 全 400 タスクのスコアを一括で底上げ | グラフ変換パス(S1〜S7、FP16 化等)を全タスクに適用するパイプライン | `artifacts/optimized/` |
| **Lane T(個別)** | 「現状スコア − target」のギャップが大きいタスクを 1 つずつ深掘り | 1 タスク = 1 エージェント。ネットワーク構造の再設計まで踏み込む | `artifacts/handcrafted/taskNNN.onnx` |

- 優先順位付け: ギャップ(target − score)の降順。target は FP16 上限(+ln2)と外部ツールの per-task 目標値で校正する
- **合流**: 提出物は「タスクごとに、全レーンの候補から最小コストの検証合格ネットを採用」で組む
  (パイプラインの候補プールに `artifacts/handcrafted/` を候補源 D として追加する)
- Lane T の検証も Lane G と同じハーネス(出力同一性 or gold + マージン)を必ず通す

## 役割分担

| 役割 | 担当モデル | 成果物 |
|---|---|---|
| **提案 (Propose)** | Fable(メインセッション) | `proposals/NNN-*.md` — スコアラ分析に基づく最適化戦略、優先度、リスク分類、受け入れ基準 |
| **実装 (Implement)** | **Codex**(`codex` CLI / `mcp__codex__codex` / `/codex:rescue`)。Codex に難しい高難度実装のみ Opus(`Agent`, `model: "opus"`)または Fable | `scripts/` 配下のコード + 実行結果 |
| **レビュー (Review)** | Fable(メインセッション) | 実装の検証、スコア前後比較、次の提案書へのフィードバック |

> 2026-06-12 まではサイクル 1〜2 を Opus が実装(Codex クォータ切れのため)。Codex 復活後は Codex が第一実装者。

## ループ

```text
[Fable] スコアラ・ONNX・前回結果を分析
   ↓
[Fable] proposals/NNN-<title>.md を書く
   (戦略・期待効果・リスク・受け入れ基準を明記)
   ↓
[Opus]  proposals/NNN を読み scripts/ に実装
   (検証ハーネスで正しさとコスト差分を必ず計測)
   ↓
[Fable] 実装レビュー + 小規模検証 (task 1-5)
   ↓
[実行]  全 400 タスクで最適化 → artifacts/reports/ にレポート
   ↓
[Fable] 結果を読み、採用 / 差し戻し / 次の提案へ
   ↓
[提出]  ローカル期待スコアが前回ベストより改善していたら Kaggle に提出
        (1 タスクだけの改善でも提出する)
```

## Kaggle 提出ルール

- **スコアが改善するたびに提出する。改善が 1 タスクだけでも提出する**(ユーザー指示)
- 判定と提出: `scripts/submit_if_improved.py` — 最新 run レポートの合計期待スコアを
  `artifacts/best_score.json` の記録と比較し、改善していれば
  `kaggle competitions submit -c neurogolf-2026 -f artifacts/submission.zip -m "<run 概要>"` を実行して記録を更新する
- 提出メッセージには run 番号と変更内容(例: `run-003 S1-S4 + best-of-source`)を必ず入れる
- 提出後は Kaggle の公式スコアを確認し、ローカル期待値との乖離を `docs/` かメモリに記録する
  (ローカル検証と公式グレーダーは一部タスクで乖離する — メモリ `local-grader-divergence` 参照)

## ディレクトリ契約

| パス | 書く人 | 内容 |
|---|---|---|
| `proposals/` | Fable | 提案書(連番)。Opus はここに書かれた範囲だけ実装する |
| `scripts/lib/scoring.py` | Opus | 公式スコアラ(`neurogolf_utils.py`)の忠実な移植 |
| `scripts/lib/optimizations.py` | Opus | 最適化パス群(提案書の S 番号に対応) |
| `scripts/optimize_submission.py` | Opus | パイプライン CLI |
| `artifacts/optimized/` | パイプライン | 最適化済み taskNNN.onnx |
| `artifacts/reports/` | パイプライン | 前後比較レポート |
| `docs/competition/` | Fable | コンペルール等の一次情報 |

## 不変ルール

1. Opus は提案書にない最適化を勝手に追加しない(発見したら報告のみ)
2. すべての最適化は「正しさ検証 → 悪化なら元に戻す」をタスク単位で行う
3. 正しさ = train + test + arc-gen 全ペアの完全一致 + スコアラが non-None を返すこと + 1.44MB 制限
4. 公式スコアラのロジックを変えない(移植時は挙動を一切変更しない)
5. 提出物は常に `artifacts/optimized/` から生成し、入力 (`inputs/`) は読み取り専用
