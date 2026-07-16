# neurogolf-solution

Kaggle コンペ [The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) の解法一式。

ARC-AGI の各タスク変換を再現する「できるだけ小さい」ニューラルネットを ONNX で作り、
`score(task) = max(1, 25 - ln(cost))`(cost = パラメータ数 + メモリフットプリント)を最大化する。

## 現ベスト

- **公開 LB スコア: 8025.43**
- 提出物: [submission.zip](submission.zip)(各タスク最大1ファイル `task001.onnx`〜`task400.onnx`)

## リポジトリ構成

| パス | 内容 |
|---|---|
| `submission.zip` | 現ベストの提出物(LB 8025.43) |
| `all_scores.csv` | タスク別 cost / score |
| `best_score.json` | 現ベストのメタ情報 |
| `scripts/` | スキャン・マージ・検証・スコアリングのスクリプト群 |
| `scripts/golf/` | ONNX ゴルフ(コスト削減・リビルド)の実装 |
| `docs/` | 手法メモ・タスク別ブリーフ・調査ノート |
| `docs/competition/` | コンペ概要・ルール |

## スコアリングの要点

- 正しさは全か無か: `train` + `test` + `arc-gen` の全ペア + 非公開ベンチで全セル完全一致して初めて得点
- cost は ln で効くため、構造の簡素化(オーダー削減)が重要
- ONNX 制約: 静的形状必須 / 禁止オペ(Loop, Scan, NonZero, Unique, Script, Function) / 1ファイル最大 1.44MB

## ライセンス

コンペ規約に従い、公開時点で OSI ライセンス供与とみなされます(入賞時は Apache 2.0 でのオープンソース化が必要)。
