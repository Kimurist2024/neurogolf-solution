# 担当範囲と証拠の対応表

README に記載した担当範囲について、**主張 → 証拠ファイル → コミット → 確認方法**を
一対一で対応付ける。この 1 ファイルで、どの成果がどのファイルから確認できるかが辿れることを目的とする。

由来と権利関係の区分は [THIRD_PARTY.md](../THIRD_PARTY.md)、
数値の出典は [docs/metrics/official-results.md](metrics/official-results.md) を参照。

## 0. 前提：本人の関与の性質

本コンペはチームで参加し、候補となる ONNX の実装にはチーム内で共有されたものや
生成 AI が出力したものも含まれる。したがって本人の関与は
**「全ての候補を単独で書いたこと」ではなく、次の 5 つ**である。

1. 何を最適化対象とするかの方針決定
2. 採用してよい候補の条件（検証ゲート）の設計
3. 候補群を同一条件で採点し、採否を判断すること
4. 提出後の実測値とローカル見込み値の差から原因を特定すること
5. 採用候補を安全に統合し、最終提出物を構築すること

以下、それぞれについて証拠を示す。

## 1. 対応表

| # | 担当した判断 | 証拠ファイル | 該当箇所 | コミット | 確認方法 |
|---|---|---|---|---|---|
| 1 | **最適化対象の決定**：学習ベースを破棄し、生成器仕様を計算手順へ書き下す方針へ転換 | [proposals/001](../proposals/001-zero-risk-onnx-cost-reduction.md) §1、[docs/research/discussion-methods-recent.md](research/discussion-methods-recent.md) | 採点式の分解、上位帯との差の測定 | `e3cb120` | 提案書の「スコアラ分析」節に、コスト内訳の特定と方針の根拠が記載されている |
| 2 | **コスト支配項の特定**：中間テンソルが支配項であり、`[1,10,30,30]` float32 が 36,000 バイトを占めると確定 | [proposals/001](../proposals/001-zero-risk-onnx-cost-reduction.md) §1、README §3.2 | `calculate_memory` の挙動分析 | `e3cb120` | 公式スコアラの読解結果と、そこから導いた削減方針の対応 |
| 3 | **削減手法の体系化**：出力の無料化・チャンネル早期集約・入力上書き・dtype 縮小の 4 型に整理 | [docs/golf/ONNX_GOLF_PLAYBOOK.md](golf/ONNX_GOLF_PLAYBOOK.md) | 全文（119 行） | `e3cb120` | 実行環境における型と演算の対応表を含む。試す前に候補を絞るための資料 |
| 4 | **採用ゲートの設計**：4 段すべてを通過した候補のみ採用、他の採用経路を設けない | [docs/golf/AGENT_LOOP_TASK_RULES.md](golf/AGENT_LOOP_TASK_RULES.md) §6、[scripts/golf/try_candidate.py](../scripts/golf/try_candidate.py)（446 行） | 判定条件と、それを機械化した実装 | `e3cb120` | 規約と実装が対応していること。`try_candidate.py` が唯一の昇格経路 |
| 5 | **採点処理の移植**：公式スコアラを挙動を変えずに移植し、ローカルで同一条件の採点を可能にした | [scripts/lib/scoring.py](../scripts/lib/scoring.py)（631 行） | `calculate_memory` / `calculate_params` / 検証部 | `e3cb120` | 公式 `neurogolf_utils` と同値であることが採用ゲート② の前提 |
| 6 | **禁止構造の確定**：公式の禁止に加え、経験的に本番で失敗する構造を列挙 | [docs/golf/BANNED_STRUCTURES.md](golf/BANNED_STRUCTURES.md) | 全文（45 行） | `e3cb120` | 各項目に、そう判断した実測の根拠が併記されている |
| 7 | **候補の同一条件評価と採否判断** | [scripts/golf/harvest_others_safe.py](../scripts/golf/harvest_others_safe.py)、[all_scores.csv](../all_scores.csv) | 候補プールの走査・採点・勝者選定 | `eaeb3ca` | 214,922 件の候補から、採点結果のみを根拠に採用を決めた実装 |
| 8 | **検証スループットの最適化**：採点時間を実測して律速を特定し、時間制限を採用条件に加えた | [docs/golf/task_scoring_times.csv](golf/task_scoring_times.csv)、README §3.6 | 400 タスクの採点時間実測と階層化 | `456e59c` | コストではなく検証時間を測った判断。分布の偏り（中央値 0.20 秒、上位 2 件がタイムアウト）が根拠 |
| 9 | **失敗の系統分離**：本番で得点しない事象を 3 系統に分け、系統ごとに別の対策を割り当てた | [docs/golf/ERROR_PATTERNS.md](golf/ERROR_PATTERNS.md) | A/B/C の分類と各対策 | `e3cb120` 以降更新 | 各系統に実測の事例（タスク番号・点数）が紐づいている |
| 10 | **乖離の原因特定手法の確立**：見込みと実測の差から原因タスクを逆算する診断法 | [docs/golf/ERROR_PATTERNS.md](golf/ERROR_PATTERNS.md)「ギャップ診断法」、[docs/golf/private_zero_tasks.md](golf/private_zero_tasks.md) | 差＝失った点数の関係と、実測での適用例 | `54371e9` `e1b6a9b` | 差 20.51 → 該当タスクを個別提出 0 回で特定、等の実例が記録されている |
| 11 | **統合と最終提出物の構築** | [scripts/merge_external.py](../scripts/merge_external.py)（708 行）、[scripts/golf/global_merge.py](../scripts/golf/global_merge.py)、[submission.zip](../submission.zip) | 候補源からタスク単位で最小コストの合格候補を選び統合 | `988202b` | 提出物の各 ONNX が、どの候補源から採用されたかの判断が実装に現れている |

## 2. スコアの推移と、その時点での判断

コミット履歴に残っている到達点と、そこで何を判断したかの対応。

| 日付 | コミット | 到達スコア | その時点の判断 |
|---|---|---|---|
| 2026-06-19 | `e3cb120` | — | 基盤を公開。採点処理の移植、削減手法の整理、禁止構造の確定 |
| 2026-06-28 | `f8ed7a8` `c4ead66` | 7769.17 | 全タスク一括の書き換えが +3 点程度で頭打ちと判定し、タスク単位の作り直しへ移行 |
| 2026-07-02 | `54371e9` | 7837.18 | 本番で 0 点になる事象を系統分離。診断手順を整備 |
| 2026-07-11 | `e1b6a9b` | 7964.59 | 候補の統合方式を確立し、採点結果のみを根拠にした採否判断へ |
| 2026-07-16 | `988202b` | 8025.43 | 最終提出物を確定して公開 |
| 2026-07-16 | `eaeb3ca` | — | 比較・検証に用いた候補プール（214,922 件）を証跡として追加 |
| 2026-08-02 | `7f5fce0` | **8025.82** | 公式最終スコアを記録 |

## 3. 学士相当能力との対応

| 能力 | 本コンペでの該当箇所 | 証拠 |
|---|---|---|
| アルゴリズムの理解と実装 | タスクの変換規則を ONNX の計算グラフへ翻訳 | [docs/golf/ONNX_GOLF_PLAYBOOK.md](golf/ONNX_GOLF_PLAYBOOK.md) §6 |
| データ構造・計算量の把握 | 中間テンソルの形状・型・メモリ量からコスト支配項を特定 | README §3.2、[proposals/001](../proposals/001-zero-risk-onnx-cost-reduction.md) |
| 数理的な理解 | `score = max(1, 25 − ln(cost))` から、桁単位の削減が必要と結論 | README §1、[docs/metrics/official-results.md](metrics/official-results.md) |
| 実験計画 | 新規生成データでの一致検査、正答率の閾値、時間制限の設計 | [docs/golf/AGENT_LOOP_TASK_RULES.md](golf/AGENT_LOOP_TASK_RULES.md) §6 |
| 定量評価 | 400 タスクのコスト・採点時間の実測と分布分析 | [all_scores.csv](../all_scores.csv)、[docs/golf/task_scoring_times.csv](golf/task_scoring_times.csv) |
| 仮説検証・原因分析 | 見込みと実測の差から原因タスクを逆算し、個別提出を消費せずに特定 | [docs/golf/ERROR_PATTERNS.md](golf/ERROR_PATTERNS.md) |
| ソフトウェア設計 | 提案・実装・レビューの分離、単一の採用経路 | [docs/orchestration.md](orchestration.md)、[scripts/golf/try_candidate.py](../scripts/golf/try_candidate.py) |
| 品質保証 | 禁止構造の確定、退行時の差し戻し手順、時間制限 | [docs/golf/BANNED_STRUCTURES.md](golf/BANNED_STRUCTURES.md)、[docs/golf/ERROR_PATTERNS.md](golf/ERROR_PATTERNS.md) |
| 技術文書の作成 | 提案書、手法集、失敗パターン集、本対応表 | [proposals/](../proposals/)、`docs/` |
| チーム開発 | 候補の受領・評価・統合と、役割境界の明示 | [THIRD_PARTY.md](../THIRD_PARTY.md)、[docs/orchestration.md](orchestration.md) |
| 研究倫理 | チーム・生成 AI・第三者の寄与と本人の寄与の区別を明示 | README「チーム成果と木村竜輝の担当範囲」、[THIRD_PARTY.md](../THIRD_PARTY.md) |

## 4. 本資料で示していないこと

誤解を避けるため、本資料が主張していない範囲を明示する。

- **400 タスク全ての ONNX を本人が書いたとは主張していない。** 候補には
  チーム共有・生成 AI 由来のものが含まれる（[THIRD_PARTY.md](../THIRD_PARTY.md)）
- **`others/` の 214,922 ファイルは本人の作成物ではない。** 本人の関与は、
  これらを同一条件で採点し採否を判断したことである
- **個々のファイルの実装者を行単位では特定していない。** 実装の一部は
  生成 AI に委譲しており、その分業は [docs/orchestration.md](orchestration.md) に記載している
