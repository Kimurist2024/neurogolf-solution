# NeuroGolf 2026 — Project Brief for Claude

Kaggle コンペ [The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) への参加リポジトリ。
ルール全文(公式ページから取得)は [docs/competition/kaggle-rules-full.md](docs/competition/kaggle-rules-full.md) にある。

## ⚠️ 言語ルール(最優先・全応答に適用)

**ユーザーへの応答・進捗報告・要約・表・結論は必ず日本語で書く。** 英語で書かない。

- サブエージェント(Agent ツール)の報告が英語で返ってきても、ユーザーに見せるときは**必ず日本語に要約・翻訳**する。英語のまま貼り付けない。
- 例外として原文ママでよいのは、ファイルパス・タスクID・コマンド・コード・ONNXオペ名など**翻訳すると壊れる固有名詞のみ**。
- 表の見出し・項目・判定(勝ち/床/退行など)も日本語にする。

## 作業体制(オーケストレーション)

**Fable が提案 → Opus が実装 → Fable がレビュー** の 2 役構造で回す。詳細は [docs/orchestration.md](docs/orchestration.md)。
提案書は `proposals/`(連番)、実装は `scripts/`、成果物は `artifacts/`。Opus は提案書に書かれた範囲だけを実装する。

## ⛳ 最重要ルール(すべての作業はここに従う)

**ARC-AGI の各タスク変換を再現する「できるだけ小さい」ニューラルネットを ONNX で作る。**

### スコアリング(これが全て)

タスクごとに、機能的に正しいネットワークのみが得点する:

```text
score(task) = max(1, 25 - ln(cost))
cost = (パラメータ総数) + (ネットワークのメモリフットプリント bytes)
```

- **正しさは全か無か**: `train` + `test` + `arc-gen` の全ペア + **非公開プライベートベンチマーク** で全セル完全一致して初めて得点。1セルでも違えばそのタスクは 0 点ではなく…正確には「得点資格なし」
- cost は ln で効くので、**パラメータ削減は桁(オーダー)で効く**。数個の削減より構造の簡素化が重要

### ONNX 制約(公式バリデータで自動チェックされる)

1. すべてのテンソル・パラメータは **静的形状(statically-defined shapes)** 必須
2. **禁止オペレータ**: `Loop` / `Scan` / `NonZero` / `Unique` / `Script` / `Function`
3. ONNX ファイル 1 個あたり **最大 1.44MB**
4. 入力は `[1, 10, 30, 30]`(one-hot チャンネル符号化、グリッド境界外は zero-hot)
5. 出力は正解セル = 該当チャンネル 1・他 0、境界外セル = 全チャンネル 0 の完全一致

### 提出形式

- `submission.zip` に各タスク最大 1 ファイル: `task001.onnx` … `task400.onnx`
- 最終提出に選べるのは **2 件**
- 1 日の提出上限: ルール文面では 5 回 / プラットフォーム設定では 100 回(API メタデータ `maxDailySubmissions=100`)。**実運用は Kaggle 画面の表示に従う**

### 締切(UTC 23:59)

| 日付 | 内容 |
|---|---|
| 2026-07-08 | エントリー締切・チームマージ締切 |
| **2026-07-15** | **最終提出締切** |

### コンプライアンス上の注意

- **複数アカウントでの提出は即失格**
- チーム外とのコードの **私的共有は禁止**(公開するなら本コンペのKaggleフォーラム/ノートブック上のみ。公開した時点で OSI ライセンス供与とみなされる)
- 外部データ・外部ツールは「全参加者が低コストで合理的にアクセス可能」なら使用可
- 入賞時は提出物とコードを **Apache 2.0 でオープンソース化**し、再現手順を提出する義務がある(学習コード・推論コード・環境記載を含む)
- 評価指標や禁止オペレータは **主催者が途中で変更・再採点する可能性あり**(アナウンスを追うこと)

## データ配置

| パス | 内容 |
|---|---|
| `inputs/neurogolf-2026/` | コンペ公式データ。`task001.json`〜`task400.json` + `neurogolf_utils/neurogolf_utils.py` |
| `~/.cache/kagglehub/competitions/neurogolf-2026/` | kagglehub 経由の同一データ(`kagglehub.competition_download('neurogolf-2026')`) |
| `inputs/neurogolf-6347-80/` | 参考カーネル `rajathrpai/neurogolf-6347-80` の出力(`base_submission/` + `overrides/`、LB 63.4780 相当) |
| `inputs/neurogolf-6347-76/` | 参考カーネル `vyankteshdwivedi/neurogolf-6347-76` の出力(task001〜400 の ONNX) |
| `notebooks/surgical-onnx-precision-parameter-reduction/` | 参考ノートブック(上記 2 カーネル出力を入力に ONNX を精密削減する手法) |
| `docs/competition/kaggle-rules-full.md` | コンペ概要・評価・ルールの全文 |

### task JSON の構造

```text
{ "train": [...], "test": [...], "arc-gen": [...] }
各ペア: { "input": grid, "output": grid }
grid: 0-9 の整数の二次元リスト(1x1〜30x30)
```

3 つのサブセット **すべて** で完全一致が必要(さらに非公開セットでも検証される)。
