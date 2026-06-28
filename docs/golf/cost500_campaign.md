# cost≈500 → <100 リビルド・キャンペーン (共有ブリーフ)

8 エージェントで cost≈500 帯のタスクを **ゼロから最小静的 ONNX に作り直し**、cost を可能な限り下げる
(理想は <100)。割り当ては `docs/golf/cost500_assign.json` の `agentN` を見ること。

## ゴールと採否基準 (厳守)

- 目標: 各タスクの ONNX を **正しさを完全維持したまま** より安く作り直す。理想 cost<100。
- **100 に届かなくても、現職より厳密に安く・fresh ゲート ADOPT なら勝ち**。届かない=失敗ではない。
- **採用していい候補は `verify_fix.py --k 30` が `ADOPT` を返したものだけ**。fresh fails が 1 でもあれば捨てる。
- 自己申告・可視 gold 通過だけでは絶対に採用しない (private セットで 0 点になる。過学習厳禁)。

## 入出力コントラクト (ONNX)

- 入力: `[1,10,30,30]` float32。one-hot チャンネル符号化 (セル値 v → チャンネル v が 1.0)。グリッド外 = 全チャンネル 0。
- 出力: `[1,10,30,30]`。正解セル = 該当チャンネル 1・他 0。グリッド境界外セル = 全チャンネル 0。
- **すべてのテンソル/パラメータは静的形状**。動的形状禁止。
- **禁止オペ**: `Loop` / `Scan` / `NonZero` / `Unique` / `Script` / `Function` / `Compress`。
- **ORT 1.24 + `ORT_DISABLE_ALL` で実行可能必須**。未実装オペ (例 TopK(11) 等) は本番で 0 点になる。疑わしいオペは使う前に単体で実行確認。
- ファイルは 1.44MB 未満。

## 手順 (タスクごと)

1. **規則を確定する**
   - タスクデータ: `inputs/neurogolf-2026/task<NNN>.json` に `train` / `test` / `arc-gen` の各 `{input, output}` グリッド。
   - 正規生成器: `inputs/arc-gen-repo/tasks/task_<hash>.py` (hash は assign.json と all_scores.csv にある)。これが真の変換規則。`common.py` も併読。
   - Python で `reference(grid)->grid` を書き、**train+test+arc-gen の全ペアで完全一致**することを確認。1ペアでも外れたら規則理解が誤り。生成器のソースを正典とする。

2. **最小 ONNX を設計する**
   - 全域 30×30×float のデコードテンソルを安易に持たない。座標/ROI/ラベル/テンプレート/共通色圧縮など、規則を表現する最小の静的テンソルだけ持つ。
   - 既存の現職 ONNX のオペ構成を `unzip -p submission.zip task<NNN>.onnx > /tmp/x.onnx` で覗いて参考にしてよい (ただし作り直しが目的)。
   - 既存のリビルド例: `scripts/golf/scratch_codex/task051/build_*.py` が良い雛形 (one-hot 入力→規則→最小テンソル)。
   - ビルドは `onnx.helper` 直書き、または numpy で初期化テンソルを作って組む。`import onnx; onnx.checker.check_model(m)` で検証。

3. **検証 (2 段)**
   - ops/形状/gold/cost: `python3 scripts/golf/try_candidate.py --task <N> --onnx <PATH>`
     → `PASS score: cost=...` と `COMPARE optimized: cost=...` が出る。現職 cost より低いことを確認。
   - **fresh ゲート (必須・k は 30 固定、絶対に変えない)**:
     `python3 scripts/verify_fix.py --task <N> --onnx <PATH> --k 30`
     → JSON の `"decision": "ADOPT"` かつ `fresh_fails: 0` のときだけ合格。

4. **成果物の置き場所**
   - 合格 ONNX は **`artifacts/cost500/agent<N>/task<NNN>.onnx`** に保存 (現職と同じファイル名)。
   - **submission.zip には触らない。Kaggle 提出もしない**。マージと提出はメインセッションが一括で行う。

## 注意・既知の床

- `docs/golf/briefs/task<NNN>.md` があれば必読。床が文書化されているタスクは無理に削らず、その旨報告。
- task177 はグレーダーで ERROR を出した過去あり (`docs/golf/ERROR_PATTERNS.md`)。作り直すなら特に慎重に、fresh ゲート必須。
- 「全域 f32 デコード床」は 1 個の Conv(kernel[1,10,K,K], valid) でデコード+クロップ同時化すると破れることがある (`fused-decode+crop`)。
- ORT 1.24: `ReduceMax`/`ReduceMin`/`ArgMax` は bool 不可 (uint8 を使う)。`Where` の cond は bool 必須。`BitShift` の uint16 は未実装。

## 報告フォーマット (最後に必ず)

各担当タスクについて 1 行ずつ、日本語で:

```
task<NNN>: 現cost→新cost (削減率)  判定[ADOPT/REJECT/床/未達]  手法ひとこと
```

最後に「ADOPT で `artifacts/cost500/agent<N>/` に保存したファイル一覧」を明記すること。
