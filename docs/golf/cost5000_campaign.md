# cost≥5000 ゼロ再構築キャンペーン (共有ブリーフ)

8 エージェントで cost≥5000 の超高コストタスクを **生成器仕様からゼロ再構築**し、正しさを完全維持
したまま cost を大きく下げる。割り当ては `docs/golf/cost5000_assign.json` の `agentN`。

## なぜ今これか (重要な前提)

- バンドル harvest は既に実施済み。**他者の安いネットは全部 fresh ゲートで REJECT**(可視gold通過でも
  fresh 生成器で落ちる=過学習で private 0点)。詳細 `docs/golf/cost5000_harvest_result.md`。
- だから**既存の安い候補をコピーしても無駄**。生成器の規則から作れば構造的に fresh-clean になる。これが唯一の道。
- 高コスト帯は ln で効くので、現職が「全域30×30 float デコード」等の無駄構造なら、座標/ROI/テンプレート/
  ラベル方式に置換して **桁で削れる**可能性がある(例 19883→3000 で +1.9点)。

## ゴールと採否基準 (厳守)

- 目標: 担当タスクを正しさ維持で現職より**厳密に安く**作り直す。
- **採用は `python3 scripts/verify_fix.py --task <N> --onnx <PATH> --k 30` が `ADOPT` かつ `fresh_fails:0` のものだけ**。
  fresh が 1 でも失敗したら捨てる(高コストタスクで private 0点になると現職の ~16点を丸ごと失う。リスク非対称)。
- 1つでも fresh-ADOPT で安くできれば大成功。全タスク床でも正直に床と報告すればよい。

## 入出力コントラクト (ONNX)

- 入力 `[1,10,30,30]` float32 one-hot(セル値 v→ch v=1.0、グリッド外=全ch 0)。出力 `[1,10,30,30]`(正解ch=1・他0、境界外=全0)。
- 全テンソル静的形状。禁止オペ `Loop/Scan/NonZero/Unique/Script/Function/Compress`。
- **ORT 1.24.4 + `ORT_DISABLE_ALL` で実行可能必須**。疑わしいオペは単体で実行確認。ファイル<1.44MB。

## 手順 (タスクごと)

1. **規則を生成器から確定** — `inputs/arc-gen-repo/tasks/task_<hash>.py`(+`common.py`)を読み、Python で
   `reference(grid)->grid` を書く。`inputs/neurogolf-2026/task<NNN>.json` の train+test+arc-gen 全ペアで完全一致を確認。
   1ペアでも外れたら規則理解が誤り。生成器ソースが正典。
2. **現職を分解** — `unzip -p submission.zip task<NNN>.onnx > /tmp/x.onnx` でオペ/初期化子/最大中間テンソルを確認。
   どこにコストが乗っているか(全域デコード? 大きな初期化子? 中間テンソル?)を特定し、そこを安い構造で置換する設計を立てる。
3. **最小 ONNX を設計** — 全域 float デコードを避け、規則を表現する最小の静的テンソルだけ持つ。
   雛形: `scripts/golf/scratch_codex/task051/build_*.py`(one-hot入力→規則→最小テンソル)。`onnx.checker.check_model` で検証。
4. **検証 2 段**:
   - `python3 scripts/golf/try_candidate.py --task <N> --onnx <PATH>` → `PASS score: cost=...` と現職比較。
   - **`python3 scripts/verify_fix.py --task <N> --onnx <PATH> --k 30`** → `"decision":"ADOPT"`, `fresh_fails:0` のみ合格。**k は 30 固定で絶対変えない**。
5. **保存** — 合格 ONNX を `artifacts/cost5000/agent<N>/task<NNN>.onnx` へ。**submission.zip に触らない・Kaggle提出しない**(マージと提出はメイン)。

## 既知の床・注意 (無駄撃ちを避ける)

下記は過去に「構造的床」と判定済み。**無理に削らず、それでも別アルゴリズムで破れそうなら挑戦可だが、
数学的に不可能と確認したら即報告して次へ**(1タスクに張り付いてセッションを溶かさない):

- 床気味/高難度: **task054**(D4対称モザイク, f32 Conv decode床), **task133**(テンプレ転写, GatherND/ConvTranspose pin),
  **task044**(f32 decode3600+lab900, 禁止メモ化以外<8000不可), **task118/018/064/366/286/158/364/145**(vision系の
  matched-filter/bit-packing/dtype-locked 床。ただし**別ソース/別アルゴリズム再構築なら破れた実績あり**)。
- グレーダー乖離注意: **task173/191**(ローカル合格でも公式で挙動が変わる履歴)。fresh ゲート必須、margin 確認。
- de-cursed(過去の悪履歴は抹消済、LBで判定): task023/285/204/233。

## 報告フォーマット (最後に必ず・日本語)

各担当タスク 1 行:
```
task<NNN>: 現cost→新cost (削減率)  判定[ADOPT/REJECT/床/未達]  手法ひとこと
```
最後に「ADOPT で `artifacts/cost5000/agent<N>/` に保存したファイル一覧」を明記。床判定はその数学的根拠を一言添える。
