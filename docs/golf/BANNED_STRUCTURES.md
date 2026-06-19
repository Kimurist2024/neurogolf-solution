# 使ってはいけないネットワーク構造(NeuroGolf ONNX)

提出ネットで**避けるべき構造・オペレータ**の確定リスト。1 タスクでも該当すると、そのタスク 0 点〜**提出全体 ERROR**(無効)になる。
判定の最終軸は **「ローカル `score_and_verify`(ORT 1.24 + `ORT_DISABLE_ALL`)がそのネットをロード+採点できるか」**。
詳しい失敗事例は [ERROR_PATTERNS.md](ERROR_PATTERNS.md) 参照。

---

## 1. 公式バリデータが必ず弾く(=作っても無駄)

| 区分 | 禁止 | 備考 |
|---|---|---|
| **禁止オペレータ(公式)** | `Loop` `Scan` `NonZero` `Unique` `Script` `Function` | コンペ公式の禁止リスト。バリデータが自動 REJECT |
| **経験則で追加禁止** | `Compress`、すべての `*Sequence*` 系、**ネストグラフ(サブグラフ)** | worker_prompt の蓄積。動的長・制御フロー系は全滅 |
| **動的形状** | shape inference 後に未確定な形状を残す | **全テンソル・パラメータ静的形状が必須**。可変長 = 不可 |
| **opset ドメイン** | `''` / `ai.onnx` 以外のドメイン | カスタムドメイン不可。`Equal` は opset≥11 |
| **sparse_initializer** | sparse initializer を 1 つでも含む | grader が `Error processing onnx networks`(**task223 実証**)。dense のみ |
| **ファイルサイズ** | 1 ONNX > **1.44MB** | 超過で REJECT |

## 2. ORT DISABLE_ALL 未実装オペ(**オペ名でなく opset/構成依存**)

ロード時に `NOT_IMPLEMENTED` でクラッシュ → ローカルで採点不能 = **grader でも ERROR**。

| ネット | ローカル ORT | grader | 判定 |
|---|---|---|---|
| **TopK opset11 / 該当構成 opset16**(task173_candidate_k11) | `NOT_IMPLEMENTED` | **ERROR**(2026-06-17 実測) | ✗ 使用不可 |
| **TopK opset18**(task285_K32) | ロード可・採点可 | **COMPLETE**(2026-06-17 実測) | ✓ 使用可 |

→ **TopK 等を op 名で一律禁止しない**。同じ op でも **opset/ノード構成でローカル ORT が通るか通らないかが分かれる**。ローカルで通れば grader も通る。

## 3. 構造ではなく「中身」で落ちる別系統(構造リストでは防げない)

- **A1: ローカル4ゲート(lib+official gold+margin+fresh)全通過でも grader ERROR** — task177(ops=ArgMax/ReduceMax/Gather/Pad/Conv、禁止 op なし・sparse なし)。稀だが存在。**事前検知不可** → ERROR 通知が出たら該当タスクを base へ revert して即再提出。
- **private 乖離**(構造は合法だが非公開ベンチで低得点): monster 帯の大幅削減・roi/aggressive 手法。→ 別途 [ERROR_PATTERNS.md](ERROR_PATTERNS.md) B 章。

---

## 唯一守ればよい検証則

1. マージ/採用前に必ず **`verify_fix.py`(fresh-gate, k=30)** に通す = `official.sanitize_model` + ORT 1.24 DISABLE_ALL でロード&採点 + fresh 監査。
2. **ロードできない / `NOT_IMPLEMENTED` / sanitize 失敗 → REJECT**(grader も同じく落ちる)。
3. **ロードできて採点完了 → 構造起因の grader ERROR はほぼ無い**(opset18 TopK で実証)。op 名だけを理由に捨てない(取りこぼし)。
4. バッチ検証は 1 件のクラッシュで全滅し得るので、怪しい候補は **単体 `--task/--onnx` で隔離検証**。
5. ローカル比較不能(cost=0 等、例: task179/241)は「縮退/0点」と決めつけず、fresh-gate が通れば採用 + LB A/B(grader は普通に計測する)。
6. それでも稀に A1 で ERROR が出る → **Kaggle は最良提出を保持するので LB は無傷**。該当タスク revert で再提出すればよい。
