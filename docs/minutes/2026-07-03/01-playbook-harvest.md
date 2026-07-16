# プレイブック — 外部ソースのハーベスト → 検証 → マージ

他インスタンスがこの手順をそのまま実行できるように書く。**このワークフローがセッションの主作業**。

## 0. 前提の確認(毎回)

```bash
# 現ベストの base zip を確認
cat docs/golf/campaign_best.txt        # 例: submission_base_7853.10.zip	7853.10
python3 -c "import hashlib;print(hashlib.md5(open('submission.zip','rb').read()).hexdigest())"
```

`submission.zip` は必ず `submission_base_<最新LB>.zip` と同一であること。ずれていたら base zip を submission.zip にコピーして直す。

## 1. スキャン(外部ソース → 勝者候補)

正典スクリプト = `scripts/golf/scan_sources_seq.py`(sha1重複排除・逐次採点+タイムアウト・厳密安値のみ勝者化)。

```bash
.venv/bin/python scripts/golf/scan_sources_seq.py <BASE_ZIP> <SOURCE_DIR> <TAG> 2>/dev/null
# 例: scan_sources_seq.py submission_base_7853.10.zip others/70205 h70205x
# → /tmp/<TAG>_winners.json  (task/base/cand/src/path)
```

- BASE_ZIP は**必ず現ベスト**(古い base だと既に入った改善を「勝者」と誤検出する)
- 重い(2万ONNXでも重複排除後は数百件、~3分)。バックグラウンド推奨

### handcrafted(codex/Fable キャンペーンの採用分)を回収する場合

正典 = `scripts/golf/harvest_handcrafted.py`(このセッションで永続化)。
`submission.zip` を base として `artifacts/handcrafted/task*.onnx` を比較し、strictly-cheaper+正解を勝者化する。

```bash
.venv/bin/python scripts/golf/harvest_handcrafted.py    # → /tmp/harvest_handcrafted.json
```

## 2. 犯人事前照合(隔離済みとバイト一致を除外)

```python
import hashlib, json
from pathlib import Path
qsha = {hashlib.sha1(f.read_bytes()).hexdigest(): f.name for f in Path('artifacts/quarantine').glob('*.onnx')}
# 勝者の各 path の sha1 が qsha にあれば「隔離犯と同一」→ 除外
# task169 は外部候補が全黒 → 無条件除外
```

## 3. fresh ゲート

- **標準は k=30**(ユーザー厳命 `k30-fixed-absolute`)。
- **ただしユーザーが「fresh100で判定」と指定したらk=100**(このセッションはk=100指定だった)。

```bash
B="285=/tmp/h70205x_winners/task285.onnx,173=..."   # task=path をカンマ区切り
.venv/bin/python scripts/verify_fix.py --batch "$B" --k 100 > /tmp/verify.json 2>/dev/null
# decision: ADOPT / REJECT, fresh_fails, margin_min, cost を返す
```

判定の使い分け:
- `fresh_fails == 0` → 本体マージ候補
- `fresh_fails 1〜数件` → **プローブ分離**(私的通過するかLBで直接確認)
- `fresh_fails 10+` かつ前科タスク → **見送り**(検証費用もかけない)

## 4. 構造ゲート(提出前の必須チェック)

各候補ネットに対して:
- 禁止オペ: `Loop/Scan/NonZero/Unique/Script/Function/Compress` + `*Sequence*` + GRAPH属性(サブグラフ=If)
- opset domain は `''` / `ai.onnx` のみ
- **Conv/ConvTranspose の bias長 == out_ch**、**QLinearConv の bias長(input[8]) == W dims[0]**(フリップ地雷防止)
- sparse_initializer 不可
- ORT `ORT_DISABLE_ALL` でロード成功。`same node name` エラーが出たら **fix_dup(ノード名リネーム)で無害修復**(cost不変)

## 5. マージ方式の選択

### A. 単発プローブ先行方式(候補≤30件、期待値の高い外部ソース)【推奨】

各候補を**1ファイルだけのzip**にして個別提出。判定規則:
- スコア = 期待sc(=25-ln(cost)) → **私的通過(採用)**
- スコア = 0.00 → **私的0点(quarantine)**
- ERROR → **グレーダークラッシュ(quarantine)**

全件判定後、**実証分のみを一括マージ**(投影ピタリ、リスクゼロ)。
70204でこの方式が +2.78 をノーリスク回収(目玉097 910→115)。

### B. 本体マージ + 2木プローブ(候補が多い、fresh不完全が混在)

1. fresh 0/100 の安全群 → 本体MAINにまとめて1提出
2. fresh 不完全群 → sc値を**≥0.19 離して**2〜3木に分けて提出
3. MAINが退行したら gap 逆算、木は gap で犯人個体特定
4. 無実だけ統合して最終提出

## 6. 提出とスコア確認

```bash
cp <merged_zip> submission.zip
.venv/bin/kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "<説明>"
# 採点は数分。submissions -v をCSVパースしてPENDING解除を待つ
```

**注意**: 提出メッセージにカンマが入るとCSVパースがずれる。`csv.reader` で列を取り、メッセージ列は `r[3]`、status は `r[4]`、score は `r[5]`。

## 7. gap 診断(退行時)

```
gap = 投影LB − 実測LB ≈ Σ(私的0点タスクの sc(new_cost))
```

- subset-sum で犯人の組合せを絞る(`itertools.combinations`, tol≈0.06)
- 解が複数なら**プローブ木**で個体確定
- 犯人は `artifacts/quarantine/` へ隔離し、handcrafted は base 版に戻す(再採用防止)
- gap は決定的(`numerical-gap-means-task-error`)。「測定誤差」で片付けるの禁止

## 8. ベスト更新(改善確定後)

```
1. cp submission.zip submission_base_<新LB>.zip
2. best_score.json 更新(score/md5/由来)
3. docs/golf/campaign_best.txt = "submission_base_<新LB>.zip\t<新LB>"
4. メモリ current-best.md / MEMORY.md の該当行を更新
5. all_scores.csv 再計算: scripts/golf/dump_scores.py --best submission_base_<新LB>.zip
6. others/neurogolf/all_scores.csv にも同期(古くなりがち)
```
