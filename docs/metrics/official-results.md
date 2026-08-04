# 数値の出典と再計算手順

README に記載した数値について、出典・元データ・再計算コマンドを対応付ける。
第三者が同じ値を再現できることを目的とする。

対象コミット: `3832f25`（本表作成時点の `main`）

## 1. 外部の公式記録（本リポジトリ外が出典）

| 指標 | 値 | 出典 | 検証方法 |
|---|---|---|---|
| 最終順位 | 2,963 チーム中 10 位 | Kaggle 公式順位表 / 功績証明書 | [最終順位表](https://www.kaggle.com/competitions/neurogolf-2026/leaderboard)、[docs/assets/kaggle-certificate.png](../assets/kaggle-certificate.png) |
| 最終スコア | 8025.82 | 同上 | 同上 |
| 獲得メダル | Competition Gold Medal | 功績証明書（2026-07-16 授与） | [docs/assets/kaggle-certificate.png](../assets/kaggle-certificate.png) |
| チーム名 | Kimura@SeSDA & OverfitOracle | Kaggle 公式順位表 | 同上 |

これらは Kaggle 側の記録であり、本リポジトリの内容からは再計算できない。
リポジトリ内の記載は転記である。

## 2. リポジトリ内データから再計算できる数値

元データは [`all_scores.csv`](../../all_scores.csv)（400 行、列: `rank, task, hash, cost, score, archetype`）。
以下をリポジトリのルートで実行すると、README の記載値が再現される。

```bash
python3 - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open('all_scores.csv')))
cost  = sorted(int(r['cost']) for r in rows)
score = [float(r['score']) for r in rows]
print('タスク数          :', len(rows))
print('score 単純合計    : %.4f' % sum(score))
print('cost 中央値       :', statistics.median(cost))
print('cost 平均         : %.2f' % statistics.mean(cost))
print('cost 最大 / 最小  : %d / %d' % (max(cost), min(cost)))
print('cost <= 100 の件数:', sum(1 for c in cost if c <= 100))
print('score 25.00 到達  :', sum(1 for s in score if s >= 25.0))
PY
```

期待される出力と、README のどこに現れるかの対応は次のとおり。

| 指標 | 値 | README の記載箇所 |
|---|---|---|
| タスク数 | 400 | §1 問題設定 |
| `score` 列の単純合計 | 8025.1927 | 「スコア及び順位に関する注記」 |
| cost 中央値 | 124.5 | §2 最終結果 / §4 結果 |
| cost 平均 | 453.15 | （README では最終値未記載） |
| cost 最大 | 7,977 | §2 最終結果 / §4 結果 |
| cost 最小 | 0 | — |
| cost ≤ 100 の件数 | 174 | §2 最終結果 / §4 結果 |
| score 25.00 到達 | 4 件 | §4 結果（「複数存在する」） |

### 注意: ローカル合計と公式スコアは一致しない

`all_scores.csv` の合計 **8025.1927** は**ローカル採点環境での測定値**であり、
公式最終スコア **8025.82** とは 0.63 の差がある。原因は次のとおり。

- ローカルと採点環境で ONNX Runtime の実測テンソル形状が一致しないタスクがある（README §3.8）
- 一部タスクはローカルでコストが測定できない（同 §3.8.2）

**公式スコアの根拠には Kaggle 順位表を用いる。** `all_scores.csv` は
タスク別のコスト分布を示すための資料であり、公式スコアの導出には使わない。

## 3. 開始時点の数値

| 指標 | 値 | 出典 |
|---|---|---|
| 合計スコア | 6347.82 | [docs/research/discussion-methods-recent.md](../research/discussion-methods-recent.md)（run-012 時点の集計） |
| cost 平均 | 29,727 | 同上 |
| cost 中央値 | 15,230 | 同上 |
| cost 最大 | 109,037 | [lowest_50_scores.csv](../../lowest_50_scores.csv) の 1 行目（task367） |

開始時点の `all_scores.csv` は上書きされているため、
当時のバックアップ（`all_scores.csv.7805bak` 等）と上記文書が根拠となる。

## 4. その他の実測値

| 指標 | 値 | 元データ | 再計算 |
|---|---|---|---|
| 採点所要時間の分布 | 1 秒未満 313 / 1–3 秒 47 / 3–5 秒 15 / 5–10 秒 12 / 10 秒以上 11 / タイムアウト 2 | [docs/golf/task_scoring_times.csv](../golf/task_scoring_times.csv) | `tier` 列を集計 |
| 採点時間の中央値 | 0.20 秒 | 同上 | `status=ok` の `seconds` の中央値 |
| 候補プールのファイル数 | 214,922 | `others/` | `git ls-tree -r --name-only HEAD -- others \| wc -l` |

```bash
# 採点所要時間の分布
python3 -c "
import csv, collections, statistics
rows=list(csv.DictReader(open('docs/golf/task_scoring_times.csv')))
print(collections.Counter(r['tier'] for r in rows))
ok=[float(r['seconds']) for r in rows if r['status']=='ok']
print('中央値 %.2f 秒' % statistics.median(ok))"
```
