# 再現手順

第三者が本リポジトリの内容を検証するための手順をまとめる。
検証は次の 3 段階に分かれ、**後ろへ行くほど必要なものが増える**。

| 段階 | 必要なもの | 確認できること |
|---|---|---|
| A. 数値の再計算 | Python のみ | README の統計値が元データと一致すること |
| B. 提出物の構造検査 | Python + onnx | 提出物が制約（禁止演算・静的形状・サイズ）を満たすこと |
| C. 正しさの採点 | 上記 + コンペ配布データ | 各 ONNX が実際に正答すること |

**段階 C にはコンペの配布データが必要だが、規約上リポジトリには含めていない。**
Kaggle のコンペページから取得する必要がある。

## 環境

| 項目 | 値 |
|---|---|
| OS | macOS（arm64）で開発。採点環境は Linux |
| Python | 3.12 |
| onnx | 1.21 系 |
| onnxruntime | **1.24 系**（採点環境と同じ系列である必要がある） |
| numpy | 2.4 系 |

バージョンの根拠は [proposals/001](proposals/001-zero-risk-onnx-cost-reduction.md) §5「環境」。

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "numpy>=2.4,<3" "onnx>=1.21,<1.22" "onnxruntime>=1.24,<1.25"
```

`onnxruntime` のバージョンが異なると、中間テンソルの実測形状が変わり
**コストの計測値がずれる**。README §3.8 に記載した環境差の一部はこれに起因する。

## 段階 A: 数値の再計算（配布データ不要）

README の統計値が [`all_scores.csv`](all_scores.csv) と一致することを確認する。

```bash
python3 - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open('all_scores.csv')))
cost  = sorted(int(r['cost']) for r in rows)
score = [float(r['score']) for r in rows]
print('タスク数          :', len(rows))              # 400
print('score 単純合計    : %.4f' % sum(score))       # 8025.1927
print('cost 中央値       :', statistics.median(cost))# 124.5
print('cost 最大         :', max(cost))              # 7977
print('cost <= 100 の件数:', sum(1 for c in cost if c <= 100))  # 174
PY
```

所要時間: 1 秒未満。詳細は [docs/metrics/official-results.md](docs/metrics/official-results.md)。

## 段階 B: 提出物の構造検査（配布データ不要）

[`submission.zip`](submission.zip) の各 ONNX が、コンペの構造上の制約を満たすことを確認する。
**正しく解けるかどうかは確認できない**（それは段階 C）。

```bash
.venv/bin/python - <<'PY'
import zipfile, onnx, io
BANNED = {"Loop","Scan","NonZero","Unique","Script","Function","Compress"}
LIMIT  = int(1.44 * 1024 * 1024)
bad = []
with zipfile.ZipFile('submission.zip') as z:
    names = [n for n in z.namelist() if n.endswith('.onnx')]
    for n in sorted(names):
        b = z.read(n)
        if len(b) > LIMIT:
            bad.append((n, 'size', len(b))); continue
        m = onnx.load_model_from_string(b)
        ops = {nd.op_type for nd in m.graph.node}
        if ops & BANNED:
            bad.append((n, 'banned-op', sorted(ops & BANNED))); continue
        if any(i.domain not in ('', 'ai.onnx') for i in m.opset_import):
            bad.append((n, 'domain', [i.domain for i in m.opset_import])); continue
        if m.graph.sparse_initializer:
            bad.append((n, 'sparse-initializer', len(m.graph.sparse_initializer))); continue
        for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
            d = vi.type.tensor_type.shape.dim
            if any(x.HasField('dim_param') or x.dim_value <= 0 for x in d):
                bad.append((n, 'dynamic-shape', vi.name)); break
print(f'検査対象 : {len(names)} ファイル')
print(f'違反     : {len(bad)} 件')
for b in bad[:20]: print('  ', b)
PY
```

期待される結果: **検査対象 400 ファイル / 違反 0 件**。

所要時間: 数十秒。メモリ: 1GB 未満。

## 段階 C: 正しさの採点（配布データが必要）

### 入力データの用意

コンペの配布データを `inputs/neurogolf-2026/` に配置する。
規約上、本リポジトリには含めていない。

```text
inputs/neurogolf-2026/
├── task001.json … task400.json
└── neurogolf_utils/neurogolf_utils.py
```

```bash
# kagglehub 経由で取得する場合
.venv/bin/pip install kagglehub
.venv/bin/python -c "import kagglehub; print(kagglehub.competition_download('neurogolf-2026'))"
```

タスクの JSON は `{"train": [...], "test": [...], "arc-gen": [...]}` の形で、
各要素が `{"input": grid, "output": grid}`（`grid` は 0–9 の二次元リスト）である。

### 1 タスクだけ検証する

```bash
.venv/bin/python scripts/golf/try_candidate.py --task 1 --onnx <検証したい ONNX のパス>
```

配られた全ての例で一致するか、コストがいくつか、採用の関門を通るかが出力される。
所要時間は 1 タスクあたり中央値 0.20 秒だが、重いタスクは 10 秒以上かかる
（[docs/golf/task_scoring_times.csv](docs/golf/task_scoring_times.csv)）。

### 400 タスクをまとめて採点する

```bash
.venv/bin/python scripts/golf/score_dir_full.py <ONNX を並べたディレクトリ>
```

所要時間は合計で約 500 秒（`status=ok` の合計）。
ただし 2 タスクは 60 秒を超えて中断するため、実際にはこれより長くなる。

### 提出物を生成する

```bash
.venv/bin/python scripts/optimize_submission.py --tasks all --zip
```

`artifacts/submission.zip` が生成される。

## 既知の非決定性と環境差

再現時に**一致しない可能性がある**箇所を明示する。詳細は README §3.8。

| 事象 | 影響 | 対処 |
|---|---|---|
| 実行のたびに検証結果が揺れる | 並列処理の影響で判定が逆転することがある | 検証時は並列処理を無効にする（`intra_op_num_threads=1` / `inter_op_num_threads=1`） |
| Mac と Linux で判定が逆転する | 出力が 0 付近のとき符号が反転し、6 タスクで手元と本番の結果が食い違った | 出力の値を 0〜0.25 の範囲に置かない |
| 手元でコストが測れないタスクがある | 一部タスクでコストが 0 と表示される | 壊れたファイルと誤解しないこと。本番では正常に採点される |
| zip 内の並び順で結果が変わる | 同じファイル群でも順序だけでスコアが変わる | 末尾の並び順を固定する（`best_score.json` の `order_sensitive`） |
| ローカル合計と公式スコアが一致しない | 8025.1927 と 8025.82 で 0.63 の差 | 公式スコアの根拠は Kaggle 順位表 |

## 段階 C で再現できないこと

- **非公開データでの得点は再現できない。** 採点にのみ使われるデータは配布されていないため、
  「本番で 0 点になるか」は手元では判定できない。README §3.7 の系統 A・B はこの性質に由来する
- **公式順位表のスコアは再現できない。** Kaggle 側の記録であり、リポジトリからは導けない
