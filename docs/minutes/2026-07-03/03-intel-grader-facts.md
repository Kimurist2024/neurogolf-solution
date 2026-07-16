# グレーダー確定事実・スコアラトリック・地雷カタログ

出典: このセッションで Discussion/主催者発言/ORT issue を精読して確定(メモリ `grader-facts-canonical`
`kaggloop-scorer-tricks` `local-ort-contamination` `sakana-true-rules` に永続化済み)。

## スコアリング(全て)

```
score(task) = max(1, 25 - ln(cost)),  cost = params + memory_bytes
```
- 正しさは**全か無か**(train+test+arc-gen + 非公開private全セル完全一致で初めて得点)
- **params = initializer の要素数**(バイトではない → INT4パッキングは無意味)
- **memory = 各中間テンソルの max(静的shape bytes, ORT実測bytes) の総和**。`input`/`output` 名は無料
- ln で効くのでオーダー削減が正義。数%削減で止めない

## グレーダー環境(主催者確定)

- **ORT 1.24.4 固定**(v1.25+ で導入されたオペは失敗)+ `ORT_DISABLE_ALL`
- onnx==1.21.0 / numpy==2.4.4 / **シングルプロセス・zip順採点**
- サーバは定数畳み込みしない → 残した定数サブグラフは全額課金(自前でinitializerに畳め)
- 審査順: (1)load+1.44MB → (2)禁止オペ → (3)実グリッドprobe → (4)スコアラビリティ検査。
  ランタイムERRORはスコア検査より先=1ファイルで全体0点
- 非公開 = 公式 ARC-GEN 生成器(github.com/google/arc-gen)の別シード + 極小 holdout

## 禁止集合(公式表より広い・実測)

`Loop / Scan / NonZero / Unique / Script / Function` + **`Compress`** + **全 `*Sequence*`** +
**`If`(GRAPH属性)** + カスタムドメイン + 入出力2個以上 + 動的次元 + テンソル名に `kernel_time`。
**TopK は禁止されていない**(uint8 TopK のみ Kaggle ORT が拒否)。

## サーバ検証済みトリック(cost削減の武器)

| トリック | 効果 |
|---|---|
| **initializer再利用は1回計上** | 同一initializerを複数入力に配線→params1回。rank分解Einsum(971→300) |
| **静的value_infoでデータ依存Slice/Pad合法化** | ArgMax由来動的startsのcropを無料inputから直接(7985→4171) |
| **終端GridSample fp16グリッド** | gather+mask+zero-padを出力1ノードに融合、grid fp16(10770→5606) |
| **ConvInteger u8レンダラ** | u8+x_zero_point=1で符号付き重み相当をu8コスト、i32出力>0判定(1472→688) |
| **opset≥14でuint8算術** | Add/Mul等がuint8受理→f32に戻さずchain維持でmemory1/4 |
| **CenterCropPad連鎖** | diff-1で毎段pad_before=0のshift無し→cloak基底<29 |

## 地雷カタログ(このタスクは要注意)

### フリップ地雷(解決済)
- **task169** = QLinearConv の b2 bias長1 < out_ch2 の UB。ヒープ汚染で±18.43ランダム化。
  **b2ゼロ埋め[-51,0]で恒久修正**(7842.17ベース以降フリップ消滅)。cost 711→712。
  → **bias検査はConv/ConvTranspose/QLinearConv全て対象にすること**
- ORT 1.24.4 は同一プロセス連続採点で Conv 出力汚染(KleidiAI, issue#28654)。
  `dump_scores.py` の単一プロセス400連続はfalse-accept/reject温床。再現資材 `artifacts/ort_bug/`

### 私的0点常習(外部候補は無条件プローブ or 見送り)
- **task169**: 外部候補が全黒 → **無条件見送り**
- **task077**(4犯)/**task101**(3犯)/**task365**(3犯)/**task086**(3犯)/**task145**(2犯)
  → 外部候補は fresh 全通過でも**単発プローブ必須**
- task157: グレーダークラッシュ地雷(私的専用の実行時例外)。ただし床1201の健全版は単発プローブで採用実証済

### グレーダーERROR(提出全体0化)
- task157cand(1362)/ task324 / task079 giant / task054(node名重複でload失敗) など
- crashtest 全H×Wは誤検出多数(base同一crashはsafe)。try_candidate昇格ゲートにしない

## 真ルール正典(再構築・fresh検証のオラクル)

`inputs/sakana-gcg-2025/raw/task001.py〜task400.py` = 同一400タスクの検証済みPython正解(400/400全ペア一致)。
- unsound現職の再構築仕様書、fresh検証のオラクル、「単純ルール×重いONNX」の機械抽出に使える
- task番号↔ARC原ID対応 = `scoreboard_comparison.csv`

## +100計画の在庫(どこから稼ぐか)

- 総params 376k / **総memory 210k(cost の36%)** → mem半減で理論 +115.7、mem→1/3(床除外77件)で +64.2
- **memshave が本命**(私的0点リスクが構造的に低い=出力を変えない変換)。メモリ `memshave-campaign` 参照
