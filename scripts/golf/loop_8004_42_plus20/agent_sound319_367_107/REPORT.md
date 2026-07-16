# SOUND再監査 — task319 / task367

## 結論

固定採用できる候補は0件。`submission_base_8008.14.zip`、root、`others/`は変更していない。

| task | 現cost | 最良truthful候補 | 結果 |
|---:|---:|---:|---|
| 319 | 1003 | 51736 | 現行はfresh 97.36–97.70%かつshape cloak。truthful化は大幅コスト増 |
| 367 | 2179 | 3913 | true-rule exact候補は全gate合格だが現行より1734高い |

## task319

真ルールはType Dの大域cross-correlationである。

1. 最頻色を背景、最頻の非背景色を2倍拡大色とする。
2. 残る各色の画素を2×2へ拡大したとき、拡大色画素とのtranslation overlapが最大になる位置を数える。
3. raw solverの個数tie-breakを適用し、選ばれた色のbounding boxを背景埋めで出力する。

可読化した`raw p`はknown 267/267だが、generator freshでは4898/5000と4918/5000だった。現行SHA `29d5bf...`もknown全267件をdefault/disable・1/4 threadsですべて通す一方、freshは両ORTモードで4885/5000、4868/5000に留まった。初期化子`corr_pattern_f16=[5,5,12]`による固定補正も含む。

さらにgeneratorへ、同じobservable inputから異なるoutputを返す2通りの有効パラメータを構成できる。入力SHAは`a41dee...`、出力SHAは`119a18...`と`8ccd98...`。latentなsprite0の識別情報が入力から消えるため、決定的ONNXで全valid-call exactは原理的に不可能である。

現行はdeclared/runtime shape mismatchが26件。metadataを実shapeへ修正した同一計算グラフはtruthful・known全件になるが、実測costは1003→51736。fresh不一致も計算意味が同じなので解消しない。dead/init/no-op/CSE/optional/fold/absorbのexact scanには追加のshaveがなかった。

## task367

真ルールは、黒0と灰5からなる2–4個の中空矩形について、灰のconnector lineを除外しながら矩形内部の黒だけを黄4へ変える処理。矩形は3..7、gridは10..20、左右端で1列だけclipされ得る。このclipとconnectorのため、単純enclosure/flood-fillは過去の大量freshで失敗済みで、corner/endpoint検出と有界下向き伝播が必要になる。

現行SHA `b2b73f...` はcost2179で、ORT_DISABLE_ALLではknown 266/266・fresh 5000/5000×2だが、default ORTはCenterCropPad shape errorでsessionを作れない。declared/runtime mismatchは65件で、graph output宣言自体も`[1,1,11,1]`対実`[1,10,30,30]`。したがってstrict gateでは不採用。

既存のgenerator-derived truthful control（cost3915）を再監査し、同値なinitializer `ax1`と`s1`を`axd`へ統合したexact shaveを作った。

- candidate SHA: `7673a580bc645f491eb85b110b142d3c6ed5dcac91df0b676c9556c6b156bdbf`
- cost: 3915→3913（memory 3540、params 373）
- known: default/disable × threads 1/4ですべて266/266
- fresh: 2seed × 5000 × default/disableの全20,000実行で正解、error 0
- truthful shape、standard domain、full checker、strict inference、UB0、nonfinite0

ただし3913は現2179より1734高く、strict-lowerではないため固定採用しない。現行のSize/Add shape chainをconstant foldする案はCenterCropPadのrank/axis cloakを露出させ、全ORTモードでload不能になった。

## 成果物

- `audit/rule_audit.json`: 真ルール、task319 collision witness
- `audit/build_manifest.json`: authority・truthful control・exact shave 7候補
- `audit/candidate_audit.json`: full checker、known×4、cost、truthful shape、UB/nonfinite監査
- `audit/fresh_two_seed.json`: 各seed 5000件のdefault/disable結果
- `candidates/task367_truthful_exact_dedupe_7673a580bc64.onnx`: soundだが非strict-lowerの最良control
- `winner_manifest.json`, `probe_manifest.json`: 空

`SOUND_REBUILD_PROMPT.md`の影響として、known一致を採用根拠にせず、generator fresh、shape truthfulness、provenanceを先に評価した。その結果、局所的なスコア改善に見えるshape cloakを候補から除外した。
