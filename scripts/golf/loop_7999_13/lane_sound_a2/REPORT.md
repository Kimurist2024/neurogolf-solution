# SOUND lane A2 report — baseline 7999.13

## 結果

task009 / task077 / task118 / task173 を、過去の private-zero SHA や出力
lookup に依存せず generator の真ルールから再監査した。採用条件は visible
全件、fresh generator 5000/5000、実行例外ゼロ、標準 ONNX 構造ゲート通過、
かつ実 baseline より strict に低コスト、の全条件である。

**strict winner は 0 件。予測改善は `+0.0`、`7999.13 -> 7999.13`。**
条件を満たさない置換は一切採用せず、候補 ZIP も作成していない。

| task | baseline cost | baseline fresh | 最良の健全 control | control fresh | strict-cheaper な最良候補 | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 009 | 2619 | 5000/5000 | 2619 | 5000/5000 | cost 2072, 375/5000 | reject: incumbent が健全、安い候補は 4625 fail |
| 077 | 3364 | 4961/5000 | 47796 | 5000/5000 | cost 3345, 4807/5000 | reject: sound control は +44432 cost |
| 118 | 3914 | 4393/5000 | deterministic inverse 不存在 | 到達不能 | cost 3911, 4288/5000 | reject: generator が入力非同定 |
| 173 | 3525 | 4940/5000 | 53570 | 5000/5000 | cheaper な sound 候補なし | reject: sound control は +50045 cost |

`candidate_audit.json` に各候補の SHA-256、known/fresh 件数、cost と棄却理由を
固定した。`winner_manifest.json` の accepted は空配列である。

## generator から復元した真ルール

### task009

generator は logical bitmap の各セルを間隔 2 の line-grid に置く。physical
grid は `3*S-1`、`S=6..10`。logical bitmap 上で同色 endpoint の各 pair が
同じ行または列にあるとき、その間を同色で接続してから physical grid に戻す。
セルごとの同値な規則は、3 セル刻みの水平 left/right 色集合の共通部分と、
垂直 up/down 色集合の共通部分の和集合から色を選ぶこと。

現 incumbent は lookup ではなく、この規則を標準 ONNX 931 node で実装し、
265/265 と fresh 5000/5000 を通過する。cost は 2619。最小構成の主要内訳は
101 scalar decode、101 Cast、257 Equal、452 Where、Concat/final label と 52
params で、同じ scalar decode 系の冗長削減は既に実コストと相殺される。
発見した cost 2072 候補は known を暗記できるが fresh は 375/5000 だけで、
汎用 rule compiler ではない。よって同コスト incumbent の維持が唯一の SOUND
判断である。

### task077

入力では黄色 4 の長方形が static 背景に隠され、赤 2 の可視 border fragment
から黄色領域を復元する。現 generator では各 rectangle の全行・全列に赤が
ある。historical visible fixtures との互換には、`t=2..3`, `w=2..7` の 12 個の
固定 window を列挙し、border 行/列の赤可視性と「赤が無い隣接 2 列を許さない」
条件を使う必要がある。

この規則をそのまま実装した標準 ONNX control は 266/266、fresh 5000/5000、
cost 47796。これに対し baseline cost 3364 は bit-pack/shape-cloak の heuristic
で fresh 39 fail、cost 3345 候補は 193 fail だった。したがって厳密な 12-window
検出を維持しつつ 3364 未満にする証拠は得られない。

追加で root から依頼された value_info 省メモリ probe も検証した。

- 全 value_info を `[1,1]` 相当にした cost 3290 候補: known 実行失敗、fresh
  0/5000。Slice の buffer reuse で `{1,1} != {H,W}` runtime exception。
- `cp_10` だけを `[1,1]` にした cost 3335 候補: known 実行失敗、fresh
  0/5000。`{1,1} != {30,30}` runtime exception。

いずれも serializer 上の cost 低下であり、実行可能なモデルではない。

### task118

random gray static の上に global radius 2 または 3 の plus cross を 1–4 個置く。
cross の black cell は赤 2、gray cell は cyan 8 になり、入力生成時に cyan 8 は
gray 5 へ隠される。出力は hidden gray cross cell を 8 に戻す。

この input→output は関数ではない。cross の全セルが元から gray の redless
cross は、入力では random gray static と完全に同じ 5 になる。radius 3 の外端が
すべて gray のときは radius 2 とも区別できない。既存の constructive witness は
seed 314218、10x7、center `(7,4)`、radius 2。30000 generator draw の redless
cross を含む grid は 148 件、0.493% であり、任意の deterministic ONNX に
fresh 100% を要求する本 lane の gate は情報理論的に到達不能である。

readable observable rule は visible 267/267、fresh 4852/5000。cost 3911 の安い
候補は fresh 4288/5000、現 baseline も 4393/5000 だった。strict 100% を
満たしたという主張は不可能なので、どの候補も採用しない。

### task173

1–3 個の sprite family があり、family ごとに outer color と center color が
一意。sprite type は X / plus / horizontal / vertical。各 family には完全な
3x3 sprite が 1 個あり、partial copy は center-only または outer-only。
完全 sprite の 180 度対称 pattern から center↔outer/type mapping を動的に
求め、partial copy を同じ pattern で補完する。

この真ルールを実装した標準 ONNX control は 266/266、fresh 5000/5000、cost
53570。現 cost 3525 は custom crop/QLinearConv/TopK/ScatterND と injection
constant を含む heuristic で fresh 60 fail。確認できた他候補も cost 6038 で
653 fail、cost 8568 で 25 mismatch に加えて 1 件 out-of-bounds だった。
family ごとの動的 template detection と routing を残しつつ baseline より安い
SOUND graph は得られなかった。

## 独立検証と構造ゲート

readable reference の結果:

- task009: known 265/265、fresh 5000/5000;
- task077: known 266/266、fresh 5000/5000;
- task118: known 267/267、fresh 4852/5000（非同定性の実証）;
- task173: known 266/266、fresh 5000/5000。

retained SOUND controls task009 / 077 / 173 は repository verifier でもそれぞれ
known 全件、fresh 5000/5000、runtime error 0、margin 1.0。さらに全 control が:

- `onnx.checker.check_model(..., full_check=True)` 通過;
- strict shape inference + data propagation 通過;
- inferred dimension は全て static positive;
- `[1,10,30,30]` input/output;
- standard ONNX domain のみ;
- function / sparse initializer / nested graph / banned op / sequence op / Conv
  bias issue が全て 0。

証跡:

- `reference_audit.py`, `reference_*.json`: readable rule と known/fresh 比較;
- `sound_control_manifest.json`, `sound_controls/`: provenance 固定 control;
- `verify_sound_controls_fresh5000.log`: model known/fresh/margin 実行結果;
- `structural_audit.py`, `structural_audit.json`: checker/static/domain/UB gate;
- `candidate_audit.json`: baseline、候補、control の比較;
- `winner_manifest.json`: accepted 0、projected gain 0。

root `submission.zip`、score ledger、CSV、`artifacts/`、`handcrafted/` は変更して
いない。A2 lane 外の成果物を merge していない。
