# Lane 132 — task054 / task204 / task208 exact-SOUND shave

## 結論

**昇格可能な候補は0件、projected gainは0.0です。**

authority は score **8009.46** の root `submission.zip`、SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
です。開始時・終了時で同一で、root ZIP / `all_scores.csv` / `others/` は
このlaneから変更していません。

| task | member SHA-256 prefix | authority cost | memory + params | decision |
|---:|---|---:|---:|---|
| 054 | `783e18d6e3ec` | 2258 | 2008 + 250 | keep authority |
| 204 | `312fa4435c54` | 2222 | 2076 + 146 | keep authority |
| 208 | `6c9bad970152` | 1422 | 1300 + 122 | keep authority |

32個の再生成候補は、known4実行失敗2、full/strict失敗16、未対応ORT
kernelまたはunscorable 11、strict-cheaperでないもの3に完全分類しました。
`winner_manifest.json` は意図的に空です。

## nominal lower 2件の却下

### task054 — 2258 -> 1284（偽のlower）

generic no-op判定が `fill_upd3 = Mul(__rupd, one3_u8)` を「1倍」とみなして
除去しました。しかし、このMulは値を変えない一方で `[1,4,1]` を
`[1,4,3]` にbroadcastする必須演算です。候補はfull checkerとstrict shapeを
通るものの、disable-all threads 1/4の双方で **0/266、266 runtime errors**。
`ScatterElements` のindices `[1,4,3]` とupdates `[1,4,1]` が一致しません。

したがって、これはall-input identityではなく、score低下も実行途中でprofileが
欠けた結果です。

### task204 — 2222 -> 1315（偽のlower）

固定入力契約から `Shape(input)[0] == [1]` をinitializerへ置換する数学的恒等変換を
試しました。full checker / strict data propagationは通りましたが、allocator計画が
変わり、disable-all threads 1/4の双方で **0/268、268 runtime errors**。
Sliceで `{1,19,1} != {1,20,1}` のbuffer reuse mismatchが発生しました。

authorityにないdisable-all failureを新規導入するため却下です。Shapeを残して後段だけ
int32に狭める案も、`CenterCropPad` function bodyがint64/int32を混在させるためORT load
時に拒否されました。

## task別の探索結果

### task054

- memory主成分は Concat 634、Einsum 396、ScatterElements 360 bytes。
- boolean mux
  `highR XOR (xpose AND (highR XOR highC)) == Where(xpose,highC,highR)` は8行の
  全truth-tableで証明しました。
- `2*Cast(Db)-1 == Where(Db,1,-1)` もDbの全2値で証明しました。
- ただし、このORT buildは必要なbool-output / int8-output `Where` kernelを持たず、
  両候補ともload不可です。
- 実サイズ30と一致する`CenterCropPad`の除去は値としては恒等ですが、poisoned
  `value_info`とのstrict shape conflictを露出し、採用不能でした。
- FP16定数の再結合は行っていません。既存の低cost rendererは既にfp16/uint8主体で、
  shape/broadcastを保ったstrict-cheaperな標準ORT演算への置換はありませんでした。

### task204

- 真ルールは、非重複の青い正方形輪郭の内部を、辺長が奇数ならorange、偶数ならredで
  埋めるものです。
- memory主成分は BitwiseXor 504、BitwiseAnd 360、Mul 360、Concat 240 bytes。
- 4段のint32 XOR-prefixを単一乗算へ畳む案は却下しました。合法な辺長3の境界mask
  `0x00000005`だけで、現行結果`0x03000003`に対し因数化結果`0x04fffffb`となります。
  shifted bitの重なりでXORを加算扱いできず、途中値は符号付きint32境界も越えるため、
  オーバーフローを無視した結合則は使えません。
- incremental `CenterCropPad`は奇数差のextraを末尾側へ送ることで片側crop/padを
  実現しています。直結すると同額であり、Slice/Padへ正直に置換するとstrict shapeと
  memory floorを悪化させます。
- authorityは2seed fresh **3000/3000、runtime error 0**（disable-all）でした。

### task208

- 真ルールは、最初の黒cutoutと同じ寸法・背景を持つ2番目のcutout位置を検出し、
  そこへbox色の輪郭を描くものです。
- memory主成分は19x19 bool borderのLess 361、Cast 158、Einsum 132、Slice 130 bytes。
  borderはすでに1 byte/elementで、単純な型幅削減余地はありません。
- `starts_clean:uint16`は合法入力上で残存候補が2の冪になるため、float16 Castを外した
  ArgMaxは順序上同値です。しかしschema上許可されても、このORT buildにuint16 ArgMax
  kernelがありません。同様にuint16 BitShiftとuint8 Einsumも未実装でした。
- fallback polynomialの分配則は **全256^3 uint8入力**で不一致0でしたが、上記uint8
  Einsum kernel欠如で実行不能です。
- FP16について、`x/2` と `x*0.5` はNaNを含む全65536 bit patternでNumPy・
  disable-all ORT・default ORTすべてbit差0。ただしcostは同額で定数が増えるため未生成。
  一方、`(x*0.66015625)/2`を`x*(0.66015625/2)`へ再結合すると全域で1286反例があり、
  却下しました。
- authority自体は2seed fresh **2952/3000、runtime error 0**（disable-all）であり、
  SOUND exact replacementではありません。過去のprivate-zero/metadata shaveは再利用せず、
  strict-cheaperな全合法入力SOUND rebuildも得られませんでした。

## known4 / fresh / shape健全性

authorityのdisable-all knownは以下です。

| task | threads1 | threads4 | default ORT |
|---:|---:|---:|---|
| 054 | 266/266, error0 | 266/266, error0 | pre-existing load failure |
| 204 | 268/268, error0 | 268/268, error0 | pre-existing load failure |
| 208 | 266/266, error0 | 266/266, error0 | pre-existing load failure |

2seed fresh（各1500、disable-all）はtask054 2972/3000、task204 3000/3000、
task208 2952/3000で、全task runtime error 0、nonfinite 0、near-positive 0です。
default ORTは3 authorityすべて既存のshape declaration conflictでsession作成不能でした。
新候補がこれを改善せず、さらにdisable-all failureまで増やす場合はfail-closedで却下しました。

runtime trace上の既存declared/actual mismatch数はtask054=40、task204=53、task208=9。
このlaneは新しいshape cloakを昇格させていません。

## 証拠

- `candidate_manifest.json`: 32候補の全stageとcost/構造
- `winner_manifest.json`: 空の昇格manifest
- `audit/known4.json`: authorityとnominal lowerの4設定known
- `audit/fresh_two_seed.json`: 2seed fresh
- `audit/domain_proofs.json`: float16全域、uint8全域、boolean、int32反例
- `audit/memory_anatomy.json`: 公式memoryのtensor/op別内訳
- `audit/runtime_shape.json`: declared/actual mismatch
- `audit/structural.json`: banned/nested/UBを含む構造gate
- `evidence_sha256.json`: 上記証拠のSHA-256

最終判断は **NO_PROMOTION** です。
