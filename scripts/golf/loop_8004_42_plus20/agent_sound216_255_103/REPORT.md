# SOUND task216 / task255 rebuild audit

## Verdict

**採用可能な strict-lower exact モデルはありません。root / `others/` / submission は変更していません。**

- authority: `submission_base_8008.14.zip`, SHA-256
  `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`
- task216 authority: cost `1499`, member SHA-256
  `9a5f4f10d6e014b3f053ce1dabeb39cbeaf95964ae685aa71514fd695caf0756`
- task255 authority: cost `1307`, member SHA-256
  `5bcf1caa5a31f2800cb5b1a0dc578355701168da8081d995d475765816943030`
- 新規 projected gain: `0`

## 真ルール

### task216 (`8efcae92`)

20×20上に間隔1以上で置かれた3〜4個の青矩形があり、各矩形内の一部が赤になる。
赤セル数は全矩形で相異なり、最多赤矩形をその高さ・幅で切り出して原点へ出力する。
連結成分によるnumpy参照は既知266件と、独立2 seedの合法生成例各5000件で完全一致した。

### task255 (`a64e4611`)

ランダム背景へ動脈・静脈矩形を黒で描き、その内部を緑にしてから、入力では緑だけを黒へ隠す。
出力は隠された緑を復元する。しかし、下端でclipされたlow-right veinについて
`tall=3` と `tall=4` が**同一入力**を生成し、期待出力は15セル異なる。
したがって入力だけを受け取る決定論的ONNXに100% exact解は存在しない。

## 履歴監査

既存の全履歴rescreenをSHA単位で再利用した。

| task | unique SHA | authority未満のstatic候補 | strict survivor |
|---|---:|---:|---:|
| 216 | 58 | 24 | 0 |
| 255 | 60 | 11 | 0 |

task216のauthority未満候補はknown不完走/不一致、実コスト超過、shape/data-prop違反で全滅。
task255はそれらに加え、真ルール自体の非関数性でexact候補になり得ない。

## Authority監査

両authorityは全既知例をdefault/`ORT_DISABLE_ALL`で通すが、runtime shapeはtruthfulではない。

| task | known disable/default | runtime shape mismatch | UB | nonfinite |
|---|---:|---:|---:|---:|
| 216 | 266/266, 266/266 | 53 | 0 | 0 |
| 255 | 265/265, 265/265 | 19 | 0 | 0 |

task216では例として、`input_hid`が宣言`[1,1,1,1]`に対し実行時
`[1,7,30,30]`、可変crop `pair`が宣言`[1,2,1,1]`に対し実行時
`[1,2,h,w]`になる。task216の公開データだけでも出力cropは124種類あり、
`h=3..18`, `w=4..18`。単一の静的shapeでこのSliceをtruthfulには表せない。

## Fresh 2 seed × 5000

task216は遅い配置rejectionを避け、すべてgeneratorの合法パラメータ範囲内で6種の配置族を生成した。
同一左端に3〜4矩形を積む場合、最大合法開始`row=17` / `col=16`、2×2、ランダム合法配置を含む。
task255はnative generatorをそのまま使用した。全実行をdefault/disableの両方で検査した。

| task/model | seed | disable | default |
|---|---:|---:|---:|
| 216 authority | 2162551031 | 3331正解 / 2誤り / 1667 error | 同左 |
| 216 authority | 2162551032 | 3333正解 / 0誤り / 1667 error | 同左 |
| 216 truthful exact control | 2162551031 | 5000/5000 | 5000/5000 |
| 216 truthful exact control | 2162551032 | 5000/5000 | 5000/5000 |
| 255 authority | 2162551031 | 4758/5000 (95.16%) | 4758/5000 |
| 255 authority | 2162551032 | 4722/5000 (94.44%) | 4722/5000 |

全runでnonfinite=`0`、margin `(0,0.25)`=`0`。task216 exact controlの最小正値は`1.0`。

## task216の具体的阻害要因

最も安い既知の**truthful + exact** controlは
`scripts/golf/scratch/task216/cand3.onnx`で、cost `31511`
(`memory=31372`, `params=139`)。checker/full strict shape、standard domains、UB0、
runtime mismatch 0、known両モード全件、fresh両seed全件を通したが、1499未満ではない。

低コストbitset系は最後にデータ依存`Slice(input, starts, ends)`で
`[1,2,h,w]`を作り、`Pad`で30×30へ戻す。この中間shapeは合法入力ごとに変わるため、
static + truthful契約と両立しない。実際、cost9135の真ルール寄りQLinear rebuildにも
この1件のshape mismatchが残る。固定shape方式へ変えると選択マスク/空間fieldが必要で、
標準的なfloat入力の単一`[1,1,20,20]` decodeだけで1600 bytesとなり、1499を超える。

これは任意のONNXに対する抽象的計算量定理ではないが、現行op契約と全既知構築族に対する
具体的な構造floorである。shape cloakを再利用せずstrict-lowerを満たす経路は確認できなかった。

## 成果物

- `RESULT.json`: 最終判定の短いmachine-readable manifest
- `inventory.json`: 58/60 unique SHAの履歴監査要約
- `authority_audit.json`: known dual-ORT、構造、runtime shape
- `fresh_two_seeds.json`: authorityの2×5000 fresh
- `control_audit.json`, `fresh_truthful216.json`: truthful exact controlの証拠
- `task255_ambiguity.json`: 同一入力・異出力の有限証明

候補ONNXの昇格、root ZIP更新、`others/`への書き込みは行っていない。
