# NeuroGolf 提出エラー / 失敗パターン総まとめ

最終更新: 2026-06-16。マージ・提出で踏んだ「無効化 / 退行 / 乖離」パターンを全部ここに集約する。
**マージ採用・提出前に必ず参照すること。** 関連メモリ: only-merge-fresh-verified / local-grader-divergence /
sparse-initializer-banned / ort-unsupported-op-grader-zero / private-set-overfit / aggressive-no-fresh-gate-result /
sub18-grader-failure。

---

## A. グレーダー REJECT(`Error processing onnx networks for tasks: [...]`)= 提出全体が ERROR・無効

そのタスクのネットを公式グレーダーが処理できず、**提出 zip 丸ごと SubmissionStatus.ERROR**(スコア付かず)。
1タスクでも該当すると全体が落ちる。**ローカルが通っても落ちることがある**のが厄介。

| # | 原因 | 実例 | 検知 / 回避 |
|---|---|---|---|
| A1 | **ローカル official_gold は通るがグレーダーは拒否**(op/attr/dtype の組合せがグレーダーの ORT で未対応) | **task177**(2026-06-16, o_plus6_372 由来。ops=ArgMax/ReduceMax/Gather/Pad/Conv/Equal/Greater、sparse=0、banned op 無し。lib+official gold+margin+fresh 0/30 を全通過したのに ERROR) | **完全な事前検知は不可**。ERROR が出たら該当タスクを base に戻して即再提出。`verify_fix.official_gold` はグレーダーと**完全一致しない** |
| A2 | **sparse_initializer 入り** ONNX | task223(sparse_initializer → `Error processing onnx networks`) | sparse 不使用。提出前に sparse_initializer 数を確認 |
| A3 | **ORT DISABLE_ALL 未実装オペ**(セッション生成でクラッシュ=0点)。**オペ名ではなく opset 依存** | **TopK opset11** は ORT DISABLE_ALL で `NOT_IMPLEMENTED`(task173_candidate_k11=ローカルでも落ちる→grader でも ERROR)。一方 **TopK opset18 は grader が受理**(2026-06-17 task285 で実測 COMPLETE、ERROR なし)。禁止 op 公式リスト外でも opset 次第で落ちる | **判定は「ローカル ORT 1.24 DISABLE_ALL でロード/採点できるか」。ローカルで `score_and_verify` が通る op は grader も通る公算が高い**。ローカルで NOT_IMPLEMENTED が出る op/opset だけ除外すればよい(TopK 一律除外は過剰) |
| A4 | **過去に公式エラーを起こした既知タスク** | task054 / task158(worker_prompt にも警告記載) | これらのネット差し替えは特に保守的に。乖離履歴と重複 |
| A5 | **sub18 系 zip** | sub18 系はローカル合格でもグレーダー0点(-102.6 事故) | sub18 系統は単体実測なしで採用禁止 |

> **重要教訓(2026-06-16)**: ローカルの `lib_gold + official_gold + margin + fresh(k30)` 4 ゲート全通過でも
> **A1(task177)でグレーダー ERROR**。つまり fresh-gate は「private 乖離」を防ぐが「グレーダー処理拒否」は防ぎきれない。
> → **ERROR 通知が出たら、報告タスクを base へ revert して即再提出**するのが唯一確実なリカバリ。

---

## B. private 乖離(ローカル fresh-gate を通るが非公開ベンチで低得点/0)= 提出は COMPLETE だが projection を下回る

提出は成立(スコア付く)が、ローカル projection より**実測が大幅に低い**。Kaggle は最良提出を保持するので
**LB 退行はしない**が、その提出は「無視される」=労力の無駄になる。

| # | 原因 | 実例 | 回避 |
|---|---|---|---|
| B1 | **monster cut(高コスト帯の大幅削減)** | 2026-06-16: roi zip の monster cut t101/219/23/76(base>30k)を含む提出が「proj 6844.5」→ private 失敗 → **18 件の monster cut を落として salvage → 6828.12**。`merge_finalize.py --max-base-cost 30000` で除外 | **base cost > 30000 の削減は除外**(divergence guard)。fresh-gate だけでは防げない |
| B2 | **roi / aggressive 手法**の削減全般(mid-band でも) | roi zip(submission6824_plus16_roi…)。**【実証 2026-06-16】mid-band roi カット t79/90/268/319/396 を含む提出 B' が 6830 → 6754.71(-75!)で総崩れ。5 タスクが private でほぼ 0 点**。monster だけでなく mid-band でも roi は乖離する | **roi/aggressive 由来は丸ごと不採用が安全**。どうしても試すなら安全版A→roi版B の 2 段 A/B 提出(Kaggle最良保持で下振れ無害)。上記 5 タスクの roi ネットは再採用禁止 |
| B3 | **例データ過学習**ネット | task096: visible に fit → 非公開で 0 点(-15.4) | 仕様コンパイルのみ。visible-gold/margin/自己申告だけで採用しない |
| B4 | **local↔grader 既知乖離タスク** | 191 / 219 / 264 / 230 / 282 / 317、task220(長時間ランの一過性) | これらは特に慎重。fresh-gate 通過でも単独 A/B 推奨 |
| B5 | **fresh-gate を端折った採用** | task023(旧チェック全通過も fresh 22/400 失敗)。task192@910 で「fresh 失敗→LB0点」実証 | `verify_fix` の fresh-gate(k≥30)を必須に。visible gold だけは不可 |
| B6 | **zero-input コスト測定バグ** | task188: projected≠actual の正体。zero 入力でコスト測定すると誤る | コストは実入力 `score_and_verify` で測る |

---

## C. fresh-gate は両方向に誤る(REJECT を鵜呑みにしない / ⚠️重要訂正 2026-06-17)

fresh-gate(`verify_fix.verify_one`, k=30)の REJECT は**「過学習の確定診断」ではなくリスク信号**。generator は実ベンチに無いエッジケースまで生成するため、**実機(LB)では正しいネットを fresh 1-2/30 失敗で誤って弾く(false-reject)**ことがある。

**実証(2026-06-17):** fresh-gate が REJECT した3つを個別 LB 提出した結果、**3つとも LB が上がった**:
- task118(fresh 2/30)→ **+0.62**、task233(fresh 2/30)→ **+0.24**、task18(fresh 1/30)→ **+0.13**。combined で **6834.70→6835.70(+1.00)**。

一方で fresh-fail が**本当に**LB-乖離を意味する true-reject もある: task192@910、task023、**task285(fresh 1/30 → LB 6844.55→6830.41 = -14.14、2026-06-17 実測)**。**fresh-gate だけでは true/false を区別できない**。だから fresh-reject は**必ず単体で隔離 LB テスト**(本番に混ぜない)。Kaggle 最良保持で true-reject(-14)も無害に弾ける。これまでの実績: false-reject 8件(118/233/18/277/23/370/64/+α)+ true-reject 1件(285)。

→ **唯一の真の判定器は LB**。Kaggle は最良提出を保持するので、**fresh-fail が低率(≤~10%)の候補は捨てずに LB A/B(best に載せて提出)で実測判定**する([[reattempt-and-low-fail-abtest]] の方針を実証)。fresh-gate は「ADOPT は概ね安全」側に使い、「REJECT は LB で再確認」する非対称運用が正しい。fresh-fail が高率(例 task209 の 10/500 ≒ 2%? 要再評価、task204)のものは別途精査。
- なお fresh-gate は **A1 型(グレーダー処理拒否)と B1/B2 型(monster/roi 乖離)には無力**な点も変わらず注意。

---

## D. 運用 / ツールのエラー

| # | 事象 | 対処 |
|---|---|---|
| D1 | **kaggle CLI submit がハング**(プロセスが完了せず stuck、複数溜まる) | `pkill -f "kaggle competitions"` → 再 submit。`kaggle competitions submissions`(一覧)も時々ハングするので連発しない。macOS に `timeout` は無い |
| D2 | **ORT Conv バッファバグ** | ORT 1.24 + DISABLE_ALL のループ推論時のみ Conv 出力が壊れる。乖離タスクの一部の正体の可能性 |
| D3 | **ローカル scorer が cost=0 / None**(測定アーティファクト。**「縮退/0点」と誤判定しないこと**) | task179(transpose)/ task241 はローカル `score_and_verify` が測定不能(=0 表示)。**【実証 2026-06-16】グレーダーは普通に計測して得点する**。anchor179_241 の cost=1 ネットは fresh-gate を 0/30 通過(lib+official gold+margin 全 OK)= 正真正銘の最小正解で score 25。**ローカル cost=0 だけを根拠に過学習扱いして捨てるのは誤り**。ローカル比較不能タスクは fresh-gate(通れば採用)+ LB A/B で判定する |
| D4 | **handcrafted dir のゴミ** | `task213.onnx.trace.onnx`(ORT プロファイル出力)が `task*.onnx` glob に混入。スキャン時は実ネットだけ対象に |

---

## 安全マージ手順 = 3 提出プロトコル(提出回数を浪費しない確定フロー)

**個別 reject を 1 本ずつ LB テストするのは提出の浪費。最大 3 提出で済む:**

1. `merge_targeted.py --base <現ベスト> --k 30 --out merge.zip` を実行 →
   **安全版 `merge.zip`(ADOPT=fresh-pass のみ)** と **リスキー版 `merge_risky.zip`(ADOPT + fresh-reject 全部)** を同時生成。
   - grader-killer(TopK 未対応 opset / sparse / load 不能)は `score_bytes` が None を返すため**どちらにも入らない=計算で事前除外**。
2. **提出① 安全版**:必ず改善する保証付き。COMPLETE を確認して新ベストにアーカイブ。
3. **提出② リスキー版**:fresh-reject を一括投入。
   - **COMPLETE かつ安全版より高い** → これが新ベスト(全 reject が false-reject だった)。
   - **`Error processing onnx networks for tasks: [X]`** → 上述 A1。**メッセージが X を名指しするので二分探索不要**。risky から task X を base へ revert = **提出③ 修正版**。
   - **COMPLETE だが安全版より低い**(B2 true-reject が混入、例 task173 -14.5)→ 何もしない。**安全版が既にベスト**として残る(Kaggle 最良保持)。
4. 真の判定器は LB。fresh-gate REJECT は false/true が混在(C 章)するが、上記でまとめて捌ける。monster/roi もリスキー版に入れて LB に判定させる(`--max-base-cost` は使わない方針へ移行)。

> 旧: 個別 reject テスト×N 本 → 新: 安全/リスキー/(修正)で最大 3 本。grader-ERROR は計算(エラーメッセージ)で犯人特定できるのが鍵。

### ⚡ ギャップ診断法(束で退行したとき、二分探索なしで乖離タスクを1発特定)

**fresh-pass(0/30)でも private 乖離する net は稀にある**(2026-06-17, task29 が SAFE 束を -10 退行させた実証)。束が退行したら**提出を増やさず計算で犯人を出す**:

1. 束の **局所推定スコア**を計算 = `Σ score(local_cost)`(各変更タスク) + 不変タスク。例: 6857.77。
2. **ギャップ = 局所推定 − 実測LB**。例: 6857.77 − 6842.22 = **15.55**。
3. 乖離タスクは grader で **得点 0** になり、推定していた **new_score を丸ごと失う**。よって **ギャップ ≈ 乖離タスクの new_score**(複数なら合計=subset-sum、通常 1〜2)。
4. 変更タスクを **|new_score − ギャップ| 昇順**で並べる → 最小のものが犯人。例: task29 new_score 15.542、|Δ|=0.008 で一致 → 犯人 task29。
5. その task を base へ revert → 再提出(**修正版**)。例: 6842.22 → **6857.74 回復**。
6. **大物カット(score がギャップと不一致)は自動的にシロ**。名前/ソースで切らない(危険)。high-k 監査も不要。

> 前提: 局所 cost ≈ grader cost(anchor 採用後は local sum ≈ LB でほぼ成立)。ギャップに ±0.3 程度の不確かさがあるので、僅差の上位 1〜2 候補を revert すれば 1 提出で確実に修正できる。
