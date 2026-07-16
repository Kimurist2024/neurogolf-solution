# Low-cost second pass (8011.05 authority)

Decision: **NO_SAFE_STRICT_LOWER_FOUND**

既存25点の4件を除外し、cost<=10の残り17件を最新authorityに対して再探索した。
ReverseSequence、legacy属性だけのSlice/Pad、全Transpose、ConvTransposeの
initializer crop、Einsum scalar/factor prune、および全ローカル履歴archiveを対象にした。

- generated attempts (pre-dedup): 7381
- historical locations: 29867
- historical unique task/hash pairs: 125
- evaluated unique candidates: 7489
- safe strict-lower tasks: 0
- targets changed from 8010.03: []

既知完全一致からfresh 2x2000へ進める、strict-lowerかつfail-closedな候補はなかった。
root submission / all_scores / stage は変更していない。

## Notable rejected lead

task223 の sparse initializer 版（dense 5要素 → stored 2要素）は full checker、
strict shape、ORT既知265件×4設定を通過した。しかし公式 scorer の
`calculate_memory` が `SparseTensorProto.name`（存在しないfield）を参照して
`AttributeError: name` になるため、スコア不能のエラータスクとして棄却した。
再現結果: `sparse_task223_audit.json`
