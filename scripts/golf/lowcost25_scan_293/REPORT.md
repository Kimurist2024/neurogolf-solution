# Low-cost safe reduction scan (latest authority 8010.03)

対象は score=25 の既存4件を除く cost<=10 の17件。最新 authority
`submission_base_8010.03.zip` を再プロファイルし、Identity/Transpose/Einsum 等の
25点課題と同型の192候補に加え、単純空間変換（legacy Pad、Slice+Pad、Upsample+Slice）
を確認した。

結論は **NO_SAFE_COST_IMPROVEMENT**。既知例・4 ORT設定・strict checker・静的形状・
runtime trace を満たす strict-lower 候補は0件だった。

- task053: Pad単体は cost 0 だが、背景チャネルの1-hotを失い known 0/60。
- task135: Slice+Pad は正答するが cost 360（現行2）で退行。
- task326: Slice+Pad は正答するが cost 160（現行4）で退行。
- task307: Upsample+Slice は正答するが cost 144000（現行4）で退行。
- 残り13件: 25点モデル同型テンプレート／initializer除去を全て known 不一致で棄却。

fresh検証に進める候補、採用候補、root/stageへの変更はない。詳細な全候補・ハッシュ・
runtime計測は [evidence.json](evidence.json) に保存した。
