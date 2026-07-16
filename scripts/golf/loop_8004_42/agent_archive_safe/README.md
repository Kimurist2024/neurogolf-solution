# Archive-safe lane (8004.42 fixed rebase)

This lane only audits historical ONNX candidates against
`submission_8004.42_fixed_rebase_meta.zip`. It does not build or merge a ZIP.

Fixed tasks that must not regress:
`13,15,20,31,68,71,79,88,105,109,132,158,174,183,189,206,221,240,243,259,300,302,344,349,358,379,398`.

Admission requires complete known correctness, fresh-generator accuracy of at
least 95% (100% preferred), zero runtime errors, strict static shape inference,
and zero Conv-family bias UB. Lookup/private-zero/shape-cloak candidates and
known-dangerous task153 artifacts are excluded.
