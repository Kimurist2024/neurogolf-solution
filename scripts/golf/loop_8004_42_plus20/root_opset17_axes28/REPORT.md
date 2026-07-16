# 8005.16 Reduce-axes downgrade scan

**Safe adoptees: 0.**  All 400 members were checked for moving constant Reduce
axes from opset-18 inputs back to opset-17 attributes. No whole-model downgrade
remained schema-valid: affected models use opset-18+ operators such as
CenterCropPad, GroupNormalization, BitwiseAnd, or Reduce schemas incompatible
with the attribute conversion. No candidate was emitted; gain counted is
`+0.0`.

Evidence: `build_manifest.json`.
