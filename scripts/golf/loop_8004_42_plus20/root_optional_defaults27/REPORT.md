# 8005.16 optional-default scan

**Safe adoptees: 0.**  All 400 latest members were scanned for removable exact
default inputs: zero Conv/ConvTranspose/Gemm/QLinearConv bias, zero integer-op
zero points, zero quantization zero points, zero Pad value, and unit Slice
steps. No candidate both used an exact default and made an initializer dead.

Gain counted is `+0.0`. Evidence: `build_manifest.json`.
