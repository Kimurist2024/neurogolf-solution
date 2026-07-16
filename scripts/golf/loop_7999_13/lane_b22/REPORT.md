# B22 task224/task400 strict optimization report

## Outcome

- Exact source: `submission_7999.13_wave15_candidate_meta.zip`
- Source SHA-256: `0f106fa0d9599d4853397e0f9310e3ae1bcf47d6f418c6b9dec31e4a4490bc36`
- Tasks: 224 and 400
- Safe winners: **0**
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or shared handcrafted artifact was modified.

## True rules

The obfuscated Sakana references were expanded literally and checked against
train, test, and all 262 arc-gen cases. Both readable implementations are
266/266.

- task224 is a Type-D global geometry rule. The four gray extremal markers
  identify the outer rectangle; the non-gray color is recovered from the
  nonzero color set, and the missing outer border is restored.
- task400 is a Type-D global geometry/crop rule. The 5x5 blue cutout locates a
  patch, and the output is the diametrically opposite 5x5 patch of the fixed
  size-24 dihedral construction.

## task224

The exact Wave15 member costs `162 = 24 memory + 138 parameters`, is 266/266
under both `ORT_DISABLE_ALL` and default ORT, and has truthful declared/runtime
shapes. It is not an admissible optimization source under the campaign gate:
its final direct-output `Einsum` has **62 operands**, well above the 16-operand
limit. A derived candidate therefore had to be both cheaper than 162 and a
safe reconstruction of the global rule.

All four archived below-incumbent models were independently rechecked:

| cost | transformation | disabled | default | decision |
|---:|---|---:|---:|---|
| 156 | reuse H0B for H1B | 0/266 | 0/266 | reject |
| 156 | reuse H1B for H0B | 0/266 | 0/266 | reject |
| 158 | reuse Csum for Cdiag | 0/266 | 0/266 | reject |
| 158 | reuse Cdiag for Csum | 0/266 | 0/266 | reject |

There are no unused or identical initializers. The exact algebraic scans find
zero shared-operand fusion plans, zero identity operands, and zero single-use
Einsum inlining opportunities even with an operand cap of 128. A sound
Reduce/ArgMax/broadcast reconstruction necessarily loses the incumbent's
direct-output giant-Einsum scoring trick and did not yield a sub-162 model.

## task400

The exact member nominally costs `164 = 123 memory + 41 parameters` and is
266/266 in both ORT modes, but it is a strict **shape-cloak rejection**. Four
tensors are declared `[1,1,1,1]` while runtime produces:

- `scores`: `[1,10,30,30]`
- `scores_h`: `[1,10,30,30]`
- `loss`: `[1,1,30,30]`
- `code_i8`: `[1,1,30,30]`

The traced truthful intermediate footprint is 56,814 bytes, not the nominal
123. Thus a derived model had to be a truthful safe rebuild below cost 164.

The complete A17 loose-history inventory contains 21 byte-distinct nonbaseline
task400 models; the minimum actual cost is 165 and none is below 164. The
current graph has no unused or identical initializer.

The only large local reduction is collapsing the 50-element two-channel int8
feature into one channel before `QLinearConv`. This is not exact. Across the
complete known generator set, the crop codes map to nine reachable output
colors. All 256 int8 feature multipliers and all 256 per-class int8 decoder
weights were exhaustively checked; **zero one-feature decoders** separate all
nine colors. Algebraically, one QLinear feature is only a half-line classifier
after its shared zero point, whereas the two wrapping features are needed to
place all nine codes on separable directions.

## Admission disposition

No strictly cheaper model survives known correctness plus truthful-shape and
structural safety gates. Fresh 5000/5000 was therefore not run: there is no
eligible candidate to validate or promote.
