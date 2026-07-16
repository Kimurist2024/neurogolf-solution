# Lane B31 — task192 SOUND rebuild

No task192 model is safe to promote. The retained authority member remains
SHA-256 `e7f9a11b...`, cost **1609**. Projected gain is **+0.0**. This lane did
not write a ZIP or modify the shared submission.

## Rule and controls

The raw rule selects the most frequent color `A` in `1..9`, breaking ties
toward the smaller color. A cell is emitted as `A` iff its center is nonzero,
an `A` is present in the center-inclusive horizontal radius-1 window, and an
`A` is present in the center-inclusive vertical radius-1 window. Every other
in-grid cell must emit channel 0; the padded exterior must emit no channel.

The existing bitset control is generator-SOUND at known **265/265** and fresh
**5000/5000**, but costs **3325** (`memory=3208`, `params=117`). It is therefore
1716 worse than the retained LB-white baseline and remains rejected.

## Small-Conv synthesis

`build_conv3_sound_probe.py` extracts local one-hot patches only from fresh
public-generator cases and solves minimum-L1 linear separators. It never uses
visible fixtures or stores output grids.

A 3x3 selected-color classifier is separable and has nominal cost **1009**.
The first emitted probe intentionally exposed a crucial scoring constraint:
Neurogolf thresholds every output channel independently with `> 0`; it does
not take an argmax. The probe had a zero background logit, so it failed known
gold and fresh **0/500** even though its in-grid argmax labels could match.

Adding the required positive channel-0 output makes the 3x3 separator
infeasible on 500 fresh cases. Both cheaper 3x4 alignments (left padding 1 and
2, nominal cost **1309**) are also infeasible. The obstruction is structural:
the target uses saturated Boolean presence in each axis and then an AND. Raw
Conv receives unsaturated counts, so the background complement cannot be
represented by one linear threshold. This agrees with the independent archive
audit in `lane_c35`, where five very cheap grouped-Conv models scored only
49--88% fresh and were all rejected below the user's 95% gate.

## Packed-rule lower bound for the tested family

The sound control's generator-equivalent rectangle morphology has twelve
counted 30-word intermediates through `box`: two float pack results, two casts,
and eight uint32 morphology results. That is already
`12 * 30 * 4 = 1440` bytes.

Replacing the old table/Gather renderer with a free terminal signed-overflow
Einsum still needs at least the packed feature tensor `[V, K, sentinel]`
(`3 * 30 * 4 = 360` bytes), plus the 40-byte histogram and 8-byte ArgMax. Thus
this family has a memory floor of at least **1848** before dynamic channel
factor construction and parameters. It cannot beat cost 1609. The exact raw
rule additionally needs a dynamically selected `A` row mask and is no cheaper.

## Decision

- Retain baseline task192 cost 1609.
- Reject the SOUND bitset control cost 3325.
- Reject all 3x3/3x4 linear probes; none reaches the correctness gate.
- Do not run known-dual/fresh5000/external500 promotion gates because no strict
  cheaper candidate survives the preliminary exactness requirement.

Machine-readable evidence is in `rejected_manifest.json` and
`conv3_probe_manifest.json`.
