# Wave C4 final report

The exact `submission_base_7999.13.zip` members were the sole cost baseline.
One strict winner was found: task270 improves from **608 to 595**, for a score
gain of **+0.02161347642053717**. The candidate is isolated in this lane; no
root submission, score CSV, or aggregate baseline artifact was changed.

## Accepted: task270

`task270_pack_petals.onnx` shares the two flower-petal moment streams without
changing the task rule. Petal colour 3 is encoded in the low lane and colour 7
in a `2048`-scaled high lane. Counts and first moments are unpacked with exact
nearest quantization; the squared-row moment uses divide-plus-uint8-cast so
values above 255 retain the incumbent's intentional uint8 wrapping rather
than saturating.

| metric | exact baseline | candidate |
|---|---:|---:|
| memory | 408 | 404 |
| parameters | 200 | 191 |
| total cost | 608 | 595 |
| nodes | 50 | 59 |

Strict validation passed all gates:

- library gold and official gold both pass; minimum stable margin is
  `0.41015625`;
- fixed gate fresh audit: **5000/5000**, zero wrong and zero runtime errors;
- independent seed `270799913`: **5000/5000**, zero wrong and zero errors;
- all **256/256** legal petal-presence masks pass the full ONNX;
- all **79/79** valid renderer axis-state pairs, including padding, pass;
- minimum positive renderer/full-model value is `0.72216796875`, with no
  positive in the unsafe `(0, 0.25)` band;
- full checker, strict shape inference, static shapes, standard domain,
  no nested graphs/functions/sparse tensors/banned ops all pass;
- the graph contains no Conv or ConvTranspose, so the Conv bias/UB finding
  count is zero.

The final candidate SHA-256 is
`f2a877344c1a3c43672303ed32a90c47a7007b336bd32008fd85221053e6d78f`.
Detailed evidence is in `task270_verification.json` and
`task270_structural_domain_audit.json`.

## Rejected candidates

- **task238:** the nominal `562 -> 535` `pat_shift` value-info shave is not a
  real optimization. It raises an ORT `Slice` buffer-reuse mismatch
  (`{1,1,5,5} != {1,1,7,7}`) on all 266 known cases and all 3000 fresh cases.
  Rebuilding the constant-one path with truthful shapes restores correctness
  but costs 587, so both variants are rejected.
- **task012:** the exact model is one depthwise Conv at cost 710. A complete
  LP hard-separation sweep covered 351 alignments for all kernel geometries
  below 70 elements that could possibly beat the incumbent. None admits a
  valid foreground classifier. The first feasible 7x10 geometry has the same
  710 parameters and is not a winner.
- **task165:** the apparent CSE improvement `592 -> 552` is a shape-cloak
  failure, not a valid shave. It raises a Slice allocator mismatch
  (`{1,9,30,30} != {1,10,30,30}`) on all 265 known cases and is rejected.
- **task066:** `WR[3,3]` does factor exactly as `Tcol[3,2] @ M[2,3]`, but the
  final output Einsum already consumes all 52 legal ASCII labels. Inline
  factorization needs one new dimension-2 latent label even after exploiting
  the one-hot shared row. Materializing the 3x3 product adds 36 bytes of
  runtime memory, exceeding the three-parameter saving, so no strict-cheaper
  candidate exists by this route.
- **tasks046, 066, and 117:** the repository-wide unique-model scan, actual
  scoring, conservative optimizer pool, and local structural probes yielded
  no cheaper known-correct candidate.

## Harvest coverage

The lane collected 6,244 raw candidate references and deduplicated them to 407
unique models across tasks 012, 046, 066, 117, 165, 238, and 270. Results were
7 exact-baseline rows, 338 static cost-floor rejects, 51 structural rejects,
7 scored alternatives, two scanner exceptions, one timeout, and one
unscorable graph. The accepted manifest contains only the independently
rebuilt and fully gated task270 model.
