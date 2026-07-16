# task209 residual exact re-golf lane 218

## Result: NO ELIGIBLE WINNER

The staged authority remains unchanged:

- `others/71407/task209.onnx`
- SHA-256 `87690aaddd78db9a54a41b4a11edb73d503966eb8d27b4b60a3569fd1db0a751`
- official profile `memory=1832, params=253, cost=2085`

One strict-lower diagnostic was found (`2085 -> 2083`), and it has a complete
operator-support equivalence proof.  It is **not admissible** because it
inherits the authority's two shape cloaks and 16 declared/runtime shape
mismatches.  Removing the cloaks and attaching truthful static shapes produces
fully standard, exact controls, but the best such control costs `2650`, above
the authority.  Nothing was copied to `others/71407`, and no root file or
ledger was edited.

## Candidate matrix

| candidate | memory | params | cost | full/strict/data-prop | runtime shape truthful | standard/no forbidden/no lookup abuse/UB0 | decision |
|---|---:|---:|---:|---|---|---|---|
| authority | 1832 | 253 | 2085 | pass | **fail: 16 mismatches** | inherited 2 `CenterCropPad` cloaks | fixed incumbent |
| inherited roundless | 1830 | 253 | **2083** | pass | **fail: same 16 mismatches** | no new forbidden/nested/lookup op; UB0 | reject by truthful-shape gate |
| decloak rbits only | 2443 | 253 | 2696 | pass | pass for retained graph metadata | one inherited cidx cloak | reject: not lower |
| decloak cidx only | 2781 | 253 | 3034 | pass | pass for retained graph metadata | one inherited rbits cloak | reject: not lower |
| full decloak | 2405 | 253 | 2658 | pass | pass | pass | reject: not lower |
| full decloak + shared Unsqueeze axes | 2405 | 247 | 2652 | pass | pass | pass | reject: not lower |
| full decloak + roundless | 2403 | 253 | 2656 | pass | pass | pass | reject: not lower |
| full decloak + both shaves | 2403 | 247 | **2650** | pass | **pass: 0/157 mismatches** | pass | reject: not lower |

All listed official profiles were measured through
`scripts/lib/scoring.py::score_and_verify` over the 266 known cases.  Every
profile is known-correct.  The truthful controls use only the standard domain,
opset 18, no nested graphs, no banned op, no `TfIdfVectorizer`/`Hardmax`, a
largest initializer of 30 elements, and zero Conv-bias UB findings.

## Exact 2-cost diagnostic

The authority computes:

```text
pclowbit -> Cast(float16) -> Log -> Selu(gamma=1.4422534704208374)
          -> Round -> Cast(uint8 pcidx)
```

`pclowbit = pball & (0 - pball)`.  On every valid input `pball != 0`, so the
value is exactly one of the 32 nonzero uint32 powers of two.  The diagnostic
changes the Selu gamma to the binary-exact `1.443359375`, removes `Round`, and
casts `pclog2` directly to uint8.

The micro-operator audit exhausts all 32 possible lowbit powers in four ORT
configurations.  `Cast(Round(old_selu(Log(Cast(x)))))` and
`Cast(new_selu(Log(Cast(x))))` return exactly `0..31`, respectively, with
`32/32` equality in each configuration.  Thus the downstream `pcidx` and final
output are identical on the complete reachable support, not merely sampled.

Diagnostic artifact:

- `candidates/task209_inherited_roundless.onnx`
- SHA-256 `d9466f26196562afe26bfcb0eaf72b9bf95330708fb396527772c28c777b0e95`
- projected gain if the shape-truth gate were explicitly relaxed:
  `ln(2085/2083) = +0.0009596929719299144`

It remains quarantined because a complete runtime trace finds these inherited
false declarations (16 total): the hidden rbits branch, its row/source/pattern
descendants, the hidden cidx branch, `codepacked`, and final `output`
(`declared [1,10,30,1]`, actual `[1,10,30,30]`).

## Runtime equivalence audit

For the strict-lower diagnostic:

- complete lowbit support: `32/32` equal in each of four ORT configs;
- known: `266/266` raw-bitwise equal to authority and `266/266` correct in each
  of four configs;
- fresh: two independent streams, 1000 each, total `2000/2000` raw-bitwise
  equal in each of four configs;
- authority/candidate fresh truth: `1915/2000 = 95.75%` in every config;
- runtime errors: 0;
- final nonfinite values: 0;
- threshold differences: 0.

The candidate introduces no semantic or runtime regression, but exact
authority pass-through does not cure the authority's pre-existing shape
misdeclarations.  Therefore it fails this lane's absolute truthful-shape gate.

## Other exact shave tested

The two runtime rank-2 tensors are reshaped to `[1,3,3,1]` and `[1,3,2,1]`.
After truthful decloaking, both fixed Reshapes can be replaced by Unsqueeze with
one shared axes initializer `[0,3]`, reducing parameters by 6 (`253 -> 247`).
With inherited false shapes, however, ONNX static inference sees the inputs as
`[1,3]`/`[1,2]`, infers wrong Unsqueeze outputs, and full checking fails.  It is
therefore only valid in the truthful controls, whose total cost remains too
high.

## Evidence

- `build_candidates.py`, `build.json`: builders, official profiles, hashes,
  structural gates.
- `audit_candidate.py`, `audit.json`: 4-config known/fresh raw audit,
  exhaustive lowbit support, runtime-shape truth, standard-op scan.
- `runtime_metadata.json`: measured fixed runtime shapes/types used to construct
  the truthful controls.
- `candidates/task209_decloak_unsqueeze_roundless.onnx`: best truthful control,
  SHA-256 `f9f25b823d0f97490dddce97cef22b11c9a12aeae1d2ad7c1ffb51b234103be4`.

Stop reason: all exact residual shaves either preserve the incumbent's 16
shape mismatches or become more expensive after truthful decloaking.  The best
fully admissible control is 567 cost units above the authority.
