# task191 exact Identity/shape rewrite audit

Authority: `submission_base_8008.14.zip` task191, SHA-256
`76795962c3367d429658d959008a48dbb5e86ab880c8e707a573336714c58d2a`,
official-like cost **3436** (memory 3381 + parameters 55).

No candidate is admitted.

## Rewrites tested

1. Remove the `shape30hw -> Identity -> shape30hw_dyn` node and feed the
   scalar shape initializer directly. ORT rejects the graph because a
   one-element shape no longer receives the dynamic-shape behavior on which
   the authority's two-axis `CenterCropPad` nodes rely.
2. Change that initializer to explicit `[30, 30]`, remove the Identity, and
   rewire its consumers. This makes the two-axis nodes explicit but breaks
   the one-axis `input_hid` `CenterCropPad`; ORT rejects the graph with
   `Number of elements of input 'shape' (2) does not match the number of axes
   (1)`.

The authority deliberately reuses a dynamically cloaked scalar target for
both one-axis and two-axis `CenterCropPad` calls. Replacing it with either a
static scalar or static pair changes validation semantics, so the nominal
Identity saving is not a valid exact rewrite.

Evidence:

- `identity_result.json`
- `identity_shape2_result.json`
- `try_identity.py`
- `try_identity_shape2.py`

Decision: **REJECT**, projected gain **+0.0**. No submission, score ledger,
probe queue, or `others/71407` artifact was changed.
