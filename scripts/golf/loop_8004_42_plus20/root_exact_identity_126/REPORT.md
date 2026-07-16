# Current 8009.46 Identity-removal audit

The current400-member no-op census found14 apparent identities.  The six
highest projected deltas were tested directly: task269, task289, task262,
task214, task102, and task353.

All six fail full checker/strict inference after exact rewiring:

- task269/289: the initializer has one shape element while `CenterCropPad`
  lists two axes; the dynamic Identity is required to keep the broadcast-like
  runtime path opaque.
- task262/353: exposing the constant window length makes Hann/BlackmanWindow
  infer length30, contradicting the supplied singleton declaration used by
  the authority.
- task214: exposing target15 reveals a downstream `CenterCropPad` declaration
  mismatch.
- task102: both exposed constants reveal18 two-axis `CenterCropPad` arity
  contradictions.

Thus the estimated +0.9504 is not a valid exact reduction; the Identity nodes
are structural shape witnesses.  Safe winner count is zero.  Root submission,
score ledgers, and staged candidates were not modified.
