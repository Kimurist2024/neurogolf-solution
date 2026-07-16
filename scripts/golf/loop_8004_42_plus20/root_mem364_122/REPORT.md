# task364 exact logic/memory audit

Authority SHA-256:
`2ba1bb84e800b98cdcac9e4d8cb8970d08e532fadf84b64f1b3d88d21ab2a3db`;
official cost 685 (memory 683, parameters 2).

Three exact approaches were evaluated:

1. Finite 0/1 arithmetic was compiled from 468 nodes to a 300-node Boolean
   DAG with zero initializers.  The Boolean truth functions are exact on the
   fixed benchmark one-hot domain, but `And/Or/Not` expose real intermediate
   shapes to the profiler and cost rises to 115431.  Rejected on cost.
2. The float16 Boolean encoding was changed to float8.  ONNX full checking
   rejects float8 as an unsupported `CastLike` target for this graph.  Rejected
   before runtime.
3. The directional `CenterCropPad` triplets were probed exhaustively against
   all two-op intermediate sizes.  Neither one-cell shift has a two-op
   equivalent, so the dominant shift circuit cannot be shortened that way.

The historical truthful task rule rebuild costs 89186, also far above the
authority.  Safe winner count is zero; no root or staged model was changed.

