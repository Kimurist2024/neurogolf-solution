# B16 — task157 / task319

## Outcome

No candidate survived the strict soundness gate. Winner count **0**, valid cost
gain **0**, valid score gain **0.0**. No root ZIP, CSV, ledger, artifact, or
handcrafted model was modified.

## Exact 7999.13 baselines

- task157: actual cost **853**, SHA `fcafb8af5728eb747bf3ad6fcadb67efce6a110884d3b57951a90f972748cd4f`. Full checker,
  strict shape inference, static-positive shapes, standard domains, Conv-bias,
  banned-op, and `<15`-input Einsum checks pass. Both ORT modes are 265/265.
  It is nevertheless prohibited because it contains the explicit visible
  fixture keys `fixk_train`, `fixk_a117`, and `fixk_a187`.
- task319: actual cost **1023**, SHA `9e57aaabb6086a97ee6ac2f6afca26f6f2dcf2896c841bd8f443f54fa15b5941`. Both ORT modes
  are 267/267, but runtime tracing finds **28**
  declared/runtime shape mismatches, so it is a shape cloak; it also includes a
  fixed correction pattern.

## New sound rebuild attempt

`candidate_task157_no_lookup.onnx` was rebuilt from the exact task157 model by
removing all three fixture-key correction branches and returning to the generic
`bstarts` rule. SHA `ddac98a3ed8133031c78b515b6ecc1daf5c5f63d3f120426cf780e6254a72a40`; actual cost **833** (raw cut **20**),
truthful static shapes, no lookup keys. It is rejected: both ORT modes are
262/265, and an independent cached 1000-case audit is 981/1000 with 16 errors
on uniquely solvable inputs. This is a real accuracy regression, not only the
generator's irreducible ambiguity.

## Archive audit

The task157 cost-520 header probe is only 4/265. The prior no-key cost-851 probe
is 262/265. For task319, archive r01 ties actual cost 1023 and is cloak/lookup;
r02 cannot instantiate in either ORT mode; r03/r04/r05 cost 1086/1131/1132 and
are all shape cloaks. Therefore no archive graph is strictly cheaper, truthful,
and known-correct.

## Generator truth

Constructive witnesses in `generator_collision_audit.json` reproduce valid
calls with identical input and distinct outputs for both hashes 6a1e5592 and
ce602527. A deterministic ONNX cannot achieve zero error over every valid call.
This does not excuse the 16 unique-case misses of the compact task157 rebuild.

## Deferred gates

Fresh-5000 dual-ORT and the external team validator were not run because no
candidate passed the earlier actual-cost + known-dual + truthful-shape +
provenance gates. Running them cannot turn a known-invalid graph into a winner.
