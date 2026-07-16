# Global exact Einsum constant absorption scan 240

## Outcome

**No strict-lower structural lead.** No candidate ONNX or ZIP was emitted, so
the authority raw pass-through runtime gate had no survivor to execute.
`submission_base_8009.46.zip`, root staging, and score ledgers were unchanged.

The scan covered every current cost>100 task containing a multi-operand
Einsum, except task310's already-staged parity factor transform:

- 148 tasks;
- 401 multi-operand Einsum nodes;
- 27 priority models whose graph is a single output-only Einsum;
- 32,946 exact constant subset actions across pairs and connected chains of
  up to four operands;
- 28,845 exact-integer actions and 4,101 exact copy/sign/dyadic actions;
- 80 tasks with at least one exact local action;
- **0 globally strict-lower selections**.

The isolated `candidates/` directory is empty.

## Exact transform classes

For each Einsum, the scanner considered constant subsets that share a mode,
plus scalar absorption. Modes appearing only inside the selected subset were
contracted; modes still used by another operand or the output were preserved,
which covers elementwise sign/diagonal factors and permutation-mode
absorption as well as ordinary latent-index precontraction.

Only two arithmetic classes were admissible:

1. integer-valued factors whose complete product-and-sum bound is exact in
   float64 and whose result round-trips exactly through the serialized source
   dtype; and
2. copy/sign/permutation or power-of-two scaling with at most one contributing
   product per result cell and exact dtype round-trip.

No SVD, tolerance, fitted approximation, or numerical-rank decision was used.

## Global reuse accounting

Local element savings were not treated as graph savings. A binary MILP priced
the complete graph:

- actions sharing an operand occurrence are mutually exclusive;
- an original initializer is removed only when every one of its uses across
  the full graph is covered;
- identical combined tensors may be shared across actions and are charged
  once;
- new combined initializer elements are charged in full;
- only selections with lower total parameters and no intermediate-memory
  increase qualify.

This is the task310 lesson applied globally: absorbing one occurrence of a
reused factor cannot claim that factor's parameter removal. After these
constraints, every apparent pair/chain saving is neutral or worse.

The busiest priority single-node cases illustrate the result: task074 had
5,081 exact local actions, task306 had 464, task348 had 423, task011 had 344,
and task030 had 335, yet none produced a negative global parameter delta.

## Fail-closed details

Four pre-existing authority graphs (tasks117, 170, 243, and245) fail strict
inference because their supplied shapes conflict with `AffineGrid` or
`Reshape` inference. They were recorded and could not emit a candidate. They
also had no globally lower selection before that gate.

Task310 was excluded explicitly because its 501→491 exact parity-factor
rewrite is already staged and was not to be rediscovered.

`scan.json` contains the authority hash, per-task cost/shape status, exact
action counts, arithmetic-class counters, global solution status, and the
zero-survivor summary. `scan.py` is the deterministic reproducer.
