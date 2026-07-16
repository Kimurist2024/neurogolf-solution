# task379 residual exact regolf

## Decision

**REJECT — no admissible strict-lower exact candidate.**

The LB8009.46 authority member was reprofiled as:

- SHA-256: `854c63d966310949803391cf4c019b02a9c0f2a53578257fee5898386e53cf64`
- cost: memory `1570` + parameters `377` = **1947**
- structure: 68 nodes, 29 initializers, no stored `value_info`
- full ONNX checker: pass
- strict shape inference with data propagation: pass

No candidate is emitted. Cost remains 1947 and gain is 0.

## Generator-support boundary

`inputs/arc-gen-repo/tasks/task_ecdecbb3.py` was inspected directly. The
support consists of 12–20 square grids, one or two horizontal cyan separator
lines (with an optional whole-example transpose), and red seeds in qualifying
rows/columns. Each seed projects vertically to the applicable separator(s),
painting the path red and a cyan 3x3 stamp with a red center at the contact.

This is a private-zero lineage. Only an exact implementation over that complete
support was admissible; no POLICY90 approximation, lookup, cloak, error path,
undefined behavior, or partial-support shortcut was considered acceptable.

## Residual exact audit

The current member is already the exact algebraic cleanup of the former cost
1949 member: `QRow1_2` was eliminated by reconstructing it from `WBasis`,
saving two parameters. All 29 remaining initializers are used and all 68 nodes
are live. Because the parameter metric counts tensor elements, merely narrowing
an initializer dtype cannot reduce cost; the terminal Einsum also requires a
common input type.

The only remaining initializer-pair precontractions identified in the
all-authority scan are four occurrences of:

- `QCore2[2,2,2]` (8 elements) with `QMode3x2[3,2]` (6 elements), and
- `QCore2[2,2,2]` with `QExpand3x2[3,2]` (6 elements).

Each occurrence appears to reduce 14 operand elements to a 12-element dense
product locally. It is not a graph-level reduction: `QCore2` has five uses and
must remain, while each map has two uses. Covering all four occurrences changes
the shared family from `8 + 6 + 6 = 20` elements to
`8 + 12 + 12 = 32`, a **12-parameter increase**.

The prior exact and diagnostic candidates were also reconciled against the
1947 authority:

| Probe | Nominal cost | Result |
|---|---:|---|
| Drop identity mode | 1942 | Invalid: `NU` is rank 2 and not identity |
| Drop redundant `E` | 1891 | Invalid: `E` carries the spatial coordinate row |
| Fold mode into `M2` | 1942 | Real-algebra rewrite changes FP16 terminal contraction/raw output |
| Add to Sum | 1947 | Invalid ONNX Runtime type: int32 input has no applicable Sum kernel; no saving |
| Exact operand reorder / reciprocal scaling | 1947 | No strict reduction and does not remove nonfinite output |
| Sentinel or uniform-scale diagnostics | 1947 | Nonexact raw algebra; no strict reduction |
| Metadata/default/type cleanup | at least 1947 | No eligible stored metadata/default or element-count saving |

The dedicated finite-output audit provides an additional hard rejection for
the current terminal design. Its 28-input float16 Einsum is the first
nonfinite-producing node; all preceding node outputs are finite. It produced
12,896 negative infinities over the known set in each audited ORT mode, while
the exact real contraction contains valid values below the float16 finite
range. A raw pass-through rewrite preserves those infinities. Widening the 274
dynamic terminal operands to float32 requires at least 548 additional
parameter elements, giving a lower bound of 2495 rather than a reduction.

## Gate outcome

Winner-only validation (two fresh disjoint seeds × four ORT configurations,
raw pass-through, and a complete formal/exhaustive generator-support proof) was
not run because no candidate passed the prerequisite `cost < 1947` screen.
This is a rejection, not a promotion. No model, archive, root submission,
score ledger, or immutable authority artifact was modified.

