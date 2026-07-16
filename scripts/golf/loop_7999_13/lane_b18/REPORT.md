# B18 task089 / task255 report

## Verdict

**No safe improvement was found and nothing was promoted.** The exact
`submission_base_7999.13.zip` members recompute to task089 cost **1361** and
task255 cost **1336**. The assignment's task255 value 1162 does not match the
exact ZIP member (SHA256 `4d3ebe16cc55...`).

## task089

The exact baseline passes all 267 known examples only under ORT with graph
optimizations disabled. Runtime tracing finds **50 declared/actual shape
mismatches**. On fresh5000 it scores 4977/5000 (99.54%) under disabled ORT, but
default ORT cannot create a session.

The priority Wave12 candidate is exactly the archive rank-1 candidate
(SHA256 `33db6c4a4422...`) and reduces measured cost 1361 -> 1184 by removing
one apparently dead `ReduceMax(decode_big -> keep_red_big)` node. That node is
not semantically consumed, but it is operationally required by the incumbent's
shape-cloaked ORT buffer plan:

- disabled ORT: candidate runtime error on 267/267 known examples;
- disabled ORT: candidate runtime error on 5000/5000 fresh examples;
- default ORT: baseline and candidate both fail session creation;
- local `try_candidate` passes structural validation but fails gold inference;
- runtime trace still shows 50 declared/actual shape mismatches.

Therefore the 95%-accuracy base-equivalence exception does not apply: the base
fresh rate is above 95%, but the shave is not executable and cannot produce a
single raw output for a bitwise comparison. The cheapest truthful no-cloak
known-correct controls cost 4142 (`candidate_rebuild_v11.onnx`) and 20362
(`cand_u8.onnx`), both worse than 1361.

## task255

The exact baseline is cost 1336, has 16 declared/actual shape mismatches, and
passes all 265 known examples in both ORT modes. It fails fresh generation in
both modes at **4723/5000 = 94.46%**, below the permitted 95% threshold.

More importantly, the generator is provably non-functional as an input/output
mapping. The independent ambiguity script constructs two valid generator
configurations with byte-identical inputs and outputs differing in 15 cells.
No deterministic input-only ONNX can be exact for both, so a cheaper public-fit
candidate is not a sound score improvement.

## Gate disposition

Full checker, strict shape/data propagation, known dual-ORT, runtime shape
tracing, fresh5000, raw/sanitized differential, and the local candidate
validator leave **zero strict survivors**. The external validator was not run
because no model reached that gate; additionally the requested
`/Users/kimura2003/Downloads/neurogolf_team_validator_v1` directory is absent
in this environment. Root ZIP/CSV/best/artifacts/handcrafted files were not
modified by B18.
