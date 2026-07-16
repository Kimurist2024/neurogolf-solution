# task158 independent review: PASS

This lane did not modify any submission, score file, root model, or `others/` artifact.

## Decision

Candidate `9d9a3ca8fb39856125925ea464ed1cc80f0301bd785ff7b60da37bd1c2b6b9d1` is **PASS** as a strict-lower replacement for immutable task158 `2823587ecc3f1b5b158357b5c32638003130f133ba6ab64a35337238f134aead`. Official cost is 7529 versus 7578 (reduction 49, score increment ln(old/new)=0.006487081728).

## Safety proof

For every invalid object slot, `obj_base=-1`. Anchor extraction bounds rows to [0, 25], so `p_mag` is [0.0, 12.5]; the gathered local-offset LUT is [0, 52]. Thus every invalid pre-cast cell index is [-1.0, 649.0] and every int32 Scatter index is [-1, 649]. This lies inside the ONNX axis-650 interval [-650, 649].
The same `p_valid` gate sets every invalid update to uint8 zero. `ScatterElements(axis=1,reduction=max)` starts from a uint8 all-zero `[1,650]` seed, so invalid collisions are identity operations and cannot erase a valid code.

## Independent execution evidence

Known: 266/266 correct and raw-bitwise equal to trusted-7612 in both ORT_DISABLE_ALL and ORT_ENABLE_ALL; runtime errors 0.
Fresh: seeds [15811317, 15811391], 2000 cases each (4000 total), candidate and trusted both correct and raw-bitwise equal in both ORT modes; runtime errors 0. Seeds differ from the author lane's 1581081/1581082.
All full-check, strict data-propagating shape inference, truthful runtime-shape, standard-domain, static-shape, Conv-bias UB0, and no-giant-Einsum gates passed.

Machine-readable details are in `review.json`; hashes are in `manifest.json`.
