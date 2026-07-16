# Lane C40 — task391 corrected SOUND search

## Outcome

No winner; verified gain **+0.0**. The authoritative task391 fallback from
`submission_base_8000.46.zip` is preserved unchanged.

- SHA-256: `fd28c74c70b48687f6c9c798c44929b98350e30e0bf7b6d6cc629a41f83c1c45`
- actual cost: 104 (memory 67 + params 37)
- score: 20.35560910085863
- file size: 1,506,160 bytes, below the 1.44 MiB limit
- known: 267/267, wrong 0, errors 0 under both ORT modes
- declared/runtime shape mismatches: 0; measured memory: 67
- moved external validator: valid/preflight pass, known 267/267, errors 0

Per the corrected assignment, the fallback is the LB-white authority. The
private-0 evidence belongs to cheaper replacements, so a higher-cost SOUND
model is not an eligible replacement.

## True generator rule

The authoritative source is
`inputs/arc-gen-repo/tasks/task_f8b3ba0a.py` plus `common.py`.

1. Count each nonzero color in the rendered input.
2. Remove the strictly most frequent logical background color.
3. Emit the other three colors by descending frequency.

The default generator samples three minority logical counts without replacement
from `{1,2,3,4}`. Rendering duplicates every cell horizontally, so their counts
are three distinct values from `{2,4,6,8}`. The minimum background logical
count is `3*6-(4+3+2)=9`, strictly above the maximum minority count 4.
Therefore there is no tie in the default/private generator. The earlier tie
warning applies only to arbitrary manual `generate(width, colors)` overrides.

The independent reference matched all 267 known examples and 5000/5000 fresh
generator cases, with zero generation errors and zero invariant failures.

## Cheaper history and LB evidence

Six discovered models are below cost 104:

| cost | SHA prefix | structure result |
|---:|---|---|
| 85 | `fe1ff20a` | three `TfIdfVectorizer` lookup nodes — reject |
| 87 | `ebf61b0c` | three `TfIdfVectorizer` lookup nodes — reject |
| 87 | `fe92a087` | three `TfIdfVectorizer` lookup nodes — reject |
| 88 | `92ee7cdb` | three `TfIdfVectorizer` lookup nodes — reject |
| 88 | `befd50ee` | three `TfIdfVectorizer` lookup nodes — reject |
| 102 | `7ccd0d27` | three `TfIdfVectorizer` lookup nodes; quarantined private-0 — reject |

`docs/golf/private_zero_tasks.md` identifies the cheap task391 h7901
replacement as the private-0 regression and records the successful rollback.
The cost-102 member is explicitly preserved at
`artifacts/quarantine/task391_7801rej_cost102_private0_soloprobe.onnx`.

The authoritative fallback itself also contains a grandfathered lookup payload,
but its LB-white status does not authorize new lookup candidates.

## SOUND cost floor found

The smallest discovered table-free rule engine is
`others/highspeed/task391_cost139.onnx`, cost 139. Its truthful memory is already
104 bytes before adding 35 parameters:

- float32 color counts: 40 bytes;
- required TopK values: 16 bytes;
- required TopK int64 indices: 32 bytes; and
- cast/slice/label-emitter intermediates: 16 bytes.

ONNX requires the otherwise-unused TopK values output, and the prior int8 TopK
probe fails ORT session creation. The table-free repeated-ArgMax alternative
costs 148 (memory 122 + params 26). The earlier strided TopK reference costs
1487. Thus no audited standard rule-engine family crosses below 104.

## Gate decision

No model survived both prerequisites: cost below 104 and no
lookup/cloak/UB/unsafe giant structure. Consequently candidate known-dual,
fresh-dual-5000, and external-validator gates were not run; those gates are
mandatory only after a candidate clears price and structure. The baseline was
independently checked by both local dual-ORT and the moved external validator.

No shared ZIP, score ledger, CSV, or aggregate file was modified.
