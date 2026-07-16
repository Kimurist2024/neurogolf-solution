# B26 audit report

Baseline score label: `8000.46`  
Baseline ZIP SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`

## Accepted: task358

- Candidate: `task358_combine_r2_r3.onnx`
- SHA-256: `8d7c4eb231ce8f92d5dd413a0a36b19f74ab5b3e843507499102012d8bdee34d`
- Cost: `161 -> 155` (`-6`)
- Projected gain: `+0.037979248065219906`
- Known cases: `265/265`, wrong `0`, errors `0`, in both ORT modes
- Fresh cases: `5000/5000`, wrong `0`, errors `0`, in both ORT modes
- Fresh margin: near-margin cases `0`; minimum positive value `170.11326599121094`
- External validator: `ACCEPT_STRICT`; random raw and threshold equality `100/100`
- Structure: checker and strict inference pass; runtime shapes truthful; Conv UB count `0`; maximum Einsum inputs `44 -> 42`

The candidate uses the exact polynomial identity `(x - 2)(x + 2) = x^2 - 4` to replace two factors with one factor. It does not add or expand an Einsum.

## Rejected: task328

- Candidate: `task328_reuse_j_diagonal.onnx`
- SHA-256: `4d0fc5264833fbf46609fde690ad8635e208a2cec381e749b5707ef828866cb2`
- Cost: `558 -> 554` (`-4`)
- Potential gain: `+0.007194275634027231`
- Known cases: `267/267`, wrong `0`, errors `0`, in both ORT modes
- Structure: checker and strict inference pass; runtime shapes truthful; Conv UB count `0`; maximum Einsum inputs remains `58`
- Fresh gate probe: `16/16`, wrong `0`, errors `0`, in both ORT modes, but 4 cases per mode had positive raw values below `0.25`; minimum positive value was `7.316870026530253e-11`.
- Status: `REJECT_MARGIN`. The strict gate forbids any value in `(0, 0.25)`. This is a terminal rejection, so the remaining 5,000-case run and external acceptance audit were not continued.

Machine-readable status is in `winner_manifest.json`; full structural evidence is in `structural_audit.json`.
