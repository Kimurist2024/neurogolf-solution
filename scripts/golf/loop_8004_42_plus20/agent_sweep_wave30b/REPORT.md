# sweep wave30b independent audit

## Outcome

- Baseline authority: `submission_base_8005.16.zip`
- Baseline SHA-256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`
- Scope: 8 tasks, 41 distinct candidate payloads
- Accepted: **0**
- Projected accepted gain: **+0.0**
- ZIP integration: **not performed**
- Verdict: **NO_SAFE_CANDIDATE**

Every candidate was independently re-profiled against the member in the latest
8005.16 archive. All 41 pass ONNX full checking, strict shape inference with
data propagation, static-positive node-output shape checks, standard-domain
checks, and the Conv-family bias UB=0 gate. Those checks do not override the
explicit no-giant gate or dual-ORT runtime failure.

## Task dispositions

| Task | Baseline SHA-256 | Cost | Representative best candidate SHA-256 | Candidate cost | Apparent gain | Decision |
|---:|---|---:|---|---:|---:|---|
| 199 | `d236c732d0df80270154b8ee593e17768dd54fc8dcec4aac93e752474651383e` | 261 | `d8202e8d67217e7c923845a9ca1cf9e3fc81537f1811239d9ac27ab01491f236` | 241 | +0.079723474 | reject: 17-input giant Einsum |
| 070 | `632f198acf95fb32697679fa357b6c0c979360e3cf5a0a7173a3ff8917562e45` | 75 | `d62a44cb5bd5f3d0c8d9d9d97155949d579fbdbfddd6fa0b4c77784d873bc9ec` | 64 | +0.158605030 | reject: private-risk lineage and 17-input giant Einsum; both cost-66 fusions still have 16 inputs |
| 333 | `5bb4ddf301f100bca1a4c6151452bf3f0bacf41cf1d8348573f28771d77eb826` | 423 | `9946cde435b6ca982b64b71d63458b6a179f8f81674b843819c16360bef3edf6` | 412 | +0.026348830 | reject: 36-input giant Einsum |
| 165 | `2e1af6681882d90b3288ac3df592557111c3e525e608dc1b69447c597e680cbb` | 592 | `60b720142245a29e2939f3da7e415549be695d74664d531dbefc5cc7d3c634ee` | 551 | +0.071771826 | reject: dual-ORT runtime gate fails |
| 169 | `580aeebba96eacc482a11e3a6da6f4295758ab24e2376a8a248327f943cb33b4` | 248 | `281e63bcb56e16acdc05af0c7f3d84efea3f5dbdaa098af76555df02a7dcbff4` | 246 | +0.008097210 | reject: private-risk candidate and dual-ORT runtime gate fails |
| 328 | `08ba1aa525d67f290c13e7b79aef339aeb5912bf0d1b0b379ff6ab8792cf576a` | 558 | `4d0fc5264833fbf46609fde690ad8635e208a2cec381e749b5707ef828866cb2` | 554 | +0.007194276 | reject: 58-input giant Einsum |
| 379 | `4ddae903db9b2d4aceef3c501691b5cd2b862bc209f58f2e88969332c06dd455` | 1949 | `854c63d966310949803391cf4c019b02a9c0f2a53578257fee5898386e53cf64` | 1947 | +0.001026694 | reject: 28-input giant Einsum |
| 013 | `d0d2eea63b192eb7d9258f8408d00a72229dfbaef130887f64b708aef3536a2d` | 638 | `ad4eb35978f3e38d1d3e2afdd55e55db871962cc2ea4c989675d9d583434103b` | 636 | +0.003139720 | reject: every variant retains a 51-input giant Einsum |

The machine-readable result records the full SHA-256, path, measured memory,
parameters, cost, gain, op histogram, structural evidence, and rejection reason
for all 41 variants (including all task013 and task070 alternatives).

## Runtime evidence for the two non-giant candidates

### task165

- Candidate cost: 551 = 481 memory + 70 params.
- ORT_DISABLE_ALL: 0 right, 0 wrong, **265/265 runtime errors**. First failure
  is a Slice buffer-reuse shape conflict (`{1,9,30,30}` vs
  `{1,10,30,30}`).
- Default ORT: session construction fails; `CenterCropPad` reports a shape
  vector/axes count mismatch. It is conservatively counted as 265/265 runtime
  failures.
- A truthful intermediate runtime-shape trace cannot complete after this
  mandatory runtime failure. Fresh generation was therefore not run.

### task169

- Candidate cost: 246 = 244 memory + 2 params.
- ORT_DISABLE_ALL: 0 right, 0 wrong, **266/266 runtime errors**. First failure
  is a Slice buffer-reuse shape conflict (`{1,1,29,29}` vs
  `{1,1,31,29}`).
- Default ORT: session construction fails with the same `CenterCropPad`
  shape-vector/axes mismatch family. It is conservatively counted as 266/266
  runtime failures.
- The candidate is from a monitored private/allocator-risk lineage, so even a
  runnable model would require known and independent fresh 100% in both modes.
  It does not reach that gate.

## Fresh policy

Fresh validation was intentionally skipped for every candidate. Thirty-nine
candidates fail the no-giant prerequisite before correctness sampling, and the
other two fail complete known execution in both ORT modes. Running fresh after
either condition would not make a candidate adoptable.

## Evidence files

- `result.json`: final complete candidate inventory and dispositions
- `audit_partial.json`: incremental task-level checkpoint
- `audit_sweep.py`: reproducible non-promoting audit driver

No baseline, submission ZIP, root submission, score file, or candidate source
was modified.
