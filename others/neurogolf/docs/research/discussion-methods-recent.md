# NeuroGolf 2026 — Current Meta Synthesis (discussions, last ~3 weeks, as of 2026-06-12)

Sources: 33 discussion threads in `docs/research/discussions/` (IDs cited inline) + our own
`docs/research/cost-gap-analysis.md` (run-012: total **6347.82**, avg cost **29,727**, median **15,230**).
Leaders: ~7700 (#1 crodoc) ≈ 19.25 avg/task ≈ implied avg cost **e^5.75 ≈ 314**.

Confidence tags: **HOST** = stated by Michael D. Moffitt / Kaggle staff; **COMMUNITY** = credible
competitor claim with data; **SPEC** = speculation.

---

## 1. How top teams break the "static-shape ceiling" (the techniques behind 7400+)

### 1.0 First fact: there is no 6360 ceiling

- The "~6360 maximum for static-shape networks" claim was **retracted by its own author**
  (705373). It came from scoring with the **stale bundled `score_network`** which still summed
  `macs + memory + params` AND counted the output tensor.
- **Actual metric (per official updates, 705373): `max(1, 25 − ln(memory + params))`** —
  MACs excluded, INPUT and OUTPUT tensors excluded from memory. A single-node graph has
  memory = 0; the [1,10,30,30] output costs nothing. There is no per-task floor of ~15.89.
- **HOST (705448)**: memory = sum of statically-declared intermediate tensor shapes, but the
  scorer also sweeps actual ONNX Runtime shapes and takes **max(static, runtime) per tensor**.
  Under-declaring buys nothing; over-declaring costs you.
- Our own data confirms our 6349 plateau is a **genuine compression gap, not a scorer
  artifact**: run-012 avg cost 29,727 / median 15,230 vs the ~314 leaders average and ~359
  uniform break-even for 7700. We already have tasks at 25.00, so our scorer formula matches;
  our graphs are simply ~50–100x too expensive.

### 1.1 What 7400+ actually consists of (with evidence)

1. **Pure functional, (near-)zero-parameter ONNX — no trained networks.**
   - 703431: T.-C. Chang's team got a "massive score boost" by *completely bypassing CNNs* —
     for many tasks the transform (e.g., object shift vectors) is **constant across the
     dataset**, so they compile pure tensor math with 0 params directly into ONNX. CroDoc (#1)
     declined to reveal his architecture but did not contradict this.
   - 703914 (6580 harness builder): trained-NN tracks **failed outright**; "understand the
     rule, express it more compactly" remains the most effective strategy.
   - 699313: Chris Deotte builds graphs **by hand** in a click-and-drag GUI — manual per-task
     program synthesis, not learning.
2. **The 600-byte regime is a median story, not an average story** (701942, jacekwl, 9 upvotes):
   his average cost is ~7,200 but **median ~860 bytes**. Because score is `25 − ln(cost)`, the
   sum is dominated by the worst tasks. Leaders' floor: robga's lowest-4 tasks total **54–60
   pts (~14–15 each)**; Yiheng Wang (top solo) has only 2 tasks at 13.xx (704570, 704006).
   **No task needs to be below ~13.5.**
3. **Per-task LLM golf loops at industrial scale** is the engine:
   - Tony Li (top-2) reached **7600 on ChatGPT Plus $20/mo** web UI alone — multi-tab
     copy-paste iteration (704942). Now ChatGPT Pro + Codex automation 2–6 days/week.
   - Jan Vorel: **+179.31 pts in one day** at ~7000 level; ~**1.4M tokens per LB point**
     (704568). Sayaka Miki: 14B tokens of gpt-5.5 in a month.
   - 703914: one agent ≈ **+10–15 pts per 8h night** (mean win +0.36, median +0.21); agents
     never find the optimum first pass — **~5 repeated passes per task** with attempt history
     are needed; "give me 10 ideas" loops and automated task-solving tracks produced nothing.
   - Failure-logging: log every "wrong" optimization per task and feed back as **negative
     constraints** (704762, Yiheng Wang).
4. **Operator-level compression techniques that the strong models execute** (703462 benchmark,
   Opus 9/10 deploy-safe vs GLM-5.1 6/10, GLM-4.7 0/10):
   - narrow dtypes (bool/int8 intermediates instead of fp32 — 4x memory cut per tensor),
   - algebraic rewrites preserving bit-identical outputs,
   - **collapsing N parallel branches into one batched MatMul**,
   - fewer, smaller intermediates (each intermediate tensor's declared bytes is the cost).
5. **Data-dependent region selection with statically declared shapes** — HOST-endorsed
   (705448 → discussion/695972): runtime-computed Slice positions are fine as long as **no
   tensor carries `dim_param`**; when the slice output shape is actually constant, patch it
   into `value_info` manually (704900). This unlocks crop/relocate logic without dynamic shapes.
6. **Exploiting in-distribution hidden tests** — HOST (705448): hidden tests are just ARC-GEN
   run extra times, **always top-left anchored** in the [1,10,30,30] tensor, all grids ≤30x30,
   validated against last year's top-10 Code Golf outputs (705373). Networks need to be correct
   only for top-left-anchored content → bounded top-left crops are legal cost savers. Examples
   with grids >30x30 are **ignored by the scorer** (703200) → skip handling them, save bytes.
   Public LB = final; no re-scoring (HOST, 705448) → **overfitting to the live scorer is a
   sanctioned strategy**.
7. **Runtime is a spendable resource**: top solutions run 12–18 min of the 30-min budget;
   fast 2-min submissions correlate with lower scores (704762). Yash bhaskar's team (7377)
   explicitly researches "leveraging runtime for cost reduction" (704982).

### 1.2 What does NOT work (multiple independent reports)

- Trained neural networks (703914, 703431 by implication).
- Brute-force operator-combination synthesis to *solve* tasks from scratch (704844, Chet).
- Fully autonomous "solve the benchmark" loops without per-task grinding (703914 trajectory:
  overnight "win gold" Codex plateaued ~5000–5800 until he built a harness).
- Trusting local scoring: task310 measured local >20, real Kaggle 14 (703232).

---

## 2. Ranked actionable methods for us (expected points × feasibility)

Our position: 6347.82, 170 tasks below 15 pts (5 in [12,13), 67 in [13,14), 98 in [14,15)),
cost spread across the whole distribution (Q1..Q5 gains-to-314: +51/+214/+310/+381/+446).
Census: 32,484 nodes, 11.6 MB intermediate memory, 243k params; top ops Cast x3649, Mul x3258,
And x2867 — heavy dtype churn and elementwise sprawl.

| # | Method | Evidence | Expected gain | Effort | Risk |
|---|--------|----------|--------------:|--------|------|
| 1 | **Worst-cost-first re-synthesis queue (floor raising).** Rank all 400 by cost; rewrite as compact functional graphs starting from the priority-50 (task255 @362k → task281 @70k). | jacekwl median-vs-mean insight (701942); robga floor data (704570); our scenario table: all tasks cut to cost 900 = **+1018**, top-50-only = +234 | +200–400 near term, +1000 if sustained to mid-pack | High but parallelizable; this is just method #2 pointed at a sorted list | Low — proven attainable scores exist for our worst tasks (Tony's bottom-15 list with exact values, 704006) |
| 2 | **Per-task LLM golf loop with repeated passes + negative-constraint logs.** One task per session, full context, ~5 passes/task, log failed optimizations and feed back. ChatGPT web multi-tab (Plus suffices) + Codex for grinding; Opus for hard equivalence-preserving rewrites. | Tony Li 7600 on $20/mo (704942); +10–15 pts/agent/8h, 5-pass finding (703914); 1.4M tokens/pt (704568); Opus 9/10 deploy-safe (703462) | +10–15/day per agent-night, compounding | Medium setup, then continuous | Token cost; hidden-set overfit (mitigate with #5, #6) |
| 3 | **Kill the Cast/fp32 tax: narrow dtypes + intermediate-count reduction + branch collapsing.** Our 3649 Casts and 11.6 MB of intermediates are exactly the cost the metric charges. Batch parallel branches into single MatMul/elementwise ops; keep intermediates bool/int8. | 703462 (the three winning idea-classes); metric = ln(memory+params) with max(static,runtime) per tensor (705373, 705448) | Mechanical 2–4x cost cut on mid-pack ≈ +0.7–1.4 pts/task on hundreds of tasks | Medium — semi-mechanical, scriptable as a rewrite pass | Bit-identity must be verified per task; runtime shape sweep punishes sloppy declarations |
| 4 | **Replace Conv/MaxPool morphology with constant functional math (zero-param rewrite).** Our worst tasks are Conv/MaxPool-heavy (255: Convx57, 101: Convx31, 286: MaxPoolx59, 118: Convx9+MaxPoolx10). Where the transform is constant across the dataset, hardcode it. | 703431 ("massive score boost", 0-param functional topology); 703914 NN tracks failed | task255 alone: 12.20 → ~18+ if cost drops to <1k; similar for 101/133/158/96 | High per task (real re-derivation of the rule) | Per-task; some rules genuinely need structure — fall back to #3 |
| 5 | **Single-task zip probing on Kaggle as ground truth (100 subs/day).** Submit one-task zips to get true per-task scores; bisect bundles for 0-scoring tasks. | 703282 (technique), 703232 (task310 local >20 vs real 14), 703112 (100/day confirmed by staff) | Indirect — prevents wasted days and stale-baseline agent loops (Ali lost a day, 699840) | Low | Burns submission budget; undocumented scoring timeout exists (703286) |
| 6 | **Synthetic ARC-GEN holdout + insoluble-sample filter.** Generate extra samples from google/arc-gen as a local hidden set; filter natively-insoluble samples; catches ~80–90% of hidden-set overfit before submission. | Tony Li + Cona + Paritosh (704762); Luke G (692892) | Indirect — protects gains from #1/#2 (10 of 40 nightly submissions fail hidden set otherwise, 703914) | Low–medium one-time build | Double-edged: some solutions fail synthetic but pass Kaggle — treat as advisory, arbitrate via #5 |
| 7 | **Brute-force minimal-DAG enumeration as a cheap second track.** Dumb enumeration of small op-chains that reproduce the current solution's output bit-exactly; agent only steers and verifies. | 703914 ("many micro-gains, far cheaper in tokens") | Many +0.1–0.5 micro-gains across mid-pack; low token cost | Medium one-time tool build | Diminishing returns; needs exact-output verification harness |
| 8 | **Submission hygiene + value_info patching for dynamic slices.** Every Conv-family op (Conv/ConvInteger/QLinearConv/ConvTranspose/DeformConv) gets bias with len == out_channels; zero `dim_param` anywhere incl. outputs; patch constant output shapes of runtime-positioned Slices into value_info; verify with ORT 1.26.0 + `mlas.disable_kleidiai=1` (ARM64), multiple same-process runs. | HOST confirmations: 699840, 702485, 705448; community fix 704900/695972 | Prevents silent 0-scores and bundle contamination of *clean neighbors* (702256); unlocks crop tricks | Low — a pre-submit lint script | None; pure defense + enabler |

Lower-priority / watchlist: bounded top-left crop sizing for floor-bound tasks (open question,
705373); runtime-for-cost tricks (yash's team research focus, no recipe public, 704982);
end-of-comp team merges to combine per-task bests (Chan Kha Vu SPEC, 701942).

---

## 3. Live scorer quirks — exploit or avoid

| Quirk | Status | Action |
|-------|--------|--------|
| **Order-sensitivity / zip-member contamination** (699840, 702256): ORT reuses scratchpad memory across tasks in one process; a malformed Conv (bias len < out_channels) poisons *later, perfectly valid* models. Same zip can score differently twice. | HOST: root cause confirmed; **will NOT be fixed** ("fix your convolutions"); upstream: microsoft/onnxruntime#28654. Residual non-bias reproducers existed as of 05-23 (702256). | **Avoid being a victim**: explicit correct-length (zero) biases on every Conv-family op. **Exploit is sanctioned**: trying member orders until a bundle passes is final-LB-safe per host (705448). |
| **Single-file vs full-bundle score differences** (702256: task262 = 18.60 alone, 0.42 in bundle) | Same ORT bug. | Bisect bundles to find 0-scoring tasks; use single-task zips as per-task ground truth (703282). |
| **"Your network performance could not be measured"** (702485, 704900) | HOST: any `dim_param` (even on outputs) → `calculate_memory()` returns None → `TypeError: '<' not supported between NoneType and int`. | Pre-submit lint: assert every tensor shape is fully static `dim_value`; patch constant Slice output shapes into value_info (695972, host-endorsed). |
| **max(static, runtime) shape per intermediate** (705448, HOST) | Live. | Never over-declare shapes; verify runtime shapes match declarations locally. |
| **Grids >30x30 ignored by scorer** (703200, COMMUNITY unchallenged) | Live. | Legally skip >30x30 handling; save bytes. |
| **Inputs always top-left anchored, ≤30x30** (705448, HOST) | Live. | No translation invariance needed; bounded top-left crops are safe. |
| **Public LB = final; no re-evaluation** (705448 HOST, 703200) | Live. | Optimize against the live scorer, not local ARC-GEN distributions. LB-overfit is a valid strategy. |
| **100 submissions/day** (703112, staff-confirmed after the 5/day incident) | Live. | Budget ~dozens/day for single-task probes + nightly batches. |
| **Undocumented scoring timeout** (703286, unanswered; Kaggle 30-min runtime per submission per 705448) | Risk. | Keep full-bundle runtime well under 30 min; profile slowest tasks (Tony's slow-10: 358, 350, 212, 335, 246, 022, 375, 009, 074, 070). |
| **Mid-comp scorer/utils updates can silently shift baselines** (699840, Ali: 6096.94 → 6071.01; agent rejected real improvements for a day) | Happened once. | Re-validate baseline score after any `neurogolf_utils.py` change before trusting deltas. |
| **Stale bundled `score_network`** (705373) | Trap. | Ensure our local scorer = `max(1, 25 − ln(memory + params))`, MACs and I/O tensors excluded. (Run-012 data suggests ours already matches — confirm once.) |

---

## 4. Tools to adopt

1. **Open-source ONNX-builder web GUI** — discussion/699429 (Kaggler's open version of Chris
   Deotte's Codex-built GUI, 699313): view ARC puzzles, click-to-add nodes, drag/connect, edit
   params, run and see output grids, download .onnx. Best-in-class for manual rewrites of
   priority-queue tasks. **Pull into our docs/tooling set.**
2. **Agent harness features proven at ~6580** (703914) — adopt selectively:
   - per-node param/memory breakdown of the current solution (we have the census; make it per-task, live),
   - run model and **visualize intermediate tensors as grids** (debugging compact rewrites),
   - pre-submit verification identical to pipeline checks,
   - **embedding-based semantic search over our own attempts/rules/tool catalog** ("biggest
     win — stops agents reinventing scripts"),
   - candidate cost + delta vs live baseline; staged submit pipeline with per-task Kaggle verdict wait,
   - attempt-history journals so repeated passes (~5/task) see prior failures.
   - Agent role split: researcher (solves tasks) / developer (improves harness) / observer
     (read-only health). High/xhigh reasoning required; Codex = stable grinder, Claude =
     smarter but disobedient.
3. **Benchmark notebook with 10 optimized ONNX files + case studies** (task316/task354):
   https://www.kaggle.com/code/jsrdcht/glm-vs-opus-onnx-cost-opt-neurogolf-2026 (703462) —
   reference material for what deploy-safe rewrites look like.
4. **google/arc-gen** generators (692892) — build the synthetic holdout (method #6).
5. **ORT 1.26.0 + `mlas.disable_kleidiai=1`** for faithful local repro of the scoring env on
   ARM64 (699840, HOST).
6. Codex macOS app Settings → Profile shows daily token usage (704568) — for budget tracking.
7. Community per-task score spreadsheet proposal (703232) — monitor, low value so far.

---

## 5. Open questions to answer experimentally

1. **Is our local scorer exactly `max(1, 25 − ln(memory+params))` with I/O excluded and
   max(static, runtime) sweep?** One-time audit vs 3–5 single-task Kaggle probes. (Run-012
   histogram suggests yes, but task310-style local-vs-real gaps burned others.)
2. **Crop sizing for floor-bound tasks**: hidden inputs are top-left anchored and ≤30x30 —
   per task, what is the actual hidden size range, and how small can a bounded top-left crop
   be before hidden tests fail? (Raised, unanswered in 705373.) Probe via single-task subs.
3. **Residual non-bias bundle contamination** (702256: task317 had no bias issue yet
   contaminated): does our current bundle have any task whose single-file score ≠ in-bundle
   contribution? Bisect once at our next full submission.
4. **Where is the scoring timeout?** (703286 unanswered.) Measure our full-bundle runtime;
   establish margin before spending runtime on cost tricks (yash-style).
5. **Runtime-for-cost mechanics**: what concretely does "leveraging runtime for cost
   reduction" buy under max(static, runtime) scoring? (704982 names the direction, no recipe.)
6. **Synthetic-holdout calibration**: what fraction of our candidate improvements that pass
   public ARC-GEN fail (a) our synthetic set, (b) Kaggle? (Tony: 80–90% catch rate; 703914:
   ~10/40 nightly rejects.) Needed to set agent-loop accept thresholds.
7. **Which of our 170 sub-15 tasks have proven-attainable targets?** Cross-reference Tony's
   bottom-15 exact scores (704006) and robga's lowest-4 totals (704570) against our list;
   for the rest, use last-year code-golf byte lengths as difficulty priors (Tony's
   target-anchor prompting technique).
8. **Insoluble ARC-GEN samples**: which tasks natively generate insoluble samples (Cona,
   704762), and are any of our "local failures" actually that?
