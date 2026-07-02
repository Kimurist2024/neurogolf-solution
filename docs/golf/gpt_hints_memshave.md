# MEMORY-SHAVE MODE — incumbent-preserving memory-footprint golf

THIS TASK'S INCUMBENT NET IS ALREADY FUNCTIONALLY CORRECT. Do NOT rebuild the
rule from scratch unless every transformation below is exhausted. Your job is to
CUT THE MEMORY FOOTPRINT of the incumbent while keeping outputs bit-identical.

Scorer memory fact (scripts/lib/scoring.py:calculate_memory):
  memory = sum over EVERY intermediate tensor (node outputs) of
           num_elements * dtype_itemsize   [static shape, strict inference]
  - tensors literally named "input" / "output" are EXCLUDED (free!)
  - initializers are NOT memory; they count as PARAMS = element COUNT (not bytes)
  - runtime trace can only RAISE a tensor's counted size, never lower it

Ranked transformation list (verify each step, keep only strict cost wins):

1. DTYPE NARROWING IN PLACE. Rewrite the compute chain so intermediates use the
   smallest dtype that holds the values (bool/int8/uint8 > f16/int16 > f32/int32
   > int64). NEVER add a Cast just to narrow — a Cast output is a NEW
   intermediate and usually a net loss. Change the dtype OF the producing ops
   themselves (and their initializers; initializer dtype is free for params).
2. NODE FUSION / CHAIN COLLAPSE. Every eliminated node output removes its whole
   footprint. Fuse arithmetic chains into fewer ops (single Einsum, combined
   Conv with folded bias, merged Where/Mul/Add). Keep Einsum operand count < 15
   (>= 15 hangs local ORT). The FINAL node's output named "output" is free —
   ensure the biggest terminal tensor IS the graph output, not followed by a
   trivial Cast/Reshape whose input gets counted.
3. CONSTANT FOLDING -> INITIALIZER. Any subgraph independent of "input" produces
   a constant: precompute it and store as an initializer. Converts bytes ->
   element-count (4:1 win for f32/int32 tensors, 2:1 for f16, neutral for
   uint8). Reverse direction (closed-form regeneration) only if elements >>
   bytes saved.
4. EARLY CROP / ROI. Slice down to the region of interest as the FIRST op so
   every downstream intermediate is small. Chained diff-1 CenterCropPad passes
   are shift-free and let you shrink cloak bases below 29.
5. SHAPE DIET. Squeeze size-1 dims early, avoid broadcasting f32 [1,10,30,30]
   monsters, prefer per-channel [1,1,30,30] or coordinate vectors.

HARD safety rules (unchanged from the standard loop):
- outputs must be bit-identical decisions: official decode wrong=0 on
  train+test+arc-gen AND raw values keep margin (nothing in (0, 0.25)).
- fresh generator instances k=30, 0 failures (scripts/verify_fix.py --k 30).
- if the incumbent uses Conv/ConvTranspose, bias length MUST equal out_channels
  (flip-mine prevention) — fix it for free while you are in the graph.
- banned ops and static shapes as per the standard rules; opset domains '' /
  ai.onnx only; file <= 1.44MB.

ADOPTION LOOP (differs from rebuild mode): promote every strictly-cheaper
correct net via try_candidate.py, then KEEP SHAVING the same task. Exit only
when (a) cost <= the campaign GOAL, or (b) two consecutive transformation
attempts fail to produce a strictly cheaper adopted net, or (c) timeout.

---- GRADER FACTS + SERVER-VALIDATED TRICKS (intel 2026-07-02, forum/top teams) ----

Grader environment (organizer-confirmed): ORT **1.24.4 pinned** + ORT_DISABLE_ALL,
onnx==1.21.0, numpy==2.4.4, single process, zip-order evaluation.
- Ops introduced in ORT >= 1.25 FAIL on the grader. Op ceiling = 1.24.4.
- ORT_DISABLE_ALL means the server does NO constant folding: any constant
  subgraph you leave in the graph is charged full memory — fold it yourself
  into an initializer (lever #3 above, confirmed).
- Banned set (verified wider than rules page): Loop/Scan/NonZero/Unique/Script/
  Function + **Compress** + all *Sequence* ops + **If** (GRAPH attr) + custom
  domains + multi-input/output + dynamic dims + tensor names containing
  "kernel_time". TopK is NOT banned (LB-verified) but avoid uint8 TopK inputs
  (Kaggle ORT rejects) and verify any TopK net end-to-end locally.
- memory per tensor = max(static shape bytes, runtime-profile bytes): a tensor
  whose runtime shape exceeds its declared static shape is charged the LARGER
  value — never rely on understating static shapes.

Server-validated cost tricks (from LB-proven notebooks; verify per net):
A. INITIALIZER REUSE IS COUNTED ONCE. Feed the same initializer into multiple
   inputs of one node (or several nodes) — params charged a single time.
   Example: single Einsum 'ra,ai,zcij,bj,sb->zcrs' with U[30,5],S[5,30] each
   fed twice = rank-5 row+col remap at 300 params, memory 0 (971->300 win).
B. STATIC value_info LEGALIZES DATA-DEPENDENT Slice/Pad. A Slice whose starts
   come from ArgMax etc. passes checker+profiler if you attach a static-shape
   value_info to its output (extent must be a fixed constant). Lets you crop a
   bbox straight off the FREE "input" tensor and Pad back at the end
   (7985->4171 win). Prove the static extent from the generator first.
C. TERMINAL GridSample WITH fp16 GRID. fp32 input + fp16 grid [1,30,30,2] is
   legal on ORT 1.24.4; gather+mask+zero-pad collapses into ONE node whose
   output is "output" (free). Out-of-range coords set to constant 3.0 via
   Where -> padding_mode='zeros' zeroes them automatically (10770->5606 win).
D. ConvInteger u8 RENDERER. u8 codes + x_zero_point=1 gives effective {-1,0,1}
   weights at u8 param cost; i32 terminal output grades fine via >0 threshold;
   pads attribute (free) zero-fill decodes as background. W may even be a
   runtime-computed tensor (Equal->Cast u8->Add), i.e. data-dependent weights
   are legal (1472->688 win).
E. uint8 ARITHMETIC NEEDS opset >= 14. Add/Mul etc. accept uint8 only from
   opset 14 (rejected in 10-13). Set opset accordingly and keep whole chains
   in uint8 instead of Cast-ing back to f32 (memory /4). Opset/IR version is
   NOT restricted by the host.
F. ITERATED MaxPool FLOOD-FILL BEATS closed-form transitive closure: a 900x900
   reachability intermediate (~3.2MB) loses to H+W unrolled small steps. MACs
   and node count are FREE — only params + live tensors cost.

Dead ends (measured by others — do not waste time): INT4/sub-byte packing
(params = element COUNT, not bytes), sparse_initializer (sanitizer breaks /
dense memory charged anyway), fp16-everything without margin checks.
