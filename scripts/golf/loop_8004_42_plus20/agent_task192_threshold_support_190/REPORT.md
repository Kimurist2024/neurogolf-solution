# task192 fixed-threshold support analysis

## Outcome

No fixed threshold is exact on the full support of generator
`task_7e0986d6`. Among k=26..36, **k33 is empirically best** on the exact two
requested fresh streams: 9,995/10,000 = **99.95%**. It is suitable only for a
normal policy90 lane, not policy100 and not a generator-support guarantee.

No ONNX file, root submission, score ledger, or `others/71407` artifact was
modified.

## Selector condition

Let:

- `B` be the box-color count after distractor overwrites;
- `D` be the distractor-color count after `remove_neighbors`;
- `k` be the HardSigmoid threshold.

For an integer histogram count `c`, `HardSigmoid(c-k, alpha=1)` is positive
exactly when `c>k`. Because the only non-background colors are box and
distractor, the selector is exact iff

```text
B > k  and  D <= k.
```

If `B<=k`, the required box color is missing. If `D>k`, the isolated
distractor color is selected and its pixels survive the downstream local rule.
Thus either condition is a real model failure, not merely a selector mismatch.
The two explicit cases below were also run through every k26..36 ONNX candidate
and all 22 candidate/case combinations were wrong with runtime0/nonfinite0.

## Formal reachable bounds

### Distractor count

`random_pixels` is row-major. `remove_neighbors` retains a sampled cell iff no
sampled left or upper orthogonal neighbor precedes it. Its output is therefore
a 4-neighbor independent set. Conversely, every independent set is reachable
by sampling exactly that set.

For a fixed `height x width` grid:

```text
0 <= D <= ceil(height*width/2), both tight.
```

Since both dimensions range from 10 to 20, the tight global range is
`0 <= D <= 200`. The upper endpoint is the 20x20 checkerboard.

### Box area and surviving box count

For a box of area `a=w*h`, at most `ceil(a/2)` mutually nonadjacent
distractors can overwrite it. Boxes have a one-cell separation, so this bound
adds over boxes. For fixed boxes:

```text
sum floor(w_i*h_i/2) <= B <= sum w_i*h_i.
```

There are 3..5 boxes and every side is 3..10. Hence the global lower bound is
three 3x3 boxes:

```text
B_min = 3*floor(9/2) = 12.
```

It is reachable by overwriting the five checkerboard cells in each 3x3 box.

For the upper bound, expand every box by half a cell on all sides. The expanded
`(w+1)x(h+1)` rectangles are disjoint inside at most 21x21. A finite DP over
the 64 allowed side pairs gives necessary area bounds:

- 3 boxes: at most 300;
- 4 boxes: at most 361;
- 5 boxes: at most 355.

The global maximum 361 is reachable in a 20x20 grid using a 10/9 split in each
dimension, with one separating row and column: areas
`10x10 + 9x10 + 10x9 + 9x9 = 361`. Thus the tight range is
`12 <= B <= 361`.

Full-support exactness would require both `k < 12` and `k >= 200`, an
impossibility for any fixed k—not just k26..36.

## Explicit reachable counterexamples

Both examples keep the box color as the unique histogram argmax, so the
failure is specifically introduced by replacing ArgMax with a fixed threshold.

### Universal false negative: B=26, D=1

- grid 10x10;
- three 3x3 boxes at `(0,0)`, `(0,4)`, `(4,0)`;
- one distractor at `(0,0)`.

The rectangles have total area 27. One isolated overwrite gives `B=26,D=1`.
For every k=26..36 the selector chooses neither color. Sampling exactly that
pixel has conditional probability `0.05 * 0.95^99 = 0.0003116068`; the exact
valid rectangle/color parameters also have positive generator probability.

### Universal false positive: B=48, D=37

- grid 20x20;
- three 4x4 boxes at row 0, columns 0/5/10;
- 37 checkerboard distractors below the boxes.

No box cell is overwritten, so `B=48>D=37`. For every k=26..36 both colors
are selected. The exact pixel-set event is extremely rare
(`5.96e-57`) but has positive probability and proves lack of support guarantee.

All parameter lists and per-candidate differing-cell counts are in
`audit/result.json`.

## Same-stream k26..36 comparison

The table uses exactly seeds `192800661` and `192930007`, 5,000 generator calls
each. “low B” is `B<=k`; “high D” is `D>k`. No sampled failure had both causes.

| k | seed 192800661 | seed 192930007 | total correct | low B | high D | accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 26 | 4982/5000 | 4985/5000 | 9967/10000 | 0 | 33 | 99.67% |
| 27 | 4991/5000 | 4990/5000 | 9981/10000 | 1 | 18 | 99.81% |
| 28 | 4995/5000 | 4994/5000 | 9989/10000 | 2 | 9 | 99.89% |
| 29 | 4997/5000 | 4996/5000 | 9993/10000 | 2 | 5 | 99.93% |
| 30 | 4997/5000 | 4997/5000 | 9994/10000 | 2 | 4 | 99.94% |
| 31 | 4997/5000 | 4997/5000 | 9994/10000 | 4 | 2 | 99.94% |
| 32 | 4997/5000 | 4997/5000 | 9994/10000 | 4 | 2 | 99.94% |
| **33** | **4998/5000** | **4997/5000** | **9995/10000** | **5** | **0** | **99.95%** |
| 34 | 4996/5000 | 4995/5000 | 9991/10000 | 9 | 0 | 99.91% |
| 35 | 4995/5000 | 4992/5000 | 9987/10000 | 13 | 0 | 99.87% |
| 36 | 4994/5000 | 4990/5000 | 9984/10000 | 16 | 0 | 99.84% |

k31 exactly reproduces the independent ONNX audit's 4997/5000 on each seed.
k33 has five failures; its Wilson 95% failure-rate interval is approximately
`0.0214%..0.1170%` (accuracy `99.8830%..99.9786%`). This strongly clears the
normal 90% policy but cannot justify 100% or “guaranteed.”

## Reproduction and guards

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/golf/loop_8004_42_plus20/agent_task192_threshold_support_190/analyze_support.py
```

Evidence: `audit/result.json`. Root hashes remained the 8009.46 authority, and
the `others/71407` tree digest remained
`cfc9471b3a5aec68bf20a11870cde5abe4d23ec35d793c0d02f5ccad1bce8b54`.
