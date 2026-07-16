# Cost-Gap Analysis vs 7700-Point Leaders (run-012)

- Our total: **6347.82** (avg 15.870/task, avg cost 29,727, median cost 15,230)
- Leaders: **7700** -> avg 19.250/task -> implied avg cost **e^5.75 = 314**
- Gap: **1352.18 points**

## Cost deciles

| p0 | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 | p100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 680 | 1,406 | 5,011 | 8,897 | 15,230 | 24,620 | 35,333 | 53,670 | 77,594 | 362,131 |

## Score histogram (floor bins)

| score bin | tasks |
|---:|---:|
| [12,13) | 5 |
| [13,14) | 67 |
| [14,15) | 98 |
| [15,16) | 77 |
| [16,17) | 58 |
| [17,18) | 26 |
| [18,19) | 41 |
| [19,20) | 12 |
| [20,21) | 1 |
| [21,22) | 9 |
| [22,23) | 4 |
| [25,26) | 2 |

## Cost buckets

| bucket | tasks |
|---|---:|
| <= 100 | 16 |
| <= 314 | 23 |
| <= 1000 | 68 |
| <= 5000 | 119 |
| <= 20000 | 221 |
| <= 50000 | 310 |
| > 50000 | 90 |

## Scenario table — reduce every task with cost > T to exactly T

| target T | score@T | tasks affected | total (all affected) | gain (all) | total (top-50 only) | gain (top-50 only) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 20.395 | 384 | 8187.01 | +1839.19 | 6692.34 | +344.52 |
| 900 | 18.198 | 343 | 7366.22 | +1018.40 | 6582.48 | +234.66 |
| 2,000 | 17.399 | 310 | 7109.17 | +761.35 | 6542.55 | +194.73 |

## Break-even and gap distribution

- Uniform ceiling that exactly matches 7700: every task at cost **<= ~359** (total 7700.0).
- Gain-to-314 (leader-average cost) by cost quintile — the gap is spread across the
  whole distribution, NOT concentrated in the tail:

| quintile (cost range) | gain if all reduced to 314 |
|---|---:|
| Q1 (0 .. 1,395) | +51.2 |
| Q2 (1,409 .. 8,704) | +214.3 |
| Q3 (9,025 .. 24,607) | +310.3 |
| Q4 (24,639 .. 53,661) | +380.8 |
| Q5 (53,706 .. 362,131) | +446.3 |

Implication: the top-50 queue is the highest *per-task* payoff, but matching the
leaders requires re-synthesizing essentially the whole mid-pack (median cost 15,230)
down to a few hundred bytes per task, not just fixing the worst 50.

## Whole-set census (optimized_pre_merge, 400 models)

- Total nodes: 32,484; total intermediate memory 11,608,090 B; total params 243,457
- Top ops: Cast x3649, Mul x3258, And x2867, Add x2164, Sub x1735, Where x1618, Gather x1349, Equal x1335, Greater x1311, Slice x1117, ReduceSum x951, ReduceMax x853, Max x839, Reshape x814, Min x776

## Priority queue — 50 highest-cost tasks

| task | cost | score | memory | params | nodes | interm. | top ops |
|---:|---:|---:|---:|---:|---:|---:|---|
| 255 | 362,131 | 12.20 | 348,789 | 13,342 | 649 | 648 | Castx124, Maxx106, Mulx73, Equalx61, ReduceMaxx58, Convx57 |
| 101 | 223,753 | 12.68 | 220,744 | 3,008 | 152 | 151 | Mulx32, Convx31, Castx23, Addx17, Equalx16, Reshapex8 |
| 133 | 191,451 | 12.84 | 191,249 | 202 | 420 | 473 | Mulx78, Castx72, Subx41, Addx39, Divx18, Floorx18 |
| 158 | 189,716 | 12.85 | 189,350 | 366 | 249 | 248 | Castx28, Mulx27, Reshapex19, Gatherx18, Slicex18, Padx18 |
| 96 | 167,632 | 12.97 | 167,523 | 107 | 242 | 241 | Subx26, Castx24, Mulx24, Addx21, Constantx19, Andx18 |
| 286 | 147,163 | 13.10 | 145,879 | 1,284 | 149 | 148 | MaxPoolx59, Minx59, Mulx10, Castx7, ReduceMaxx3, Subx3 |
| 367 | 141,356 | 13.14 | 141,212 | 144 | 135 | 134 | Gatherx24, Addx16, Castx15, Subx13, Wherex12, Equalx6 |
| 233 | 126,063 | 13.26 | 125,010 | 1,053 | 518 | 520 | Andx61, Reshapex50, Mulx46, Lessx36, Castx35, Addx34 |
| 285 | 123,734 | 13.27 | 121,246 | 2,488 | 1533 | 1533 | Addx228, Andx223, Mulx205, Castx177, Maxx126, Minx120 |
| 18 | 116,038 | 13.34 | 112,970 | 3,068 | 1391 | 1395 | Addx174, Mulx157, Castx154, Subx144, Andx139, Minx84 |
| 209 | 113,834 | 13.36 | 113,691 | 143 | 198 | 197 | Castx21, Subx21, Wherex17, Andx17, Equalx15, Addx14 |
| 118 | 105,444 | 13.43 | 105,304 | 140 | 66 | 65 | Andx11, MaxPoolx10, Convx9, Greaterx7, Castx6, Equalx4 |
| 77 | 100,907 | 13.48 | 99,840 | 1,066 | 612 | 611 | Castx108, Subx84, ArgMaxx68, Andx62, Addx56, Gatherx40 |
| 251 | 100,580 | 13.48 | 100,560 | 20 | 53 | 52 | Maxx9, Mulx9, Slicex7, Subx7, MaxPoolx7, Castx4 |
| 110 | 100,383 | 13.48 | 100,033 | 350 | 163 | 162 | Equalx29, Gatherx21, Castx15, ReduceMaxx15, Reshapex14, Andx14 |
| 366 | 98,548 | 13.50 | 97,112 | 1,436 | 538 | 541 | Andx72, Castx68, Wherex65, Reshapex56, Subx29, Greaterx25 |
| 54 | 98,391 | 13.50 | 95,932 | 2,458 | 1028 | 1029 | Mulx219, Addx149, Castx118, Andx88, Wherex71, Greaterx66 |
| 66 | 98,102 | 13.51 | 97,964 | 138 | 291 | 290 | Andx71, Orx25, Greaterx24, LessOrEqualx20, Equalx19, Wherex18 |
| 5 | 97,607 | 13.51 | 94,770 | 2,837 | 1803 | 1803 | Addx276, Andx265, Castx248, Mulx206, LessOrEqualx130, GreaterOrEqualx121 |
| 2 | 97,293 | 13.51 | 97,200 | 93 | 95 | 94 | Convx24, Greaterx23, Wherex21, Slicex11, Subx6, Castx3 |
| 29 | 97,117 | 13.52 | 96,947 | 170 | 88 | 87 | Castx12, Subx12, Greaterx9, Addx9, ArgMaxx8, Mulx7 |
| 128 | 95,024 | 13.54 | 94,940 | 84 | 28 | 27 | Reshapex4, Castx3, ReduceMaxx3, Subx3, Mulx3, ArgMaxx2 |
| 239 | 94,467 | 13.54 | 94,081 | 386 | 34 | 34 | Reshapex8, Castx5, Lessx3, ReduceSumx2, Slicex2, Addx2 |
| 173 | 94,228 | 13.55 | 92,028 | 2,200 | 496 | 496 | Andx82, Castx60, Addx60, Mulx54, Maxx30, Minx30 |
| 216 | 92,713 | 13.56 | 92,612 | 101 | 257 | 259 | Castx47, Lessx38, Gatherx22, Addx18, Unsqueezex18, Squeezex14 |
| 64 | 89,763 | 13.60 | 89,677 | 86 | 65 | 64 | Andx9, Castx8, MaxPoolx8, Greaterx8, Equalx6, Slicex5 |
| 159 | 89,321 | 13.60 | 85,216 | 4,105 | 55 | 54 | Gatherx10, Padx8, Castx7, Addx6, Subx4, Mulx4 |
| 208 | 88,044 | 13.61 | 87,522 | 522 | 199 | 198 | Andx60, Equalx22, Castx21, Greaterx18, ReduceMaxx17, Wherex17 |
| 187 | 84,612 | 13.65 | 83,700 | 912 | 85 | 84 | Andx17, Castx15, MaxPoolx14, Greaterx14, Orx12, Gatherx10 |
| 398 | 84,295 | 13.66 | 83,533 | 762 | 44 | 43 | Andx7, Slicex6, Wherex6, Equalx5, Addx4, Orx4 |
| 109 | 83,822 | 13.66 | 82,868 | 954 | 41 | 40 | Castx5, Reshapex5, Subx5, Gatherx5, ReduceSumx3, ReduceMaxx3 |
| 363 | 82,766 | 13.68 | 81,146 | 1,620 | 44 | 43 | Andx8, Greaterx6, Convx6, Castx4, Notx4, Slicex3 |
| 234 | 81,703 | 13.69 | 81,621 | 82 | 58 | 57 | Wherex12, Greaterx7, ReduceSumx6, Subx5, Castx4, ReduceMinx4 |
| 19 | 80,850 | 13.70 | 80,780 | 70 | 32 | 31 | Castx4, ReduceSumx4, Addx3, ReduceMaxx2, Modx2, Lessx2 |
| 382 | 80,131 | 13.71 | 80,032 | 99 | 104 | 103 | Castx15, Subx12, Andx9, ReduceSumx8, CumSumx8, Wherex8 |
| 76 | 79,972 | 13.71 | 77,858 | 2,114 | 1328 | 1330 | Addx222, Mulx154, Andx149, Minx110, Castx103, Maxx74 |
| 36 | 79,943 | 13.71 | 79,832 | 111 | 56 | 55 | Gatherx8, Addx8, Subx7, Reshapex6, Castx5, ArgMaxx5 |
| 201 | 79,310 | 13.72 | 79,002 | 308 | 136 | 135 | Squeezex16, Mulx15, Gatherx13, Castx12, Lessx12, Wherex11 |
| 264 | 79,212 | 13.72 | 77,040 | 2,172 | 46 | 45 | Castx10, Equalx10, Mulx9, ReduceSumx9, Convx2, Wherex1 |
| 25 | 78,021 | 13.74 | 77,960 | 61 | 61 | 60 | Slicex14, Castx9, Mulx6, Andx5, ReduceSumx4, CumSumx4 |
| 300 | 77,546 | 13.74 | 77,438 | 108 | 43 | 42 | Reshapex6, Subx5, ArgMaxx5, Gatherx5, Addx5, Castx4 |
| 364 | 76,538 | 13.75 | 75,136 | 1,402 | 62 | 61 | Wherex12, MaxPoolx10, Andx7, Slicex6, Padx5, Greaterx4 |
| 384 | 74,662 | 13.78 | 74,556 | 106 | 41 | 40 | Castx7, Reshapex4, ArgMaxx4, Gatherx4, Subx4, Addx4 |
| 157 | 74,528 | 13.78 | 73,727 | 801 | 2412 | 2411 | Andx389, Equalx233, Wherex220, Castx201, Addx180, ReduceSumx155 |
| 71 | 73,848 | 13.79 | 72,704 | 1,144 | 139 | 139 | Castx24, Subx13, Reshapex11, ArgMaxx11, Equalx10, ReduceMaxx8 |
| 175 | 73,149 | 13.80 | 70,400 | 2,749 | 21 | 20 | Castx3, Mulx2, ReduceSumx2, Addx2, Convx1, Greaterx1 |
| 107 | 72,651 | 13.81 | 58,970 | 13,681 | 52 | 51 | Wherex19, Gatherx9, Lessx7, Subx4, Mulx4, Squeezex4 |
| 379 | 71,699 | 13.82 | 69,969 | 1,730 | 107 | 106 | Greaterx16, Wherex16, Subx13, Mulx11, ReduceSumx9, Castx7 |
| 205 | 70,692 | 13.83 | 69,711 | 981 | 110 | 109 | Castx19, Reshapex12, Andx10, Equalx8, ReduceMaxx7, Slicex5 |
| 281 | 70,655 | 13.83 | 69,668 | 987 | 66 | 65 | Castx10, Reshapex6, Andx6, ReduceMaxx4, Subx4, ArgMaxx4 |

## Eyeball of top-10 priority tasks (train+test pairs)

| task | cost | family | same-size | in shapes (sample) | out shapes (sample) | colors in/out | detail |
|---:|---:|---|---|---|---|---|---|
| 255 | 362,131 | **same-size cell-wise (non-pointwise: local/structural)** | True | 30x30, 30x30, 30x30, 30x30 | 30x30, 30x30, 30x30, 30x30 | 5/6 | output shape == input shape but cell value depends on context |
| 101 | 223,753 | **same-size cell-wise (non-pointwise: local/structural)** | True | 14x12, 14x12, 14x12, 17x21 | 14x12, 14x12, 14x12, 17x21 | 3/3 | output shape == input shape but cell value depends on context |
| 133 | 191,451 | **same-size cell-wise (non-pointwise: local/structural)** | True | 16x12, 16x18, 17x18, 15x18 | 16x12, 16x18, 17x18, 15x18 | 7/7 | output shape == input shape but cell value depends on context |
| 158 | 189,716 | **same-size cell-wise (non-pointwise: local/structural)** | True | 20x19, 20x21, 21x22, 22x22 | 20x19, 20x21, 21x22, 22x22 | 6/6 | output shape == input shape but cell value depends on context |
| 96 | 167,632 | **size-changing object-level/other** | False | 13x17, 18x18, 18x18, 19x19 | 7x7, 7x7, 11x11, 11x11 | 8/8 |  |
| 286 | 147,163 | **same-size cell-wise (non-pointwise: local/structural)** | True | 11x24, 13x14, 15x15 | 11x24, 13x14, 15x15 | 6/6 | output shape == input shape but cell value depends on context |
| 367 | 141,356 | **same-size cell-wise (non-pointwise: local/structural)** | True | 12x19, 13x16, 15x17, 16x18 | 12x19, 13x16, 15x17, 16x18 | 2/3 | output shape == input shape but cell value depends on context |
| 233 | 126,063 | **size-changing object-level/other** | False | 24x19, 20x10, 16x12, 19x15 | 17x9, 8x8, 9x9, 8x12 | 7/6 |  |
| 285 | 123,734 | **same-size cell-wise (non-pointwise: local/structural)** | True | 30x30, 20x20, 14x14, 24x24 | 30x30, 20x20, 14x14, 24x24 | 8/8 | output shape == input shape but cell value depends on context |
| 18 | 116,038 | **same-size cell-wise (non-pointwise: local/structural)** | True | 14x18, 14x15, 16x14, 24x19 | 14x18, 14x15, 16x14, 24x19 | 7/7 | output shape == input shape but cell value depends on context |

Machine-readable data: `docs/research/cost-gap-analysis.json`.
