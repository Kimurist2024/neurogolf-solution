#   Compact ONNX patterns for dynamic corridor / empty-rectangle fill?

- Topic ID: 704769
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/704769
- Author: Harish (@xujinghang111)
- Posted: 2026-06-06T03:05:34.529768900Z
- Votes: 0
- Total messages: 

## Body

Hi all, I’m exploring compact ONNX representations for ARC-style tasks where the rule seems to identify large empty corridors or border-touching empty rectangles, erode/trim them, and fill selected zero cells.

  I’m trying to avoid Loop/Scan/NonZero and keep node count low. So far I’m considering patterns based on ReduceSum over row/column masks, MaxPool-style morphology, and Where-based fills, but maximal-rectangle style selection tends to explode in node count if implemented naively.

  Do people have general low-node ONNX patterns for:
  1. detecting long empty row/column corridors,
  2. trimming/eroding corridor endpoints,
  3. selecting border-connected empty regions,
  4. avoiding overfitting to public examples?

  Not asking for task-specific solutions, just general ONNX golfing patterns or tradeoffs people found useful. Thanks!

## Comments (0)

(no comments)
