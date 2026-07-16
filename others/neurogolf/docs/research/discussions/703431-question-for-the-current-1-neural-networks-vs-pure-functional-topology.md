# question for the current #1: Neural Networks vs. Pure Functional Topology?

- Topic ID: 703431
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/703431
- Author: T.-C. Chang (@newlooker)
- Posted: 2026-05-31T05:48:28.616699200Z
- Votes: 0
- Total messages: 2

## Body

First of all, massive congratulations on holding the #1 spot in such a brutal competition!

I’ve been heavily experimenting with local phase transitions and spatial shifting in this dataset. Recently, my team achieved a massive score boost by completely bypassing CNNs. We discovered that for many tasks (like local object translations), the shift vectors are actually constant across the dataset. By compiling pure functional tensor math (with 0 parameters) directly into ONNX, we reached perfect accuracy with minimal Cost penalty.

Without revealing your secret sauce before the deadline, I’m incredibly curious about your macro-architecture:
Are you relying on Neural Networks (like massive Transformers/CNNs) that have genuinely learned to generalize these spatial rules, or is your top-scoring solution heavily leaning towards Program Synthesis (DSL) and deterministic rule-hardcoding similar to our pure ONNX approach?

Best of luck in the final stretch!@crodoc, I'd be very interested in your take on this if you're willing to share some high-level thoughts."

## Comments (2)

- **CroDoc** (2026-05-31T06:21:11.080Z, votes: {'canUpvote': True}):
  I think you'll have to wait until July 16th. Who knows even what happens by then. @yiheng won't stop pushing his score 😂

  - **T.-C. Chang** (2026-05-31T06:27:31.073Z, votes: {'canUpvote': True}):
    haha that's nice day!
