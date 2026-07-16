# Known caveats

- The local metric closely mirrors the published competition utility, but hidden examples can still change runtime tensor maxima or reveal generalization failures.
- Random differential tests are a strong structural check, not a proof over every possible grid.
- Some valid competition models intentionally fail on arbitrary random inputs. The tool records cases where both models fail separately from cases where only one fails.
- Raw floating-point equality can be stricter than competition correctness. The decisive task comparison is the thresholded output at `> 0`; raw equality is reported as additional evidence.
- A local cost win can still be leaderboard-negative because hidden runtime shapes differ. Isolate uncertain structural rewrites.
- Do not infer score from ONNX file size, ZIP size, node count, or FLOPs.
