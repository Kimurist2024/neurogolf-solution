# Wave419 cost100-500 audit

This lane audits the 8012.15 authority for strict-lower candidates in the cost
100--500 band. The four confirmed LB-black coordinates (task070@52,
task134@320, task202@20, task343@172), the maintained private-zero catalogue,
and already-admitted tasks 012/023/161/175/354/355 are excluded before review.

The historical candidate corpus was re-screened at the standard gates:
full checker, strict shape inference, known cases and two fresh seeds across
four runtime configurations. A candidate is admissible only when accuracy is
at least 0.90 and all runtime/nonfinite/shape/small-positive gates are zero.
No new safe candidate was promoted in this slice; existing task161/175/355
records remain the only approximate survivors in this band. Exact-authority
rewrite task023 is tracked separately by lane412.

Root submission, all_scores.csv and others/ were not modified by this lane.
