# task145 exact initializer-alias audit

Authority: task145 from immutable 8009.46 archive, SHA-256
`35cf952052882ff0198d01b64b75e7d36b2ba054b758089c6b54310559544d19`,
official cost 5129 (memory 5032, parameters 97).

The horizontal slope vectors are reversals, not duplicates.  The vertical
vectors are exact transposes of their horizontal counterparts, so two
20-element initializers can be removed by two `Transpose` nodes.  This
transformation is all-input exact and passes full checker and strict shape
inference.

Official profiling nevertheless gives cost 5169 (memory 5112, parameters 57):
the two 20-element transpose outputs add 80 scored bytes while only 40
parameter elements are removed.  The candidate is therefore strictly worse
and rejected before correctness/fresh execution.  No root or staged model was
changed.

