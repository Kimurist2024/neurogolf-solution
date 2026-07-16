# All-authority Einsum two-factor precontraction scan 182

## Result

The immutable8009.46 archive was scanned for pairs of initializer operands in
the same Einsum whose private shared index can be contracted in advance with a
smaller resulting tensor.

- 400/400 members scanned;
- 524 structural pair occurrences across60 tasks;
- 172 pairs have integer-valued source factors;
- zero pairs have both source initializers single-use;
- no candidate is auto-adopted from the inventory alone.

The largest integer opportunity is task328: four `CoreB[5,4,4] * e[5]`
occurrences each reduce84 apparent elements to20. However `CoreB` has eight
uses and `e` four, so contracting only those four occurrences adds a tensor
without removing `CoreB`; whole-factor coverage is required. This target was
forwarded to the dedicated exact task328 lane175.

Other high-count integer families are already parameter-efficient shared
factorizations. For example task304 stores `H[30,3]`, `SF[3,2]`, and
`SG[3,2]` in102 elements; replacing all16 H-selector pairs with the two dense
products would require120 elements. Per-occurrence savings therefore do not
imply a graph-level saving when a factor is reused.

`scan.json` contains every occurrence, labels, element counts, graph use
counts, dtype/integer status, and authority hashes. Root submission and score
ledgers were not modified.
