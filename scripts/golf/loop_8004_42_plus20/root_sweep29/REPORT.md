# 8005.16 broad initializer/Einsum sweep

Fifteen whole-payload builders scanned exact CSE and initializer reuse plus
constant-axis, diagonal, outer-product, shared/unique-constant, slice,
permutation, latent-component, unit-operand, contraction-reuse, and uniform
operand rewrites. Seventy-nine lower-cost artifacts reached cost screening.

**Safe adoptees: 0; gain counted: `+0.0`.**

- The 32 task010/028/060/175/229/232/304/315 latent-prune variants all miss
  known 100%; see `../agent_prune_wave30a/`.
- The 41 task199/070/333/165/169/328/379/013 variants either retain giant
  Einsums or fail at runtime in both required execution paths; see
  `../agent_sweep_wave30b/`.
- task163 prune variants are known 0/267 and retain a 53-input giant Einsum;
  see `../agent_new_mid22/task163_root_prunes_audit.json`.
- task048's remaining exact constant fusions belong to the private-risk lineage
  whose fresh results are below the mandatory private 100% guarantee.

Per-builder construction evidence is retained in the subdirectory manifests.
