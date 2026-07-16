# Exact repeated-slice Einsum factor scan 152

Every dense initializer used exclusively by Einsum was checked on every axis
for an exact one-hot selector times unique-slice table factorization.  No axis
had fewer combined factor elements than the original initializer, so no model
was emitted or profiled.  Safe adoptees: 0; gain `+0.0`.  Evidence:
`scan.json` and `scan.py`.
