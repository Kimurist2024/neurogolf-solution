# Binary Add/Sum carrier scan 141

- Scope: all 400 immutable `8009.46` authority payloads.
- Method: enumerate binary `Add` nodes and profile the schema-equivalent
  two-input `Sum` carrier, and enumerate two-input `Sum` nodes and profile
  `Add`.  Candidates were required to load and to be strictly cheaper than
  their authority before any semantic admission.
- Profiles checked: 113.
- Strictly cheaper profiles: 0.
- Runtime/semantic audits required: 0.
- Safe adoptees: 0; projected gain: `+0.0`.

The complete machine-readable result is `scan.json`; the reproduction driver
is `scan.py`.  The protected root submission and score ledgers were not
modified.
