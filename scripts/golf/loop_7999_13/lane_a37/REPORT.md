# A37 exact-shrink result

- task013: **ACCEPT_STRICT**, cost 739→731 (authority 743→731), SHA `a19721dfb3683b27613ddb176e2c367b38326a355169efbfd972491bb704f35e`.
- task105: **ACCEPT_USER_95_PERCENT_GATE**, cost 195→194 (authority 199→194), SHA `31e7878f473b9d842b5a864d410e20dd885b6ba3ccc53655c8d26cb4dff5846e`.
- Increment over the two already-safe lane candidates: **+0.016025860699**.
- Increment against `submission_base_8000.46.zip`: **+0.041729250629**, projected total **8000.501729**.
- The authority ZIP was hash-checked and not modified.

Both candidates pass full checker, strict static inference, complete dual-ORT known sets, truthful all-node runtime shapes, standard-domain and safety checks, and external differential 500/500 with `ACCEPT_STRICT`. task013 is fresh 5000/5000 on both ORT modes. task105 is fresh 4970/5000 and 4970/5000, with zero runtime errors in both modes; this is above the user-authorized 95% gate and is raw-identical to the authority baseline on the external 500-case differential.
