# Boolean Where/Cast scan 146

Four exact `Where(bool,1,0)` or inverse forms were profiled as `Cast` or
`Not+Cast`.  Full checking and strict inference passed, but no candidate was
strictly cheaper.  Safe adoptees: 0; gain `+0.0`.  Evidence: `scan.json` and
`scan.py`.
