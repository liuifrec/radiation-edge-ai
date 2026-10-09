# Offline multi-artifact verification (v0.12, development)

This change adds `radedge verify-many`, a **read-only** collection verifier.
It allows independently frozen NASA 53BP1 and DNAi records/packages to be
checked together using their existing public `verify_target` contracts.
It neither runs inference nor generates new endpoint estimates.

## Command

```powershell
radedge verify-many --json <artifact1.json> <artifact2.json>
```

The command always requests full artifact verification for each target.
`--no-artifacts` is intentionally not offered on the collection command.
The original one-record `radedge verify` command remains unchanged.

A single JSON object is printed, including the ordered per-target verification
reports, source manifest SHA256, and `n_targets`, `n_pass`, `n_fail`, and `ok`.
A failed verification (including missing/bad JSON) is reported as a failed
item without suppressing later targets. Duplicate normalized target paths are
rejected before any verification so one record cannot inflate pass counts.
The SHA256 of each inspected record is checked before and after its verification
and a changed record is treated as a failure.

Exit codes: `0` (all passed), `3` (one or more verification failures), and
`2` (invalid batch arguments, such as duplicates).

## Scientific boundary

The collection report is transient and path-bearing; it is **not** a newly
signed/frozen research-evidence manifest. It does not aggregate individual
fibers into biological replicates, assess condition effects, match FP512 and
KL720 objects, establish biological equivalence, or profile KL720 performance.
As with the underlying verifiers, a successful result verifies the declared
provenance/integrity contracts, not scientific validity beyond their scopes.

The first purpose is to make upcoming multi-field DNAi comparisons safer and
easier to audit without touching frozen v0.11 source records.
