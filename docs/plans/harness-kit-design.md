# EvalMine harness kit: adopted design

The music_analytics handover identified a real gap: producers recreated EvalMine's
external-artifact contract in application code, so failures arrived only at import time
and schema drift was easy. EvalMine now ships a small TypeScript package at
`packages/harness-kit`. It sits after generation and before
`evalmine experiment import`: it overlays completed records, validates comparison
semantics, and writes a hash-pinned bundle. It never calls a model, judges an output,
sets a spend budget, queries a ledger, or builds a report.

## Package and modules

`@evalmine/harness-kit` is an ESM package for Node 18+ with zero runtime dependencies.
TypeScript and Node types are development-only. Its public surface is:

- `types.ts`: the external manifest, record, condition, receipt, and JSON types.
- `correlation.ts`: deterministic, readable, hash-suffixed correlation IDs capped at
  120 characters.
- `prompt.ts`: an optional deterministic formatter for actual shared input plus a
  concise description of what varies.
- `preflight.ts`: record-shape, manifest-shape, and comparison-grid checks.
- `builder.ts`: ordered layers, explicit exclusions, and last-layer-wins overlay.
- `write.ts`: create-new-directory bundle output, canonical JSONL, SHA-256, and the
  required `evalmine-import.yaml` (JSON is deliberately used as the YAML-compatible,
  dependency-free serialization).

The Python schemas remain the import authority. The kit intentionally has a stricter
producer profile: `ranking_style` and a `correlation_id` are explicit. The Python v1
record schema adds `correlation_id` as optional so existing bundles remain valid.

## Contract rule map

| # | Handover rule | Kit enforcement |
|---|---|---|
| 1 | Manifest is required | The writer always emits the sole canonical manifest and refuses an existing destination. |
| 2 | `item_id` identifies the compared item, not an arm | The API keeps item and condition fields separate and documents the invariant. This cannot be proven from opaque strings, so no unreliable substring heuristic is claimed. |
| 3 | Every block has at least two arms | Preflight fails loudly after overlays and exclusions. A block is correctly keyed by `(lane, item_id, account_id)`, matching EvalMine rather than the handover's shortened tuple. |
| 4 | One real shared prompt per block; rendered arm prompts stay out | Preflight requires byte-identical non-empty prompts within a block; `sharedPrompt()` formats shared input and “what varies.” Whether supplied prose leaks an arm remains a producer responsibility because a generic library cannot infer that meaning. |
| 5 | N-way disables pairwise and position swap | Manifest preflight requires both flags false for N-way, and both true for pairwise. |
| 6 | Re-export is last-run-wins | Ordered `addLayer()` calls replace only the same full record key `(block, condition)` and report the dedupe count. |
| 7 | Cost is attributable per artifact | Kit records require a deterministic `correlation_id` no longer than 120 characters and preserve independent estimated, ledger, and dashboard receipts. The kit does not estimate, reconcile, or query spend. |

The source app's condition registries, exclusion lists, model calls, retry/budget
logic, database queries, judge, metrics, and HTML report are explicitly not extracted.
The duplicate `manifest.json` suggested by the source implementation is also omitted:
two canonical manifests create drift without adding an importer contract.

## Test plan

Node's built-in test runner uses synthetic records only. Tests cover each rule above,
including the honest structural boundary for rules 2 and 4, condition consistency,
declared fields, exclusions that create a singleton, overlay key correctness, receipt
bases, path refusal, and exact artifact hashing. A generated synthetic N-way bundle is
then imported by the real Python importer in CI and locally, proving cross-language
round-trip compatibility without a provider call or music_analytics data.
