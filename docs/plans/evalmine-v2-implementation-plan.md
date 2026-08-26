# evalmine v2 implementation plan

This roadmap turns evalmine from a stateless prompt comparator into an evidence
system for real agent work. The approved product analysis and interactive mockups
are in [evalmine-v2-planning-report.html](evalmine-v2-planning-report.html).

## Product invariants

1. The experiment unit is an **arm running an episode in an isolated workspace**.
2. Every repeat starts from the same pinned seed and a fresh agent session by default.
3. Model, instructions, plugins, runner, and authentication are independent variables.
4. Raw transcripts, diffs, validator results, judge decisions, and human labels remain
   inspectable evidence; aggregate scores never replace them.
5. Planning and validation make no provider calls. Execution requires a separate,
   explicit command with cost and mutation guards.
6. Subscription-backed agent CLIs are preferred for agent-behavior experiments;
   direct APIs remain available for controlled pipeline and cost experiments.

## Delivery slices

### Phase 1 — experiment contract and dry-run planner (implemented)

- [x] Preserve the planning report in the repository.
- [x] Define a strict version-2 experiment manifest.
- [x] Model and resolve the git seed, isolation, arms, configuration treatments,
  episodes, and evaluation.
- [x] Expand arm × episode × repeat into a deterministic, inspectable run plan.
- [x] Add zero-spend `experiment validate` and `experiment plan` CLI commands.
- [x] Add a realistic model-comparison example and automated tests.

Exit gate: malformed or biased configurations fail early; a user can inspect the
complete schedule without creating workspaces, launching agents, or spending money.

### Phase 2 — workspace and artifact substrate (implemented)

- [x] Materialize the resolved git seed and enforce tracked/untracked policies.
- [x] Create one independently disposable copy/worktree per planned run.
- [x] Apply project-instruction treatments inside each workspace and stage plugin,
  settings, and argument policy in a runner-local capsule.
- [x] Persist immutable manifest/input snapshots, environment inventory, and ledger.
- [x] Fingerprint the source repository before and after preparation and provide a
  verification gate that detects later baseline changes.

Machine-wide external-write containment remains a runner responsibility: Phase 2 records
the policy, while Phase 3 maps exact allowlisted directories into supported native
sandboxes, requires a second acknowledgement, fingerprints them before/after, or fails
closed when a runner cannot enforce the boundary.

Exit gate: two runs cannot observe one another's edits or conversation history, and
the baseline repository is unchanged.

### Phase 3 — subscription-first agent runners (implemented within current CLI limits)

- [x] Define a runner protocol with capability discovery and preflight checks.
- [x] Implement Claude Code, Codex CLI, and Gemini CLI runners.
- [x] Support multi-turn episodes by resuming only within a single planned run.
- [x] Capture structured event streams, final responses, tool activity, timing, and
  runner/model identity.
- [x] Enforce supported plugin/settings capsules and external-write sandbox policy,
  failing preflight for combinations a native CLI cannot guarantee.
- [x] Keep the existing direct-API suite engine intact as the `api-prompt` lane.

For Codex, the intended automation surface is `codex exec`: the official reference
documents it as the stable non-interactive command, with JSONL output, model and
sandbox overrides, fresh ephemeral sessions, user-config isolation, and explicit
session resume for follow-ups. Authentication remains outside manifests.

API-auth Claude arms require and receive a native per-run USD ceiling. External-write
allowlists require absolute non-broad directories, a second operator acknowledgement,
native Claude/Codex sandbox mapping, and before/after fingerprints. Claude can add
pinned workspace-local plugin directories to one session; exact installed-plugin
allowlists by name and Codex per-run plugin allowlists still fail closed because their
current automation surfaces cannot guarantee them. Gemini supports exact extension
selection. Subscription runs report their billing basis and usage without calling an
unpriced subscription `$0`. The full path uses fake process drivers in tests.

Phase-3A exit gate: the same episode can be mapped through all three agent products,
with a fresh run-local session and credential-free evidence. A real subscription-backed
dogfood run remains an explicit operator action, not an automated test.

### Phase 4 — validators and domain pipelines (implemented)

- [x] Add repository diff, test/lint command, required-file, and required-section
  validators with immutable treated-baseline evidence.
- [x] Add a pipeline runner for fan-out enrichment and dependency DAGs.
- [x] Support frozen database fixtures and application eye-test artifacts.
- [x] Port the successful music-analytics back-off workflow as an end-to-end fixture.
- [x] Add a hash-pinned importer for the real round-two fixtures, 74-row ledger, raw
  references, analyzer results, findings, and founder-facing HTML.

Exit gate: evalmine can reproduce the earlier music workflow: parallel generation,
enrichment, machine checks, blind LLM judging, scoring, and human decision.

Phase 4A adds a create-once `experiment check` envelope. Built-in checks freeze the
post-agent state before declared test/lint commands run; command validators have a
separate operator gate and record their own workspace mutations. Repository patches
are available even for copy workspaces without Git metadata. Phase 4B adds a direct-
argv workflow DAG with cycle detection, fan-out, dependency failure propagation,
separate command/provider gates, restored hash-pinned fixtures, bounded logs, captured
JSON/HTML/image artifacts, final-workspace verification, and a static workflow report.
The offline music fixture follows generate → enrich → judge → score/human-eye-test. The
real round-two companion-repository adapter additionally proves that the historical
100/250-track fixtures, all 74 ledger raw references, eight-fixture/40-arm analyzer body,
findings, and original visual report normalize into verified workflow evidence without
provider calls. Workflow copies now exclude local credentials/provider settings and large
dependency caches; fixtures can be sourced from the target root, and nodes can declare
an evidenced working directory and safe literal environment.

### Phase 5 — evidence, labeling, reporting, and MCP (implemented)

- [x] Generalize the current pairwise judge and calibration layer to episode artifacts.
- [x] Preserve position-swapped judging and Cohen's kappa against human labels.
- [x] Build a static HTML labeling queue with blind reveal, resumable local labels,
  and plan-scoped JSON export/import.
- [x] Add transcript, diff, tool/timing, validator, and final-response comparison views.
- [x] Expose plan, run, inspect, label, and report operations through MCP.

Exit gate: a CLI or agent-session run ends in a self-contained HTML evidence bundle
that supports blind human labeling and communicates the decision to others.

Phase 5A generates every within-block arm pair with a tested stable A/B mapping. The
self-contained report hides identity/configuration metadata until reveal, resumes draft
labels in browser storage, and exports portable label JSON without changing evidence.
Phase 5B adds create-once position-swapped episode judging through subscription CLIs or
the cost-capped API lane, multi-annotator label import, consensus, Cohen's kappa and
per-episode agreement, uncertainty-aware human/judge arm scores, disagreement queues,
label-completeness and calibration gates, and self-contained decision HTML. The MCP
server exposes contained plan/prepare/preflight/execute/check/report/judge/decide and
workflow operations, with server-side gates in addition to tool-call confirmations.

## Remaining operator validation and upstream limitations

- Run the first real subscription-backed dogfood experiments and label them with the
  repository owner; implementation tests deliberately make no provider calls.
- Install/probe Gemini CLI on the target machine before including Gemini arms.
- Claude installed-plugin allowlists by marketplace name and Codex per-run plugin
  allowlists remain fail-closed platform gaps. Claude workspace-local plugin directory
  additions are supported and clearly distinguished from an exact allowlist.
- The completed real music round-two evidence imports successfully. A fresh live rerun is
  still operator work: its application harness uses direct provider APIs and records cost
  after calls, but exposes no externally enforceable pre-call USD ceiling. Evalmine refuses
  to mislabel or launch that process as a guarded workflow node; add a cost-capped domain
  adapter (or route calls through the suite/API lane) before authorizing a rerun.

## Deliberate non-goals for Phase 1

Phase 1 does not launch agents, create worktrees, edit target repositories, inspect
private configuration directories, authenticate to providers, or estimate subscription
costs. Those behaviors need capability-aware preflight and artifact contracts first.
