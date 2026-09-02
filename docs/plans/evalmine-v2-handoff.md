# Evalmine v2 handoff

> **Historical snapshot — 26 August 2026.** Paths, working-tree instructions, test counts,
> plan identities, and remaining-work notes below describe that handoff date. Use the root
> `README.md` and `docs/spec.md` for the current product and operator contract.

**Prepared:** 26 August 2026

**Repository:** `/Users/hishamal-ward/public_repos/evalmine`

**Purpose:** continue from completed implementation into operator validation without
repeating the audit, redesign, or browser-tooling detour.

## Start here

1. Read this document completely.
2. Read [evalmine-v2-implementation-plan.md](evalmine-v2-implementation-plan.md) for
   the implemented contract and remaining upstream limitations.
3. Use [evalmine-v2-planning-report.html](evalmine-v2-planning-report.html) as the
   visual product map and implementation-status report.
4. Inspect the working tree before editing. It is intentionally dirty and contains
   the completed v2 implementation. Do **not** reset, clean, discard, or overwrite it.

## Product objective

Evalmine is an evidence system for controlled comparisons of real agent work, not a
generic benchmark runner. The owner's target workflows are:

- compare multiple models within the same subscription-backed agent product;
- ablate `CLAUDE.md`, candidate instructions, and plugins in isolated workspaces;
- compare Claude Code, Codex CLI, Gemini CLI, and API-backed lanes without pretending
  those are the same experimental variable;
- reproduce the music-analytics LLM back-off: parallel arms, enrichment, objective
  checks, LLM judging, calibrated scoring, human labels, and a final decision;
- evaluate long, multi-turn work on representative repositories while preventing
  earlier conversation or repository history from contaminating later arms;
- produce self-contained HTML evidence and labeling reports.

## State at handoff

All planned v2 implementation slices are present:

- strict experiment manifests and deterministic dry-run planning;
- isolated copy/worktree materialization and treated-baseline verification;
- Claude Code, Codex CLI, Gemini CLI, and direct-API runner boundaries;
- multi-turn session continuity within a run and fresh sessions across runs;
- objective repository, command, required-file, and required-section validators;
- workflow DAGs, fan-out, frozen fixtures, artifact capture, and historical music
  round-two evidence import;
- blind pairwise HTML labeling, portable labels, position-swapped judging,
  calibration, uncertainty-aware decisions, and decision HTML;
- guarded v2 MCP operations that reuse the CLI core.

Latest network-free verification:

```text
393 passed in 28.10s
```

No Claude, OpenAI, Google, or other provider/model call was made while implementing
or verifying v2.

The model-comparison example still validates and expands deterministically to nine
runs:

```bash
./.venv/bin/evalmine experiment validate examples/agent-model-comparison.yaml
./.venv/bin/evalmine experiment plan examples/agent-model-comparison.yaml
```

Current plan identity from the last dry run:

```text
opus-working-style-c6b01ad90b21
3 arms × 1 episode × 3 repeats = 9 planned runs
```

## Critical working-tree constraint

The v2 implementation, examples, tests, plans, and browser-QA helpers are currently
uncommitted. Existing tracked files are modified and many v2 files are untracked.
This is expected; preserve all of it.

Both dogfood manifests currently declare:

```yaml
seed:
  repo: ..
  ref: HEAD
  dirty: reject
  untracked: deny
```

Therefore a real `experiment prepare` against this checkout should fail closed until
the owner chooses a seed strategy. Do not silently loosen the manifest. Present the
tradeoff and get agreement on one of these paths:

1. commit the completed implementation, then use that clean commit as the seed;
2. create a separate clean seed repository/ref;
3. deliberately change the experiment to capture a dirty patch and a narrow,
   reviewed untracked allowlist.

The first option is the cleanest controlled baseline, but committing is an explicit
owner decision.

## Browser QA and Playwright MCP

The browser-QA detour is complete.

- Global Codex MCP server: `playwright`
- Package: `@playwright/mcp@0.0.78`
- Mode: attach to an existing Chrome/Edge tab through the official Playwright MCP
  Bridge extension; it must not launch or quit Chrome itself.
- Capabilities: `vision`
- Local package/output root: `.evalmine-tools/` (ignored by Git)
- Codex desktop, CLI, and IDE use the same MCP configuration. This new session should
  load it automatically; verify tool availability before relying on it.

If the bridge requests connection approval, ask the owner to select the intended tab.
Do not repeatedly retry a timed-out approval. Do not return to installing browsers,
Safari automation, WeasyPrint, Docker, or another rendering stack unless the MCP and
the existing WebKit path are both genuinely insufficient.

Chrome-free deterministic report rendering is available from the owner's normal
terminal context:

```bash
./scripts/render-report-webkit.sh
```

The script renders and width-checks:

```text
reports/browser-qa/report-desktop.png  1440 × 13484
reports/browser-qa/report-mobile.png    390 × 28031
```

Final visual QA passed. The report has no page-level horizontal overflow; mobile
workflow nodes stack, tabs wrap, and dense cards remain readable. Browser binaries,
MCP packages, and rendered reports are intentionally ignored by Git.

Important host quirk: a browser launched directly as a child of the managed agent
process can abort during macOS AppKit registration and may produce a misleading
“quit unexpectedly” dialog. Prefer the extension-connected Playwright MCP for agent
inspection. The owner can run the WebKit script in a normal terminal when a fresh
deterministic render is required.

## Remaining product work

Implementation is complete; the next phase is operator validation and gap resolution.
Work in this order:

1. **Choose the seed strategy.** Resolve the dirty-tree constraint above without
   losing work or weakening the experiment invisibly.
2. **Capability probe only.** Confirm the installed Claude Code and Codex CLI runners,
   authentication mode, requested model identifiers, sandbox mappings, and whether
   the requested models are actually selectable. Gemini is not currently installed.
   Probing must not launch an episode or spend/provider-call.
3. **Review the first dogfood manifest.** Start with
   `examples/agent-model-comparison.yaml`, but verify the exact model identifiers with
   the owner/provider before execution. Keep the three repeats, rotating order, fresh
   sessions, isolated workspaces, and two-turn episode unless evidence justifies a
   change.
4. **Estimate and present the real run.** Show run count, concurrency, what will be
   copied, what commands/providers will be invoked, expected subscription/API billing
   basis, artifacts, stop conditions, and every required gate. Obtain explicit owner
   approval immediately before provider execution.
5. **Run, check, and report.** Execute only the approved experiment; run objective
   checks; generate the blind HTML queue; let the owner label before reveal; then run
   the configured judge only with separate approval and decide from calibrated human
   plus judge evidence.
6. **Continue with config ablation.** After the model-comparison workflow is proven,
   use `examples/agent-config-ablation.yaml` to compare current config, no config, and
   the candidate treatment under the same controls.
7. **Address known upstream gaps.** Gemini needs installation/probing. Exact
   marketplace-name plugin allowlists for Claude and per-run Codex plugin allowlists
   remain fail-closed limitations. A fresh live music back-off needs a cost-capped
   domain adapter before Evalmine should authorize its direct-provider workflow.

## Safety and evidence rules

- Planning, validation, inspection, and capability probing must make no provider calls.
- Never describe subscription usage as `$0`; record the billing basis and any usage
  data the native CLI exposes.
- Never execute an agent, validator command, workflow command, external write, judge,
  or direct-provider call without the corresponding explicit client and server gates.
- Do not inspect private provider configuration directories or copy credentials into
  experiment artifacts.
- Preserve create-once evidence envelopes and verify the baseline before attributing
  any result to an arm.
- Keep arm identities hidden during human labeling and do not let judge output replace
  owner labels or failed calibration gates.

## Useful files

- `docs/plans/evalmine-v2-planning-report.html` — visual architecture/status report
- `docs/plans/evalmine-v2-implementation-plan.md` — implementation contract and gaps
- `examples/agent-model-comparison.yaml` — first model dogfood experiment
- `examples/agent-config-ablation.yaml` — instruction/plugin ablation
- `examples/music-backoff-workflow.yaml` — offline workflow fixture
- `examples/music-analytics-round2-import.yaml` — verified historical evidence import
- `README.md` — operator commands and evidence interpretation
- `docs/spec.md` — normative behavior
- `scripts/setup-browser-qa.sh` — idempotent local Playwright MCP registration
- `scripts/render-report-webkit.sh` — deterministic desktop/mobile report rendering

## Definition of a successful next session

The next session does not rebuild v2. It verifies that Playwright MCP is available,
audits the preserved working tree, makes the seed-strategy decision explicit, performs
zero-spend runner capability probes, and prepares a concrete approval-ready first
dogfood run. It stops before any provider/model execution unless the owner explicitly
authorizes that exact run.
