# evalmine

[![CI](https://github.com/hishamalward/evalmine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hishamalward/evalmine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Leaderboards do not tell you whether a model change improves the work you actually do.
evalmine turns controlled evidence from your own tasks into a decision you can defend. It
accepts direct model suites, isolated agent episodes, or completed artifacts from an
application-owned harness; verifies their provenance; builds blind review; and calibrates
an LLM judge against human labels. If the judge has not earned trust, evalmine refuses to
present its result as headline-ready.

The project is local-first, source-installed, and pre-release. The CLI and MCP server call
the same guarded library operations.

![A deterministic evalmine suite run using the fake adapter](docs/demo.gif)

The recording uses the deterministic fake adapter: no provider call, key, or spend.

## What it does

| Evidence lane | Use it when | Entry point |
|---|---|---|
| Direct suite | EvalMine should call model APIs over a YAML task set | `evalmine run` |
| Agent experiment | You are comparing agents, models, prompts, instructions, or plugins in isolated repositories | `evalmine experiment` |
| External artifacts | Your application owns generation and passes completed, sanitized records to EvalMine | `evalmine experiment import` |

A controlled workflow runner is also available for reproducible fixture, fan-out, and
artifact jobs. Outputs stay local; experiment, import, and workflow evidence is
hash-verifiable, and review or decision reports are self-contained HTML.

Direct adapters are included for Anthropic, OpenAI, Google, and OpenRouter. A deterministic
fake adapter exercises the complete suite and reporting path without network access.

## Quick start

```bash
git clone https://github.com/hishamalward/evalmine.git
cd evalmine
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,mcp]"

evalmine validate examples/everyday-eight.yaml
evalmine run examples/everyday-eight.yaml \
  --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash \
  --fake
```

Python 3.10 or newer is required. The fake run is deterministic and spends nothing. Its
report demonstrates the central rule: a win rate remains visibly uncalibrated when human ↔
judge agreement is below the suite's floor.

For a real run, export only the provider credentials you need and set an explicit cap:

```bash
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

evalmine run examples/everyday-eight.yaml \
  --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash \
  --max-cost 0.50
```

Use `OPENAI_API_KEY` for `openai/...` models and `OPENROUTER_API_KEY` for
`openrouter/...` models. OpenRouter is a dedicated adapter: nested catalog slugs, provider
pins, response-reported charges, and routing options remain part of the evidence and cache
identity.

## Agent experiments

A version-2 manifest pins a seed repository and declares the arms, episodes, repeats,
isolation policy, configuration treatments, validators, and evaluation method.

```bash
evalmine experiment validate examples/agent-model-comparison.yaml
evalmine experiment plan examples/agent-model-comparison.yaml
evalmine experiment prepare examples/agent-model-comparison.yaml \
  --out /tmp/evalmine-runs

evalmine experiment preflight <prepared-dir>
evalmine experiment execute <prepared-dir> --allow-provider-calls
evalmine experiment check <prepared-dir>
evalmine experiment report <prepared-dir>
```

Preparation and preflight launch no agents. Execution uses the authentication already owned
by Claude Code, Codex CLI, or Gemini CLI. Each arm receives an isolated workspace and fresh
session; subsequent turns resume only within that run. Judging and decisions are separate,
explicit stages—see the
[experiment lifecycle](docs/spec.md#13a-version-2-experiment-preparation).

## External artifacts and the harness kit

Use the external lane when generation or application data must remain outside EvalMine. The
producer supplies a sanitized `evalmine-import.yaml` plus hash-pinned JSONL:

```bash
evalmine experiment import examples/external-artifacts \
  --out /tmp/evalmine-external
evalmine experiment verify /tmp/evalmine-external
evalmine experiment report /tmp/evalmine-external
```

Import validates the schema and comparison grid, preserves source-file and line provenance,
and makes zero model calls. EvalMine does not query the producer's database and does not
pretend to sanitize arbitrary application data: sanitization is the producer's boundary.

TypeScript producers can use
[`@evalmine/harness-kit`](packages/harness-kit/README.md) to build and preflight a bundle.
The kit accepts completed records only. It generates nothing, judges nothing, has no ledger
or database access, and has no runtime dependencies. Until its first registry release,
consume an exact-commit source snapshot or an `npm pack` tarball.

## Controlled workflows

Workflow manifests coordinate direct-argv jobs, frozen fixtures, dependencies, and captured
artifacts without weakening the direct-API cost boundary.

```bash
evalmine workflow plan examples/music-backoff-workflow.yaml
evalmine workflow run examples/music-backoff-workflow.yaml \
  --out /tmp/evalmine-workflows --allow-commands
```

Provider-marked nodes need a second provider-call gate. Arbitrary direct-API shell nodes are
refused; use a cost-capped suite for those calls.

## MCP for Claude Code and other agents

`evalmine-mcp` exposes suite, experiment, external-import, and workflow operations over
stdio. To register it in a Claude Code project:

```bash
pip install -e ".[mcp]"
cp .mcp.json.example .mcp.json
```

Restart the MCP client after adding the file. The example keeps access inside
`${CLAUDE_PROJECT_DIR:-.}` and starts with provider calls, validator commands, external
writes, and workflow commands disabled. A mutating or spending operation requires both the
tool-call argument and its matching `EVALMINE_MCP_ALLOW_*` server gate. Provider keys are
inherited from the launch environment and are not written into `.mcp.json`.

See the [MCP contract](docs/spec.md#11-mcp-surface) for the tool list, path containment,
cost ceilings, and structured refusal behavior.

## Safety and evidence rules

- Live direct calls are planned and cost-estimated before the first request. Over-cap runs
  are refused; missing cost is never silently reported as `$0`.
- Suite placeholders, unknown fields, model prices, and key-shaped literals are validated
  before execution.
- Provider calls, validator commands, workflow commands, and external writes have separate
  authorization gates.
- Experiment, import, and workflow evidence is create-once, hash-pinned, bounded, and
  independently verifiable. Credentials are removed or redacted from captured runner and
  workflow output where supported.
- External bundles must already be sanitized. They intentionally contain the prompts and
  outputs needed for review, so producers must not include private source data unnecessarily.
- Reports keep condition identity hidden during review and reveal it only in decision
  evidence.
- Judge agreement, position sensitivity, schema pass rate, objective checks, latency, and
  cost remain separate signals. A failed calibration gate cannot produce a clean headline.

## Status

Version `0.1.0` is alpha software. The suite engine, four live adapters, isolated agent
experiments, external import, workflow runner, TypeScript harness kit, blind reports,
calibrated decisions, and guarded MCP surface are implemented and covered by network-free
tests across Linux, macOS, and Windows.

The remaining release proof is operational: a representative real evaluation must be
human-labeled, pass its calibration gate, be inspected, and produce the first human-owned
`DECISIONS.md` entry. EvalMine is not a hosted UI, RAG evaluator, fine-tuning system, or
product database connector.

## Documentation

- [Specification](docs/spec.md) — normative product and safety contract
- [How it works](docs/learning/how-it-works.md) — implementation tour
- [Harness kit](packages/harness-kit/README.md) — producer-side TypeScript API
- [Examples](examples) — synthetic suites, experiments, imports, and workflows
- [Contributing](CONTRIBUTING.md) — development workflow

This README describes the present product surface. Files under `docs/plans/` record design
and implementation history; they are context, not the current user manual. Where wording
differs, the specification wins.

## Development

```bash
pip install -e ".[dev,mcp]"
python -m pytest -q
python -m ruff check src tests
npm test --prefix packages/harness-kit
```

CI runs the complete Python suite on Linux, macOS, and Windows with Python 3.10 and 3.13,
plus the TypeScript round trip and a repository-history secret scan.

## License

MIT. See [LICENSE](LICENSE).
