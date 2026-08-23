# evalmine

A public, Python eval harness that scores a model change against your own tasks —
pairwise LLM-judge win-rates calibrated to your labels, schema-pass rate, latency
and cost — and writes a versioned report. It answers one question about a model
change: did it help me, hurt me, or cost me more for the same result? It refuses
to print a win-rate as a headline when it cannot show that its judge agrees with
you.

**Status: in progress.** The core (suite loader, cache, prices, cost guard,
judge protocol, metrics, reports, CLI) runs against the built-in fake adapter.
The three real provider adapters and the MCP surface are not built yet.

Spec: [docs/spec.md](docs/spec.md).

```
pip install -e ".[dev]"

evalmine validate examples/everyday-eight.yaml

# --fake routes every model string to the deterministic fake adapter:
# no key, no network, no spend. These two model strings are the ones the
# example suite's twelve human labels refer to, so the run exercises the
# judge-calibration path end to end.
evalmine run examples/everyday-eight.yaml \
  --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash --fake
```

The report lands in `reports/everyday-eight/<run-id>/report.md`.

MIT licensed.
