import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  HarnessBuilder,
  HarnessContractError,
  artifactKey,
  comparisonBlockKey,
  correlationId,
  preflightBundle,
  sharedPrompt,
  writeBundle,
  type BundleDefinition,
  type ExternalRecord,
} from "../src/index.js";

function definition(style: "pairwise" | "n-way" = "n-way"): BundleDefinition {
  const pairwise = style === "pairwise";
  return {
    external_artifacts: "synthetic-harness-test",
    question: "Which completed synthetic condition is strongest?",
    evaluation: {
      objectives: ["Correctness", "Usefulness"],
      blind: "condition",
      ranking_style: style,
      fields: ["summary"],
      human: { required: true, labels_per_pair: 1, coverage: "calibration-subset" },
      judge: {
        enabled: true,
        pairwise,
        position_swap: pairwise,
        calibrate: true,
        runner: "api-prompt",
        model: "synthetic/judge",
        max_cost_usd: 1,
      },
    },
  };
}

function record(
  conditionId: string,
  options: {
    item?: string;
    account?: string;
    prompt?: string;
    summary?: string;
    model?: string;
  } = {},
): ExternalRecord {
  const item = options.item ?? "item-1";
  const account = options.account ?? "account-redacted";
  return {
    lane: "summary",
    item_id: item,
    account_id: account,
    correlation_id: correlationId("synthetic", ["summary", item, account, conditionId]),
    prompt: options.prompt ?? "Shared synthetic input. Only the declared condition varies.",
    condition: {
      id: conditionId,
      model: options.model ?? `synthetic/${conditionId}`,
      prompt_variant: conditionId,
      width: "fixed",
    },
    output: { summary: options.summary ?? `${conditionId} result` },
    cost_receipts: {
      estimated: { usd: 0.001, source: "synthetic pinned estimate" },
      ledger: { usd: 0.0011, source: "synthetic ledger row" },
      dashboard_observed: { usd: 0.0012, source: "synthetic provider observation" },
    },
  };
}

function completeGrid(items = ["item-1"]): ExternalRecord[] {
  return items.flatMap((item) => [record("condition-a", { item }), record("condition-b", { item })]);
}

test("rule 1: writer emits the required manifest, pins exact JSONL, and refuses overwrite", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evalmine-harness-"));
  const destination = path.join(root, "bundle");
  try {
    const bundle = new HarnessBuilder(definition()).addLayer(completeGrid()).build();
    const written = await writeBundle(bundle, destination);
    const manifest = JSON.parse(await readFile(written.manifest_path, "utf8")) as {
      version: number;
      artifacts: { path: string; sha256: string }[];
    };
    const bytes = await readFile(written.artifact_path);
    assert.equal(manifest.version, 1);
    assert.deepEqual(manifest.artifacts.map((item) => item.path), ["records.jsonl"]);
    assert.equal(manifest.artifacts[0]?.sha256, createHash("sha256").update(bytes).digest("hex"));
    await assert.rejects(() => writeBundle(bundle, destination), /refusing to overwrite/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rule 2: item identity is structurally separate from condition identity", () => {
  const first = record("condition-a");
  const second = record("condition-b");
  assert.equal(comparisonBlockKey(first), comparisonBlockKey(second));
  assert.notEqual(artifactKey(first), artifactKey(second));
  assert.equal(preflightBundle(definition(), [first, second]).block_count, 1);
});

test("rule 3: singleton blocks fail after explicit exclusions", () => {
  const builder = new HarnessBuilder(definition()).addLayer(completeGrid());
  assert.throws(
    () => builder.build({ exclude_conditions: ["condition-b"] }),
    (error: unknown) => error instanceof HarnessContractError && error.code === "singleton-block",
  );
});

test("rule 4: shared prompts are real context and must be identical within a block", () => {
  const prompt = sharedPrompt({
    input: { document: "synthetic input", requested_shape: ["summary"] },
    whatVaries: "Only the model and declared prompt treatment.",
  });
  assert.match(prompt, /^SHARED INPUT:/);
  assert.match(prompt, /WHAT VARIES:/);
  assert.throws(
    () => preflightBundle(definition(), [
      record("condition-a", { prompt }),
      record("condition-b", { prompt: `${prompt}\narm-only rendered instruction` }),
    ]),
    (error: unknown) => error instanceof HarnessContractError && error.code === "inconsistent-shared-prompt",
  );
});

test("rule 5: ranking style and judge flags cannot disagree", () => {
  const invalid = definition("n-way");
  invalid.evaluation.judge.pairwise = true;
  assert.throws(
    () => new HarnessBuilder(invalid),
    (error: unknown) => error instanceof HarnessContractError && error.code === "ranking-flags",
  );

  const invalidPairwise = definition("pairwise");
  invalidPairwise.evaluation.judge.position_swap = false;
  assert.throws(() => new HarnessBuilder(invalidPairwise), /pairwise ranking requires/);
});

test("rule 6: later layers win only for the full block plus condition key", () => {
  const initial = completeGrid(["item-1", "item-2"]);
  const replacement = record("condition-a", { item: "item-1", summary: "replacement" });
  const bundle = new HarnessBuilder(definition())
    .addLayer(initial)
    .addLayer([replacement])
    .build();
  assert.equal(bundle.records.length, 4);
  assert.equal(bundle.summary.deduped_record_count, 1);
  const replaced = bundle.records.find(
    (item) => item.item_id === "item-1" && item.condition.id === "condition-a",
  );
  assert.deepEqual(replaced?.output, { summary: "replacement" });
  assert.equal(bundle.records.filter((item) => item.item_id === "item-2").length, 2);
});

test("rule 7: correlation IDs are bounded, deterministic, collision-resistant across part boundaries", () => {
  const first = correlationId("generation", ["a:b", "c", "x".repeat(300)]);
  const repeated = correlationId("generation", ["a:b", "c", "x".repeat(300)]);
  const different = correlationId("generation", ["a", "b:c", "x".repeat(300)]);
  assert.equal(first, repeated);
  assert.notEqual(first, different);
  assert.ok(first.length <= 120);
  assert.throws(() => correlationId("generation", [Number.NaN]), /must be finite/);
  assert.throws(
    () => preflightBundle(definition(), completeGrid().map((item) => ({ ...item, correlation_id: "x".repeat(121) }))),
    /at most 120/,
  );
  assert.deepEqual(Object.keys(record("condition-a").cost_receipts ?? {}).sort(), [
    "dashboard_observed",
    "estimated",
    "ledger",
  ]);
});

test("preflight catches condition drift, missing fields, and inconsistent lane coverage", () => {
  assert.throws(
    () => preflightBundle(definition(), [record("condition-a"), record("condition-a", { model: "synthetic/changed" })]),
    /inconsistent model\/prompt\/width metadata/,
  );
  const missing = record("condition-b");
  missing.output = { other: "not the declared field" };
  assert.throws(() => preflightBundle(definition(), [record("condition-a"), missing]), /missing declared fields/);
  assert.throws(
    () => preflightBundle(definition(), [
      ...completeGrid(["item-1"]),
      record("condition-a", { item: "item-2" }),
      record("condition-c", { item: "item-2" }),
    ]),
    /inconsistent condition coverage/,
  );
});
