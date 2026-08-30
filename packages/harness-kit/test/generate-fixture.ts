import path from "node:path";
import {
  HarnessBuilder,
  correlationId,
  sharedPrompt,
  writeBundle,
  type BundleDefinition,
  type ExternalRecord,
} from "../src/index.js";

const destination = process.argv[2];
if (!destination) {
  throw new Error("usage: npm run generate-fixture -- <destination>");
}

const definition: BundleDefinition = {
  external_artifacts: "synthetic-typescript-roundtrip",
  question: "Which synthetic completed condition is strongest?",
  evaluation: {
    objectives: ["Correctness", "Usefulness"],
    blind: "condition",
    ranking_style: "n-way",
    fields: ["summary", "risk"],
    human: { required: true, labels_per_pair: 1, coverage: "calibration-subset" },
    judge: {
      enabled: true,
      pairwise: false,
      position_swap: false,
      calibrate: true,
      runner: "api-prompt",
      model: "synthetic/judge",
      min_kappa: 0.4,
      min_labels: 2,
    },
  },
};

function fixtureRecord(item: string, condition: "condition-a" | "condition-b"): ExternalRecord {
  const prompt = sharedPrompt({
    input: { document_id: item, text: `Synthetic document ${item}` },
    whatVaries: "Only the declared condition.",
  });
  return {
    lane: "summary",
    item_id: item,
    account_id: `synthetic-account-${item}`,
    correlation_id: correlationId("roundtrip", [item, condition]),
    prompt,
    condition: {
      id: condition,
      model: `synthetic/${condition}`,
      prompt_variant: condition,
      width: "fixed",
    },
    output: {
      summary: `${condition} completed ${item}`,
      risk: condition === "condition-a" ? "low" : "medium",
    },
    cost_receipts: {
      estimated: { usd: 0.001, source: "synthetic estimate" },
      ledger: { usd: 0.0011, source: "synthetic ledger" },
      dashboard_observed: { usd: 0.0012, source: "synthetic observation" },
    },
  };
}

const records = ["item-1", "item-2"].flatMap((item) => [
  fixtureRecord(item, "condition-a"),
  fixtureRecord(item, "condition-b"),
]);
const bundle = new HarnessBuilder(definition).addLayer(records).build();
const result = await writeBundle(bundle, path.resolve(destination));
process.stdout.write(`${JSON.stringify(result)}\n`);
