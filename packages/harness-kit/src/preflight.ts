import { HarnessContractError } from "./error.js";
import { assertJsonValue, canonicalJson } from "./json.js";
import type {
  BundleDefinition,
  Condition,
  CostReceipt,
  ExternalRecord,
  JsonObject,
  JsonValue,
  PreflightSummary,
} from "./types.js";

type UnknownObject = Record<string, unknown>;

const RECORD_KEYS = new Set([
  "lane",
  "item_id",
  "account_id",
  "correlation_id",
  "prompt",
  "output",
  "fields",
  "condition",
  "cost_receipts",
  "duration_ms",
  "usage",
  "metadata",
]);
const CONDITION_KEYS = new Set(["id", "model", "prompt_variant", "width", "metadata"]);
const RECEIPT_KEYS = new Set(["usd", "source", "observed_at", "note"]);
const RECEIPT_BASES = new Set(["estimated", "ledger", "dashboard_observed"]);
const RUNNERS = new Set(["api-prompt", "claude-code", "codex-cli", "gemini-cli"]);

function object(value: unknown, where: string): UnknownObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new HarnessContractError("invalid-shape", `${where} must be an object`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new HarnessContractError("invalid-shape", `${where} must be a plain object`);
  }
  return value as UnknownObject;
}

function onlyKeys(value: UnknownObject, allowed: ReadonlySet<string>, where: string): void {
  const unexpected = Object.keys(value).filter((key) => !allowed.has(key));
  if (unexpected.length) {
    throw new HarnessContractError(
      "unexpected-property",
      `${where} has unexpected properties: ${unexpected.join(", ")}`,
    );
  }
}

function text(value: unknown, where: string, maxLength?: number): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new HarnessContractError("invalid-string", `${where} must be a non-empty string`);
  }
  if (maxLength !== undefined && value.length > maxLength) {
    throw new HarnessContractError("invalid-string", `${where} must be at most ${maxLength} characters`);
  }
  return value;
}

function boolean(value: unknown, where: string): boolean {
  if (typeof value !== "boolean") {
    throw new HarnessContractError("invalid-boolean", `${where} must be a boolean`);
  }
  return value;
}

function finiteNumber(value: unknown, where: string, minimum?: number, exclusive = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new HarnessContractError("invalid-number", `${where} must be a finite number`);
  }
  if (minimum !== undefined && (exclusive ? value <= minimum : value < minimum)) {
    const phrase = exclusive ? "greater than" : "at least";
    throw new HarnessContractError("invalid-number", `${where} must be ${phrase} ${minimum}`);
  }
  return value;
}

function integer(value: unknown, where: string, minimum: number, maximum?: number): number {
  const result = finiteNumber(value, where, minimum);
  if (!Number.isInteger(result) || (maximum !== undefined && result > maximum)) {
    const suffix = maximum === undefined ? "" : ` and at most ${maximum}`;
    throw new HarnessContractError("invalid-integer", `${where} must be an integer of at least ${minimum}${suffix}`);
  }
  return result;
}

function stringList(value: unknown, where: string): string[] {
  if (!Array.isArray(value) || !value.length) {
    throw new HarnessContractError("invalid-list", `${where} must be a non-empty array`);
  }
  const result = value.map((item, index) => text(item, `${where}[${index}]`));
  if (new Set(result).size !== result.length) {
    throw new HarnessContractError("duplicate-list-value", `${where} must contain unique values`);
  }
  return result;
}

function validateReceipt(value: unknown, where: string): asserts value is CostReceipt {
  const receipt = object(value, where);
  onlyKeys(receipt, RECEIPT_KEYS, where);
  finiteNumber(receipt.usd, `${where}.usd`, 0);
  text(receipt.source, `${where}.source`);
  if (receipt.observed_at !== undefined) text(receipt.observed_at, `${where}.observed_at`);
  if (receipt.note !== undefined && typeof receipt.note !== "string") {
    throw new HarnessContractError("invalid-string", `${where}.note must be a string`);
  }
}

function validateCondition(value: unknown, where: string): asserts value is Condition {
  const condition = object(value, where);
  onlyKeys(condition, CONDITION_KEYS, where);
  text(condition.id, `${where}.id`);
  text(condition.model, `${where}.model`);
  text(condition.prompt_variant, `${where}.prompt_variant`);
  if (typeof condition.width === "number") {
    integer(condition.width, `${where}.width`, 1);
  } else {
    text(condition.width, `${where}.width`);
  }
  if (condition.metadata !== undefined) {
    object(condition.metadata, `${where}.metadata`);
    assertJsonValue(condition.metadata, `${where}.metadata`);
  }
}

export function validateRecord(value: unknown, where = "record"): asserts value is ExternalRecord {
  const record = object(value, where);
  onlyKeys(record, RECORD_KEYS, where);
  text(record.lane, `${where}.lane`);
  text(record.item_id, `${where}.item_id`);
  text(record.account_id, `${where}.account_id`);
  text(record.correlation_id, `${where}.correlation_id`, 120);
  text(record.prompt, `${where}.prompt`);
  validateCondition(record.condition, `${where}.condition`);
  if (record.output === null || record.output === undefined) {
    throw new HarnessContractError("missing-output", `${where}.output must be completed and non-null`);
  }
  assertJsonValue(record.output, `${where}.output`);
  if (record.fields !== undefined) {
    const fields = object(record.fields, `${where}.fields`);
    if (!Object.keys(fields).length) {
      throw new HarnessContractError("invalid-fields", `${where}.fields must not be empty`);
    }
    assertJsonValue(fields, `${where}.fields`);
  }
  if (record.cost_receipts !== undefined) {
    const receipts = object(record.cost_receipts, `${where}.cost_receipts`);
    onlyKeys(receipts, RECEIPT_BASES, `${where}.cost_receipts`);
    for (const [basis, receipt] of Object.entries(receipts)) {
      validateReceipt(receipt, `${where}.cost_receipts.${basis}`);
    }
  }
  if (record.duration_ms !== undefined) integer(record.duration_ms, `${where}.duration_ms`, 0);
  for (const key of ["usage", "metadata"] as const) {
    if (record[key] !== undefined) {
      object(record[key], `${where}.${key}`);
      assertJsonValue(record[key], `${where}.${key}`);
    }
  }
}

export function validateDefinition(value: unknown): asserts value is BundleDefinition {
  const definition = object(value, "bundle definition");
  onlyKeys(definition, new Set(["external_artifacts", "question", "evaluation"]), "bundle definition");
  const name = text(definition.external_artifacts, "bundle definition.external_artifacts");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name)) {
    throw new HarnessContractError(
      "invalid-bundle-name",
      "bundle definition.external_artifacts must use only letters, digits, dots, underscores, and hyphens",
    );
  }
  text(definition.question, "bundle definition.question");

  const evaluation = object(definition.evaluation, "bundle definition.evaluation");
  onlyKeys(
    evaluation,
    new Set(["objectives", "blind", "ranking_style", "fields", "human", "judge"]),
    "bundle definition.evaluation",
  );
  stringList(evaluation.objectives, "bundle definition.evaluation.objectives");
  if (evaluation.blind !== "condition") {
    throw new HarnessContractError("invalid-blinding", "bundle definition.evaluation.blind must be condition");
  }
  if (evaluation.ranking_style !== "pairwise" && evaluation.ranking_style !== "n-way") {
    throw new HarnessContractError(
      "invalid-ranking-style",
      "bundle definition.evaluation.ranking_style must be pairwise or n-way",
    );
  }
  if (evaluation.fields !== undefined) stringList(evaluation.fields, "bundle definition.evaluation.fields");

  const human = object(evaluation.human, "bundle definition.evaluation.human");
  onlyKeys(human, new Set(["required", "labels_per_pair", "coverage"]), "bundle definition.evaluation.human");
  boolean(human.required, "bundle definition.evaluation.human.required");
  if (human.labels_per_pair !== undefined) {
    integer(human.labels_per_pair, "bundle definition.evaluation.human.labels_per_pair", 1);
  }
  if (human.coverage !== undefined && human.coverage !== "calibration-subset") {
    throw new HarnessContractError(
      "invalid-human-coverage",
      "bundle definition.evaluation.human.coverage must be calibration-subset",
    );
  }

  const judge = object(evaluation.judge, "bundle definition.evaluation.judge");
  onlyKeys(
    judge,
    new Set([
      "enabled",
      "pairwise",
      "position_swap",
      "calibrate",
      "runner",
      "model",
      "max_cost_usd",
      "max_tokens",
      "min_kappa",
      "min_labels",
    ]),
    "bundle definition.evaluation.judge",
  );
  boolean(judge.enabled, "bundle definition.evaluation.judge.enabled");
  boolean(judge.pairwise, "bundle definition.evaluation.judge.pairwise");
  boolean(judge.position_swap, "bundle definition.evaluation.judge.position_swap");
  boolean(judge.calibrate, "bundle definition.evaluation.judge.calibrate");
  if (judge.runner !== undefined && !RUNNERS.has(String(judge.runner))) {
    throw new HarnessContractError("invalid-runner", "bundle definition.evaluation.judge.runner is not supported");
  }
  if (judge.model !== undefined) text(judge.model, "bundle definition.evaluation.judge.model");
  if (judge.max_cost_usd !== undefined) {
    finiteNumber(judge.max_cost_usd, "bundle definition.evaluation.judge.max_cost_usd", 0, true);
  }
  if (judge.max_tokens !== undefined) {
    integer(judge.max_tokens, "bundle definition.evaluation.judge.max_tokens", 64, 32768);
  }
  if (judge.min_kappa !== undefined) {
    const kappa = finiteNumber(judge.min_kappa, "bundle definition.evaluation.judge.min_kappa");
    if (kappa < -1 || kappa > 1) {
      throw new HarnessContractError("invalid-number", "bundle definition.evaluation.judge.min_kappa must be from -1 to 1");
    }
  }
  if (judge.min_labels !== undefined) {
    integer(judge.min_labels, "bundle definition.evaluation.judge.min_labels", 1);
  }

  if (evaluation.ranking_style === "n-way" && (judge.pairwise || judge.position_swap)) {
    throw new HarnessContractError(
      "ranking-flags",
      "n-way ranking requires judge.pairwise=false and judge.position_swap=false",
    );
  }
  if (evaluation.ranking_style === "pairwise" && (!judge.pairwise || !judge.position_swap)) {
    throw new HarnessContractError(
      "ranking-flags",
      "pairwise ranking requires judge.pairwise=true and judge.position_swap=true",
    );
  }
}

export function comparisonBlockKey(record: Pick<ExternalRecord, "lane" | "item_id" | "account_id">): string {
  return JSON.stringify([record.lane, record.item_id, record.account_id]);
}

export function artifactKey(record: ExternalRecord): string {
  return JSON.stringify([record.lane, record.item_id, record.account_id, record.condition.id]);
}

function displayBlock(record: Pick<ExternalRecord, "lane" | "item_id" | "account_id">): string {
  return `${record.lane}/${record.item_id}/${record.account_id}`;
}

function outputFields(record: ExternalRecord): JsonObject | undefined {
  if (record.fields !== undefined) return record.fields;
  if (record.output !== null && typeof record.output === "object" && !Array.isArray(record.output)) {
    return record.output;
  }
  return undefined;
}

export function preflightBundle(definition: BundleDefinition, records: readonly ExternalRecord[]): PreflightSummary {
  validateDefinition(definition);
  if (!records.length) {
    throw new HarnessContractError("empty-bundle", "bundle contains no completed records");
  }

  const conditions = new Map<string, string>();
  const blocks = new Map<string, ExternalRecord[]>();
  const fields = definition.evaluation.fields ?? [];
  records.forEach((record, index) => {
    validateRecord(record, `records[${index}]`);
    const conditionId = record.condition.id;
    const serialized = canonicalJson(record.condition as unknown as JsonValue);
    const prior = conditions.get(conditionId);
    if (prior !== undefined && prior !== serialized) {
      throw new HarnessContractError(
        "inconsistent-condition",
        `condition ${JSON.stringify(conditionId)} has inconsistent model/prompt/width metadata`,
      );
    }
    conditions.set(conditionId, serialized);

    const key = comparisonBlockKey(record);
    const block = blocks.get(key) ?? [];
    if (block.length && block[0]?.prompt !== record.prompt) {
      throw new HarnessContractError(
        "inconsistent-shared-prompt",
        `comparison block ${displayBlock(record)} has inconsistent shared prompts`,
      );
    }
    if (block.some((existing) => existing.condition.id === conditionId)) {
      throw new HarnessContractError(
        "duplicate-condition",
        `comparison block ${displayBlock(record)} repeats condition ${conditionId}`,
      );
    }
    block.push(record);
    blocks.set(key, block);

    const values = outputFields(record);
    const missing = fields.filter((field) => values === undefined || !(field in values));
    if (missing.length) {
      throw new HarnessContractError(
        "missing-fields",
        `${displayBlock(record)}/${conditionId} is missing declared fields: ${missing.join(", ")}`,
      );
    }
  });

  const incomplete = [...blocks.values()]
    .filter((block) => block.length < 2)
    .map((block) => displayBlock(block[0]!));
  if (incomplete.length) {
    throw new HarnessContractError(
      "singleton-block",
      `every comparison block needs at least two conditions; incomplete: ${incomplete.slice(0, 5).join(", ")}`,
    );
  }

  const laneConditions = new Map<string, string>();
  for (const block of blocks.values()) {
    const lane = block[0]!.lane;
    const found = [...block.map((record) => record.condition.id)].sort().join("\u0000");
    const expected = laneConditions.get(lane);
    if (expected !== undefined && expected !== found) {
      throw new HarnessContractError(
        "inconsistent-lane-coverage",
        `lane ${JSON.stringify(lane)} has inconsistent condition coverage across items`,
      );
    }
    laneConditions.set(lane, found);
  }

  return {
    record_count: records.length,
    block_count: blocks.size,
    condition_count: conditions.size,
  };
}
