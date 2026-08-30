export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  [key: string]: JsonValue;
}

export type ReceiptBasis = "estimated" | "ledger" | "dashboard_observed";

export interface CostReceipt {
  usd: number;
  source: string;
  observed_at?: string;
  note?: string;
}

export type CostReceipts = Partial<Record<ReceiptBasis, CostReceipt>>;

export interface Condition {
  id: string;
  model: string;
  prompt_variant: string;
  width: string | number;
  metadata?: JsonObject;
}

/**
 * A completed producer artifact. `item_id` identifies the compared item and must
 * not encode `condition`; `correlation_id` joins this artifact to producer logs.
 */
export interface ExternalRecord {
  lane: string;
  item_id: string;
  account_id: string;
  correlation_id: string;
  prompt: string;
  output: JsonValue;
  fields?: JsonObject;
  condition: Condition;
  cost_receipts?: CostReceipts;
  duration_ms?: number;
  usage?: JsonObject;
  metadata?: JsonObject;
}

export type RankingStyle = "pairwise" | "n-way";
export type JudgeRunner = "api-prompt" | "claude-code" | "codex-cli" | "gemini-cli";

export interface HumanEvaluation {
  required: boolean;
  labels_per_pair?: number;
  coverage?: "calibration-subset";
}

export interface JudgeEvaluation {
  enabled: boolean;
  pairwise: boolean;
  position_swap: boolean;
  calibrate: boolean;
  runner?: JudgeRunner;
  model?: string;
  max_cost_usd?: number;
  max_tokens?: number;
  min_kappa?: number;
  min_labels?: number;
}

export interface Evaluation {
  objectives: string[];
  blind: "condition";
  ranking_style: RankingStyle;
  fields?: string[];
  human: HumanEvaluation;
  judge: JudgeEvaluation;
}

export interface BundleDefinition {
  external_artifacts: string;
  question: string;
  evaluation: Evaluation;
}

export interface ArtifactReference {
  path: string;
  sha256: string;
}

export interface ExternalManifest extends BundleDefinition {
  version: 1;
  artifacts: ArtifactReference[];
}

export interface PreflightSummary {
  record_count: number;
  block_count: number;
  condition_count: number;
}

export interface BuiltBundle {
  definition: BundleDefinition;
  records: ExternalRecord[];
  summary: PreflightSummary & {
    deduped_record_count: number;
    excluded_record_count: number;
  };
}

export interface BlockSelector {
  lane: string;
  item_id: string;
  account_id: string;
}

export interface BuildOptions {
  exclude_conditions?: readonly string[];
  exclude_blocks?: readonly BlockSelector[];
}

export interface WriteResult {
  directory: string;
  manifest_path: string;
  artifact_path: string;
  artifact_sha256: string;
  summary: BuiltBundle["summary"];
}
