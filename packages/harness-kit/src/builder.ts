import { HarnessContractError } from "./error.js";
import {
  artifactKey,
  comparisonBlockKey,
  preflightBundle,
  validateDefinition,
  validateRecord,
} from "./preflight.js";
import type {
  BuildOptions,
  BuiltBundle,
  BundleDefinition,
  ExternalRecord,
} from "./types.js";

function clone<T>(value: T): T {
  return structuredClone(value);
}

export class HarnessBuilder {
  readonly #definition: BundleDefinition;
  readonly #records = new Map<string, ExternalRecord>();
  #deduped = 0;
  #layerCount = 0;

  constructor(definition: BundleDefinition) {
    validateDefinition(definition);
    this.#definition = clone(definition);
  }

  /** Add completed records in precedence order; a later identical artifact key wins. */
  addLayer(records: readonly ExternalRecord[]): this {
    if (!Array.isArray(records)) {
      throw new HarnessContractError("invalid-layer", "addLayer expects an array of completed records");
    }
    const layer = this.#layerCount++;
    records.forEach((record, index) => {
      validateRecord(record, `layers[${layer}][${index}]`);
      const key = artifactKey(record);
      if (this.#records.has(key)) {
        this.#records.delete(key);
        this.#deduped += 1;
      }
      this.#records.set(key, clone(record));
    });
    return this;
  }

  build(options: BuildOptions = {}): BuiltBundle {
    const excludedConditions = new Set(options.exclude_conditions ?? []);
    const excludedBlocks = new Set(
      (options.exclude_blocks ?? []).map((selector) => {
        if (!selector.lane?.trim() || !selector.item_id?.trim() || !selector.account_id?.trim()) {
          throw new HarnessContractError("invalid-exclusion", "block exclusions require lane, item_id, and account_id");
        }
        return comparisonBlockKey(selector);
      }),
    );
    for (const id of excludedConditions) {
      if (!id.trim()) {
        throw new HarnessContractError("invalid-exclusion", "condition exclusions must not be empty");
      }
    }

    const allRecords = [...this.#records.values()];
    const records = allRecords.filter(
      (record) =>
        !excludedConditions.has(record.condition.id) &&
        !excludedBlocks.has(comparisonBlockKey(record)),
    );
    const preflight = preflightBundle(this.#definition, records);
    return {
      definition: clone(this.#definition),
      records: clone(records),
      summary: {
        ...preflight,
        deduped_record_count: this.#deduped,
        excluded_record_count: allRecords.length - records.length,
      },
    };
  }
}
