import { HarnessContractError } from "./error.js";
import { canonicalJson, assertJsonValue } from "./json.js";
import type { JsonValue } from "./types.js";

export interface SharedPromptInput {
  input: JsonValue;
  whatVaries: string;
}

/** Format shared judge context; it does not render or invoke any arm. */
export function sharedPrompt({ input, whatVaries }: SharedPromptInput): string {
  assertJsonValue(input, "shared input");
  if (!whatVaries.trim()) {
    throw new HarnessContractError("invalid-shared-prompt", "whatVaries must not be empty");
  }
  const renderedInput = typeof input === "string" ? input : canonicalJson(input);
  if (!renderedInput.trim()) {
    throw new HarnessContractError("invalid-shared-prompt", "shared input must not be empty");
  }
  return `SHARED INPUT:\n${renderedInput}\n\nWHAT VARIES:\n${whatVaries.trim()}`;
}
