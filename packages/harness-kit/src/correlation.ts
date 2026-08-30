import { createHash } from "node:crypto";
import { HarnessContractError } from "./error.js";

const DEFAULT_MAX_LENGTH = 120;
const HASH_LENGTH = 20;

function readable(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/[-._]{2,}/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "")
    .toLowerCase();
}

/** Build an operationally unique, log-friendly ID without unsafe truncation collisions. */
export function correlationId(
  namespace: string,
  parts: readonly (string | number)[],
  maxLength = DEFAULT_MAX_LENGTH,
): string {
  if (!namespace.trim()) {
    throw new HarnessContractError("invalid-correlation-id", "correlation namespace must not be empty");
  }
  if (!Number.isInteger(maxLength) || maxLength < HASH_LENGTH + 3 || maxLength > DEFAULT_MAX_LENGTH) {
    throw new HarnessContractError(
      "invalid-correlation-id",
      `correlation maxLength must be an integer from ${HASH_LENGTH + 3} to ${DEFAULT_MAX_LENGTH}`,
    );
  }
  if (parts.some((part) => typeof part === "string" && !part.length)) {
    throw new HarnessContractError("invalid-correlation-id", "correlation parts must not contain an empty string");
  }
  if (parts.some((part) => typeof part === "number" && !Number.isFinite(part))) {
    throw new HarnessContractError("invalid-correlation-id", "numeric correlation parts must be finite");
  }
  const identity = JSON.stringify([namespace, ...parts]);
  const digest = createHash("sha256").update(identity).digest("hex").slice(0, HASH_LENGTH);
  const display = [namespace, ...parts.map(String)].map(readable).filter(Boolean).join(":") || "artifact";
  const prefixLength = maxLength - HASH_LENGTH - 1;
  const prefix = display.slice(0, prefixLength).replace(/[-._:]+$/g, "") || "artifact";
  return `${prefix}:${digest}`;
}
