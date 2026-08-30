import { HarnessContractError } from "./error.js";
import type { JsonObject, JsonValue } from "./types.js";

function sorted(value: JsonValue): JsonValue {
  if (Array.isArray(value)) {
    return value.map(sorted);
  }
  if (value !== null && typeof value === "object") {
    const output: JsonObject = {};
    for (const key of Object.keys(value).sort()) {
      const child = value[key];
      if (child !== undefined) {
        output[key] = sorted(child);
      }
    }
    return output;
  }
  return value;
}

export function canonicalJson(value: JsonValue): string {
  return JSON.stringify(sorted(value));
}

export function assertJsonValue(value: unknown, where: string, active = new Set<unknown>()): asserts value is JsonValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new HarnessContractError("invalid-json", `${where} must not contain a non-finite number`);
    }
    return;
  }
  if (typeof value !== "object") {
    throw new HarnessContractError("invalid-json", `${where} contains a non-JSON ${typeof value} value`);
  }
  if (active.has(value)) {
    throw new HarnessContractError("invalid-json", `${where} contains a circular reference`);
  }
  const isArray = Array.isArray(value);
  const prototype = Object.getPrototypeOf(value);
  if (!isArray && prototype !== Object.prototype && prototype !== null) {
    throw new HarnessContractError("invalid-json", `${where} contains a non-plain object`);
  }
  active.add(value);
  if (isArray) {
    value.forEach((child, index) => assertJsonValue(child, `${where}[${index}]`, active));
  } else {
    for (const [key, child] of Object.entries(value)) {
      assertJsonValue(child, `${where}.${key}`, active);
    }
  }
  active.delete(value);
}
