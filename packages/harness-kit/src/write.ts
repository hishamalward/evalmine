import { createHash } from "node:crypto";
import { mkdir, rmdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { HarnessContractError } from "./error.js";
import { canonicalJson } from "./json.js";
import { preflightBundle } from "./preflight.js";
import type { BuiltBundle, ExternalManifest, JsonValue, WriteResult } from "./types.js";

const ARTIFACT_NAME = "records.jsonl";
const MANIFEST_NAME = "evalmine-import.yaml";

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

/** Write a complete source bundle to a destination that must not already exist. */
export async function writeBundle(bundle: BuiltBundle, destination: string): Promise<WriteResult> {
  if (!destination.trim()) {
    throw new HarnessContractError("invalid-destination", "bundle destination must not be empty");
  }
  const verified = preflightBundle(bundle.definition, bundle.records);
  if (
    verified.record_count !== bundle.summary.record_count ||
    verified.block_count !== bundle.summary.block_count ||
    verified.condition_count !== bundle.summary.condition_count
  ) {
    throw new HarnessContractError("stale-bundle", "bundle records changed after build; build it again before writing");
  }

  const directory = path.resolve(destination);
  await mkdir(path.dirname(directory), { recursive: true });
  try {
    await mkdir(directory);
  } catch (error) {
    const code = error instanceof Error && "code" in error ? String(error.code) : "";
    if (code === "EEXIST") {
      throw new HarnessContractError("destination-exists", `refusing to overwrite existing bundle destination ${directory}`);
    }
    throw error;
  }

  const artifactPath = path.join(directory, ARTIFACT_NAME);
  const manifestPath = path.join(directory, MANIFEST_NAME);
  let artifactCreated = false;
  let manifestCreated = false;
  try {
    const jsonl = `${bundle.records
      .map((record) => canonicalJson(record as unknown as JsonValue))
      .join("\n")}\n`;
    const artifactSha256 = sha256(jsonl);
    const manifest: ExternalManifest = {
      ...structuredClone(bundle.definition),
      version: 1,
      artifacts: [{ path: ARTIFACT_NAME, sha256: artifactSha256 }],
    };
    const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
    await writeFile(artifactPath, jsonl, { encoding: "utf8", flag: "wx" });
    artifactCreated = true;
    await writeFile(manifestPath, manifestText, { encoding: "utf8", flag: "wx" });
    manifestCreated = true;
    return {
      directory,
      manifest_path: manifestPath,
      artifact_path: artifactPath,
      artifact_sha256: artifactSha256,
      summary: structuredClone(bundle.summary),
    };
  } catch (error) {
    // Remove only files this invocation successfully created. If anything else
    // appeared concurrently, rmdir fails closed instead of deleting foreign data.
    if (manifestCreated) await unlink(manifestPath).catch(() => undefined);
    if (artifactCreated) await unlink(artifactPath).catch(() => undefined);
    await rmdir(directory).catch(() => undefined);
    throw error;
  }
}
