// Dynamic registry loader (task #47).
//
// The CLI is compiled to a single binary by `bun build --compile`, which
// inlines cli/registry.json forever — shipped binaries never learn about new
// models. This module fetches registry/v1.json from the repo's raw GitHub
// URL with a 24h on-disk cache, falling back to (in order) fresh cache →
// network → stale cache → the bundled registry.json the binary was built
// with. The CLI must never get WORSE than the bundled data: any fetch /
// parse / validation failure silently keeps the fallback chain going.
//
// Side-effect-free module (no top-level IO) so bun tests can import it
// directly, and hipfire-tui can share both the loader and the cache file at
// ~/.hipfire/registry.cache.json.
//
// registry/v1.json is a strict superset of cli/registry.json (same
// models/aliases shape + schema_version/generated_at/sha256/size_bytes/
// arch_id/quant), so a validated dynamic registry can simply replace the
// bundled REGISTRY/ALIASES maps. See scripts/registry_gen.py.

import { readFileSync, writeFileSync, renameSync, mkdirSync, unlinkSync } from "fs";
import { dirname } from "path";

export const REGISTRY_SCHEMA_VERSION = 1;
export const DEFAULT_REGISTRY_URL =
  "https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/registry/v1.json";
export const REGISTRY_CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24h
export const REGISTRY_FETCH_TIMEOUT_MS = 3500; // never hang the CLI offline

export interface RegistrySidecarV1 {
  file: string;
  sha256?: string | null;
  size_bytes?: number | null;
}

/// Same legacy shape as index.ts's ModelEntry plus the additive v1 fields.
export interface RegistryModelEntryV1 {
  repo: string;
  file: string;
  size_gb: number;
  min_vram_gb: number;
  desc: string;
  triattn?: RegistrySidecarV1;
  mtp?: RegistrySidecarV1;
  sha256?: string | null;
  size_bytes?: number | null;
  arch_id?: number | null;
  quant?: string | null;
}

export interface RegistryV1 {
  schema_version: number;
  generated_at: string;
  models: Record<string, RegistryModelEntryV1>;
  aliases: Record<string, string>;
}

export interface RegistryCacheFile {
  fetched_at: number; // epoch ms
  url: string;
  registry: RegistryV1;
}

export type RegistrySource = "cache" | "network" | "stale-cache" | "bundled";

export interface LoadResult {
  source: RegistrySource;
  /// null ⇒ caller keeps the bundled registry.
  registry: RegistryV1 | null;
}

// ─── pure validation ─────────────────────────────────────

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function validSidecar(v: unknown): boolean {
  return isRecord(v) && typeof v.file === "string" && v.file.length > 0;
}

function validEntry(v: unknown): v is RegistryModelEntryV1 {
  if (!isRecord(v)) return false;
  if (typeof v.repo !== "string") return false; // "" = local-only, allowed
  if (typeof v.file !== "string" || v.file.length === 0) return false;
  if (typeof v.size_gb !== "number" || !(v.size_gb >= 0)) return false;
  if (typeof v.min_vram_gb !== "number" || !(v.min_vram_gb >= 0)) return false;
  if (typeof v.desc !== "string") return false;
  if (v.triattn !== undefined && !validSidecar(v.triattn)) return false;
  if (v.mtp !== undefined && !validSidecar(v.mtp)) return false;
  return true;
}

/// Fail-closed structural validation. Returns the typed registry or null —
/// a registry that fails ANY check is rejected wholesale (we never serve a
/// half-broken model list when the bundled fallback is known-good).
/// `schema_version` must be exactly 1: a future v2 with breaking shape
/// changes must not be trusted by binaries that only understand v1.
export function validateRegistryV1(data: unknown): RegistryV1 | null {
  if (!isRecord(data)) return null;
  if (data.schema_version !== REGISTRY_SCHEMA_VERSION) return null;
  if (typeof data.generated_at !== "string") return null;
  if (!isRecord(data.models) || Object.keys(data.models).length === 0) return null;
  for (const entry of Object.values(data.models)) {
    if (!validEntry(entry)) return null;
  }
  if (!isRecord(data.aliases)) return null;
  const models = data.models as Record<string, RegistryModelEntryV1>;
  const aliases: Record<string, string> = {};
  for (const [k, v] of Object.entries(data.aliases)) {
    if (typeof v !== "string") return null;
    // Drop (don't fail on) aliases to tags this registry doesn't carry —
    // an alias is a convenience redirect, not load-bearing data.
    if (models[v] !== undefined) aliases[k] = v;
  }
  return {
    schema_version: data.schema_version,
    generated_at: data.generated_at,
    models,
    aliases,
  };
}

export function parseCacheFile(raw: string): RegistryCacheFile | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(data)) return null;
  if (typeof data.fetched_at !== "number" || !Number.isFinite(data.fetched_at)) return null;
  if (typeof data.url !== "string") return null;
  const registry = validateRegistryV1(data.registry);
  if (!registry) return null;
  return { fetched_at: data.fetched_at, url: data.url, registry };
}

export function cacheIsFresh(
  cache: Pick<RegistryCacheFile, "fetched_at">,
  nowMs: number,
  ttlMs: number = REGISTRY_CACHE_TTL_MS,
): boolean {
  // A fetched_at in the future (clock skew, restored backup) is NOT fresh —
  // treat it as stale so we re-fetch rather than trusting it for years.
  return cache.fetched_at <= nowMs && nowMs - cache.fetched_at < ttlMs;
}

// ─── load orchestration ──────────────────────────────────

export interface LoadOptions {
  cachePath: string;
  url?: string;
  ttlMs?: number;
  nowMs?: number;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
  readFile?: (path: string) => string; // throws if missing (fs semantics)
  writeFile?: (path: string, contents: string) => void;
}

function defaultWriteFile(path: string, contents: string): void {
  // Atomic-ish: tmp + rename, so a crash mid-write never leaves a torn cache.
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp.${process.pid}`;
  try {
    writeFileSync(tmp, contents);
    renameSync(tmp, path);
  } catch (err) {
    try {
      unlinkSync(tmp);
    } catch {}
    throw err;
  }
}

/// Fallback chain: fresh cache → network (writes cache) → stale cache →
/// bundled (registry: null). Never throws.
export async function loadDynamicRegistry(opts: LoadOptions): Promise<LoadResult> {
  const url = opts.url ?? DEFAULT_REGISTRY_URL;
  const ttlMs = opts.ttlMs ?? REGISTRY_CACHE_TTL_MS;
  const nowMs = opts.nowMs ?? Date.now();
  const timeoutMs = opts.timeoutMs ?? REGISTRY_FETCH_TIMEOUT_MS;
  const fetchImpl = opts.fetchImpl ?? fetch;
  const readFile = opts.readFile ?? ((p: string) => readFileSync(p, "utf8"));
  const writeFile = opts.writeFile ?? defaultWriteFile;

  let cache: RegistryCacheFile | null = null;
  try {
    cache = parseCacheFile(readFile(opts.cachePath));
  } catch {
    cache = null; // missing/unreadable cache file
  }
  // A cache fetched from a different URL (e.g. HIPFIRE_REGISTRY_URL override
  // changed) must not satisfy freshness for the current URL.
  if (cache && cache.url !== url) cache = null;

  if (cache && cacheIsFresh(cache, nowMs, ttlMs)) {
    return { source: "cache", registry: cache.registry };
  }

  try {
    const resp = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (resp.ok) {
      const registry = validateRegistryV1(await resp.json());
      if (registry) {
        try {
          const cacheFile: RegistryCacheFile = { fetched_at: nowMs, url, registry };
          writeFile(opts.cachePath, JSON.stringify(cacheFile));
        } catch {
          // Cache write failure is non-fatal — we still have the registry.
        }
        return { source: "network", registry };
      }
    }
  } catch {
    // Offline / DNS / timeout / non-JSON — fall through.
  }

  if (cache) {
    return { source: "stale-cache", registry: cache.registry };
  }
  return { source: "bundled", registry: null };
}
