// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (provider directory, sealed quality table,
// workload manifest), probes every provider's catalog route, registers one
// catalog with the env, then answers the env client's twenty requests. Each
// request is served by the best free model that meets its requirements, with
// failover to the next-best model when a provider refuses (402/429/5xx).
//
// Routing: a startup planner assigns each manifest request the best model
// whose provider still has soft-budget headroom (soft caps are the low end
// of the per-trial budget ranges the kit publishes for each provider), so the
// top-rated provider is not burned out on the early requests. At request time
// the planned pick goes first, then the remaining candidates in rating order,
// deprioritising providers that have already refused. Every choice is a pure
// function of the brief, the manifest and the provider answers seen so far:
// no wall clock, no unseeded randomness, so a replay leaves the same state.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages
// (JSON or SSE); each provider speaks openai, anthropic or gemini. All six
// directions are translated, preserving exactly what the provider echoes back
// (system prompt, user text, tool names, max_tokens, stop sequences, image
// hashes, JSON mode) and the provider's usage and stop reason, mapped into
// the client's wire. A JSON-mode request never goes to an anthropic-wire
// provider: that wire cannot carry response_format, so the echo would lose it.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
const asBool = (v: unknown): boolean => v === true;

/** Protocol version the register body carries. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take (the client allows 30 s). */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A fixed epoch second for every listed model's `created`: no clock in replies. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: no clock in replies. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The anthropic-version header the anthropic wire wants. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  workload: WorkloadEntry[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function parseProvider(v: unknown): ProviderDir | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli),
    context_length: typeof v.context_length === 'number' && Number.isFinite(v.context_length) ? v.context_length : null,
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: asBool(v.free),
  };
}

function parseWorkloadEntry(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: WorkloadEntry = {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
  return out;
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = parseRating(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = parseWorkloadEntry(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, workload, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probe
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderDir;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A price field of a listing: true when every price it names is zero. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function probedModelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  return {
    id,
    display_name:
      asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function probedModelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = probedModelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderDir, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: probedModelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

function ratingOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one, else the listing's own word. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, its base_url the directory's own path.
 */
function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDir;
  model: string;
  rating: number;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The soft per-provider call budget the planner reserves: the low end of the
 * per-trial budget range the kit publishes for each provider. A cap is only
 * a planning preference — failover still uses a capped provider when it is
 * the best model left, and a provider that refuses is deprioritised at
 * request time.
 */
const SOFT_CAPS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const softCap = (providerId: string): number => SOFT_CAPS[providerId] ?? 6;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** The manifest's requests, in arrival order, and the planner's pick for each. */
let plannedPicks: (Candidate | null)[] = [];
/** How many client inference requests have arrived. */
let manifestIndex = 0;
/** Calls attempted per provider, successes and refusals alike. */
const callsMade: Record<string, number> = {};
/** Refusals (402/429/5xx/network) per provider. */
const refusals: Record<string, number> = {};
/**
 * Refusals in a row per provider, reset by any success. A provider whose
 * budget is spent fails every call, so it reaches two in a row quickly;
 * a provider hit by its scheduled one-off fail calls does not, so it is
 * not written off for a refusal that already passed.
 */
const consecutiveRefusals: Record<string, number> = {};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The free candidates for one request, from the quality table's rows. */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // An anthropic-wire provider cannot carry response_format, so a JSON-mode
    // request routed there would echo json_mode false and fail the check.
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOf(brief, probe.provider.id, model.id);
      if (row === null || !row.free) continue;
      if (!meetsRequirements(requires, row)) continue;
      out.push({ provider: probe.provider, model: model.id, rating: row.rating_milli });
    }
  }
  return out;
}

/**
 * Assign each manifest request its first-attempt model: the best candidate
 * whose provider still has soft-budget headroom, else the best candidate.
 * Runs once, at startup, from the manifest alone — deterministic.
 */
function planPicks(): (Candidate | null)[] {
  const planned: Record<string, number> = {};
  return brief.workload.map((entry) => {
    const cands = candidatesFor(entry.requires, false);
    const sorted = [...cands].sort((a, b) => {
      const aCap = (planned[a.provider.id] ?? 0) < softCap(a.provider.id) ? 0 : 1;
      const bCap = (planned[b.provider.id] ?? 0) < softCap(b.provider.id) ? 0 : 1;
      if (aCap !== bCap) return aCap - bCap;
      if (b.rating !== a.rating) return b.rating - a.rating;
      const pa = planned[a.provider.id] ?? 0;
      const pb = planned[b.provider.id] ?? 0;
      if (pa !== pb) return pa - pb;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    const pick = sorted[0] ?? null;
    if (pick !== null) planned[pick.provider.id] = (planned[pick.provider.id] ?? 0) + 1;
    return pick;
  });
}

/**
 * The attempt order for one request: the planned pick first (unless its
 * provider has refused twice, which reads as spent), then the rest by rating,
 * with fewer-refused and fewer-called providers ahead on ties.
 */
function attemptOrder(requires: Requirement, jsonMode: boolean, planned: Candidate | null): Candidate[] {
  return candidatesFor(requires, jsonMode).sort((a, b) => {
    const aSpent = (consecutiveRefusals[a.provider.id] ?? 0) >= 2 ? 1 : 0;
    const bSpent = (consecutiveRefusals[b.provider.id] ?? 0) >= 2 ? 1 : 0;
    if (aSpent !== bSpent) return aSpent - bSpent;
    const aPlanned = planned !== null && a.provider.id === planned.provider.id && a.model === planned.model ? 0 : 1;
    const bPlanned = planned !== null && b.provider.id === planned.provider.id && b.model === planned.model ? 0 : 1;
    if (aPlanned !== bPlanned) return aPlanned - bPlanned;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ra = refusals[a.provider.id] ?? 0;
    const rb = refusals[b.provider.id] ?? 0;
    if (ra !== rb) return ra - rb;
    const ca = callsMade[a.provider.id] ?? 0;
    const cb = callsMade[b.provider.id] ?? 0;
    if (ca !== cb) return ca - cb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: clientApi === 'openai' && (format === 'json_object' || format === 'json_schema'),
  };
}

/** The free model ids this server lists, each once, in catalog order. */
function servedModelIds(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = ratingOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Wire translation: request bodies
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks, joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Split a data-URL image into its media type and base64 data, or null. */
function splitDataUrl(url: string): { media_type: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media_type: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One Anthropic image block as an OpenAI image_url part. */
function anthropicImageToOpenAI(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/** One OpenAI image_url part as an Anthropic image block. */
function openAIImageToAnthropic(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = asString(part.image_url.url);
  if (url === '') return null;
  const data = splitDataUrl(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media_type, data: data.data } };
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, and the cap, temperature, stops and streaming
 * pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: asString(block.id),
          type: 'function',
          function: { name: asString(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = blocksText(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asString(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else {
        const image = anthropicImageToOpenAI(block);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: asString(raw.description),
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`: system messages become the system prompt, image_url parts
 * become image blocks, tool messages become tool_result blocks, and the cap,
 * temperature, stops and streaming pass through.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    const content = raw.content;
    if (role === 'system') {
      const text = blocksText(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = blocksText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: asString(call.id), name: asString(call.function.name), input: argumentsOf(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: asString(raw.tool_call_id), content: typeof content === 'string' ? content : blocksText(content) }] });
      continue;
    }
    // A user message: text and image parts, in order.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else {
        const image = openAIImageToAnthropic(part);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: asString(raw.function.description),
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI tool definition as a Gemini function declaration. */
function openAIToolToGemini(raw: unknown): Json | null {
  if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return null;
  return {
    name: raw.function.name,
    description: asString(raw.function.description),
    parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
  };
}

/** One Anthropic tool definition as a Gemini function declaration. */
function anthropicToolToGemini(raw: unknown): Json | null {
  if (!isRecord(raw) || typeof raw.name !== 'string') return null;
  return {
    name: raw.name,
    description: asString(raw.description),
    parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
  };
}

/** The Gemini generationConfig for one client request. */
function geminiConfig(body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const config: Json = {};
  const maxTokens =
    clientApi === 'openai'
      ? typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : null
      : typeof body.max_tokens === 'number'
        ? body.max_tokens
        : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientApi === 'openai' ? body.stop : body.stop_sequences;
  if (typeof stops === 'string' && stops !== '') config.stopSequences = [stops];
  else if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops;
  if (clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request:
 * system messages become the system instruction, user images become
 * inlineData (or fileData for plain URLs), tools become function
 * declarations, and the generation config carries the cap and stops.
 */
function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    const content = raw.content;
    if (role === 'system') {
      const text = blocksText(content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (role === 'user') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url)) {
            const url = asString(part.image_url.url);
            const data = splitDataUrl(url);
            if (data !== null) parts.push({ inlineData: { mimeType: data.media_type, data: data.data } });
            else if (url !== '') parts.push({ fileData: { fileUri: url } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = blocksText(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: asString(call.function.name), args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      // A tool result rides on the last user content, so it never becomes
      // the last content itself and never overwrites the echoed user text.
      const response = { functionResponse: { name: '', response: { content: typeof content === 'string' ? content : blocksText(content) } } };
      const last = contents.length > 0 ? contents[contents.length - 1] : null;
      if (last !== null && last.role === 'user' && Array.isArray(last.parts)) (last.parts as Json[]).push(response);
      else contents.push({ role: 'user', parts: [response] });
    }
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      const decl = openAIToolToGemini(raw);
      return decl === null ? [] : [decl];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = geminiConfig(body, 'openai');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system = blocksText(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: asString(block.name), args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image parts, plus tool results as
    // functionResponse parts beside the text, so the echoed user text keeps.
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: asString(block.source.media_type) !== '' ? asString(block.source.media_type) : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: '', response: { content: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      const decl = anthropicToolToGemini(raw);
      return decl === null ? [] : [decl];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = geminiConfig(body, 'anthropic');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Wire translation: responses
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function openAIFinishToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a Gemini candidate's content parts, joined. */
function geminiCandidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIJsonToAnthropic(answer: Json, model: string): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({ type: 'tool_use', id: asString(call.id) !== '' ? asString(call.id) : 'toolu_unknown', name: asString(call.function.name), input: argumentsOf(call.function.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    id: `msg_${asString(answer.id) !== '' ? asString(answer.id) : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIFinishToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: asNumber(usage.prompt_tokens), output_tokens: asNumber(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
function anthropicJsonToOpenAI(answer: Json, model: string): Json {
  const content = Array.isArray(answer.content) ? answer.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: asString(block.id),
      type: 'function',
      function: { name: asString(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicStopToOpenAI(answer.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: asNumber(usage.input_tokens), completion_tokens: asNumber(usage.output_tokens), total_tokens: asNumber(usage.input_tokens) + asNumber(usage.output_tokens) },
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
function geminiJsonToOpenAI(answer: Json, model: string): Json {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiCandidateText(candidate);
  const usage = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiFinishToOpenAI(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: asNumber(usage.promptTokenCount), completion_tokens: asNumber(usage.candidatesTokenCount), total_tokens: asNumber(usage.promptTokenCount) + asNumber(usage.candidatesTokenCount) },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
function geminiJsonToAnthropic(answer: Json, model: string): Json {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiCandidateText(candidate);
  const usage = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: asNumber(usage.promptTokenCount), output_tokens: asNumber(usage.candidatesTokenCount) },
  };
}

/** One server-sent event: `event: <name>` when named, then the data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

interface SseFrame {
  event: string | null;
  data: string;
}

/** Split one SSE frame's lines into its event name and data payload. */
function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** A ReadableStream of SSE bytes as its frames, in arrival order. */
async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseSseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
}

function openAIUsageChunk(model: string, inputTokens: number, outputTokens: number): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence:
 * message_start, content_block_start/delta/stop per text block or tool call,
 * then message_delta with the stop reason and usage, then message_stop.
 */
async function* openAIStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const startEvents = function* (): Generator<string> {
    if (started) return;
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    });
  };
  const finishEvents = function* (stopReason: string): Generator<string> {
    if (finished) return;
    finished = true;
    if (textIndex !== -1) {
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    toolIndexes.clear();
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  };

  for await (const frame of sseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = asNumber(chunk.usage.prompt_tokens);
      outputTokens = asNumber(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: { type: 'tool_use', id: asString(call.id) !== '' ? asString(call.id) : `toolu_${index}`, name: asString(fn.name), input: {} },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') pendingStop = openAIFinishToAnthropic(choice.finish_reason);
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/**
 * An Anthropic SSE stream as OpenAI chat chunks: a role chunk, content
 * deltas, tool-call deltas, then the finish chunk, a usage chunk and [DONE].
 */
async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let doneEmitted = false;
  let inputTokens = 0;
  let outputTokens = 0;
  const tools = new Map<number, { id: string; name: string; started: boolean }>();

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = asNumber(data.message.usage.input_tokens);
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
    } else if (frame.event === 'content_block_start') {
      const index = asNumber(data.index);
      if (isRecord(data.content_block) && data.content_block.type === 'tool_use') {
        tools.set(index, { id: asString(data.content_block.id) !== '' ? asString(data.content_block.id) : `call_${index}`, name: asString(data.content_block.name), started: false });
      }
    } else if (frame.event === 'content_block_delta') {
      const index = asNumber(data.index);
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        if (!started) {
          started = true;
          yield openAIChunk(model, { role: 'assistant', content: '' }, null);
        }
        yield openAIChunk(model, { content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const tool = tools.get(index);
        if (tool !== undefined) {
          if (!tool.started) {
            tool.started = true;
            yield openAIChunk(model, { tool_calls: [{ index, id: tool.id, type: 'function', function: { name: tool.name, arguments: '' } }] }, null);
          }
          yield openAIChunk(model, { tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null);
        }
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      const stop = typeof delta.stop_reason === 'string' ? anthropicStopToOpenAI(delta.stop_reason) : 'stop';
      if (isRecord(data.usage)) outputTokens = asNumber(data.usage.output_tokens);
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      yield openAIChunk(model, {}, stop);
      yield openAIUsageChunk(model, inputTokens, outputTokens);
      finished = true;
    } else if (frame.event === 'message_stop') {
      yield 'data: [DONE]\n\n';
      doneEmitted = true;
    }
  }
  if (!finished) {
    if (!started) {
      started = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    yield openAIChunk(model, {}, 'stop');
    yield openAIUsageChunk(model, inputTokens, outputTokens);
  }
  if (!doneEmitted) yield 'data: [DONE]\n\n';
}

/**
 * A Gemini :streamGenerateContent SSE stream as OpenAI chat chunks: content
 * deltas, then the finish chunk (from the final frame's finishReason and
 * usageMetadata), a usage chunk and [DONE].
 */
async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let doneEmitted = false;
  let inputTokens = 0;
  let outputTokens = 0;

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const text = geminiCandidateText(candidate);
    if (text !== '') {
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      yield openAIChunk(model, { content: text }, null);
    }
    if (typeof candidate.finishReason === 'string') {
      if (isRecord(data.usageMetadata)) {
        inputTokens = asNumber(data.usageMetadata.promptTokenCount);
        outputTokens = asNumber(data.usageMetadata.candidatesTokenCount);
      }
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      yield openAIChunk(model, {}, geminiFinishToOpenAI(candidate.finishReason));
      yield openAIUsageChunk(model, inputTokens, outputTokens);
      finished = true;
    }
  }
  if (!finished) {
    if (!started) {
      started = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    yield openAIChunk(model, {}, 'stop');
    yield openAIUsageChunk(model, inputTokens, outputTokens);
  }
  if (!doneEmitted) yield 'data: [DONE]\n\n';
}

/**
 * A Gemini :streamGenerateContent SSE stream as the Anthropic event
 * sequence, with the final frame's finishReason and usageMetadata carried
 * into message_delta.
 */
async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textOpen = false;
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = function* (): Generator<string> {
    if (started) return;
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    });
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const text = geminiCandidateText(candidate);
    if (text !== '') {
      yield* startEvents();
      if (!textOpen) {
        textOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    }
    if (typeof candidate.finishReason === 'string') {
      if (isRecord(data.usageMetadata)) {
        inputTokens = asNumber(data.usageMetadata.promptTokenCount);
        outputTokens = asNumber(data.usageMetadata.candidatesTokenCount);
      }
      yield* startEvents();
      if (textOpen) {
        textOpen = false;
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
      }
      yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiFinishToAnthropic(candidate.finishReason), stop_sequence: null }, usage: { output_tokens: outputTokens } });
      yield sseEvent('message_stop', { type: 'message_stop' });
      finished = true;
    }
  }
  if (!finished) {
    yield* startEvents();
    if (textOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

/** One inference call to a provider, in its own wire, with the client's body translated. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientApi === 'openai' ? { ...body, model: candidate.model } : anthropicToOpenAI(body, candidate.model);
    if (stream) {
      payload.stream = true;
      payload.stream_options = { include_usage: true };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    payload = clientApi === 'anthropic' ? { ...body, model: candidate.model } : openAIToAnthropic(body, candidate.model);
    if (stream) {
      payload.stream = true;
      headers.accept = 'text/event-stream';
    }
  } else {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:${method}${stream ? '?alt=sse' : ''}`;
    payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  callsMade[provider.id] = (callsMade[provider.id] ?? 0) + 1;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
}

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** An error in the client's wire. */
function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModelIds();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a translated event stream to the client as it arrives. */
async function pipeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Answer one client request from a provider that answered 2xx: JSON straight
 * through or translated, SSE straight through or translated, in the wire the
 * client used.
 */
async function serveUpstream(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstream: Response,
  stream: boolean,
  res: ServerResponse,
): Promise<void> {
  const wire = candidate.provider.wire;
  const model = candidate.model;
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end('data: [DONE]\n\n');
      return;
    }
    if (clientApi === 'openai' && wire === 'openai') await pipeStream(upstream.body, res);
    else if (clientApi === 'anthropic' && wire === 'anthropic') await pipeStream(upstream.body, res);
    else if (clientApi === 'anthropic' && wire === 'openai') await pipeEvents(openAIStreamToAnthropic(upstream.body, model), res);
    else if (clientApi === 'openai' && wire === 'anthropic') await pipeEvents(anthropicStreamToOpenAI(upstream.body, model), res);
    else if (clientApi === 'openai' && wire === 'gemini') await pipeEvents(geminiStreamToOpenAI(upstream.body, model), res);
    else await pipeEvents(geminiStreamToAnthropic(upstream.body, model), res);
    return;
  }
  const text = await upstream.text();
  if (clientApi === 'openai' && wire === 'openai') {
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(text);
  } catch {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a body that is not JSON');
  }
  if (!isRecord(answer)) {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a JSON value that is not one object');
  }
  if (clientApi === 'anthropic' && wire === 'openai') sendJson(res, 200, openAIJsonToAnthropic(answer, model));
  else if (clientApi === 'openai' && wire === 'anthropic') sendJson(res, 200, anthropicJsonToOpenAI(answer, model));
  else if (clientApi === 'openai' && wire === 'gemini') sendJson(res, 200, geminiJsonToOpenAI(answer, model));
  else sendJson(res, 200, geminiJsonToAnthropic(answer, model));
}

/** One client inference request, served by the best free model that answers. */
async function serveInference(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    // The client still sent this request, so its manifest row is consumed.
    manifestIndex += 1;
    return sendError(res, clientApi, 400, clientApi === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error', 'the request body is not one JSON object');
  }
  const index = manifestIndex;
  const requires = requirementsFor(body, clientApi);
  const stream = body.stream === true;
  const jsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const planned = plannedPicks[index] ?? null;
  const order = attemptOrder(requires, jsonMode, planned);
  let lastError = 'no free model on the directory meets this request';
  for (const candidate of order) {
    let upstream: Response;
    try {
      upstream = await callUpstream(candidate, clientApi, body, stream);
    } catch (err) {
      refusals[candidate.provider.id] = (refusals[candidate.provider.id] ?? 0) + 1;
      consecutiveRefusals[candidate.provider.id] = (consecutiveRefusals[candidate.provider.id] ?? 0) + 1;
      lastError = (err as Error).message;
      log(`request ${index + 1}: ${candidate.provider.id}/${candidate.model} unreachable: ${lastError}`);
      continue;
    }
    if (!upstream.ok) {
      refusals[candidate.provider.id] = (refusals[candidate.provider.id] ?? 0) + 1;
      consecutiveRefusals[candidate.provider.id] = (consecutiveRefusals[candidate.provider.id] ?? 0) + 1;
      lastError = `HTTP ${upstream.status}`;
      try {
        await upstream.text();
      } catch {
        // The refusal's body does not matter.
      }
      log(`request ${index + 1}: ${candidate.provider.id}/${candidate.model} refused with HTTP ${upstream.status}`);
      continue;
    }
    consecutiveRefusals[candidate.provider.id] = 0;
    log(`request ${index + 1}: served by ${candidate.provider.id}/${candidate.model}`);
    await serveUpstream(clientApi, candidate, upstream, stream, res);
    return;
  }
  log(`request ${index + 1}: no provider served it (${lastError})`);
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'server_error', lastError);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveInference(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveInference(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief, probes);
  plannedPicks = planPicks();
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModelIds().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
