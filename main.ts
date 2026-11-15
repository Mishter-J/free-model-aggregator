// SPDX-License-Identifier: MIT
//
// Free-model aggregator, second entry.
//
// One long-running server that answers OpenAI Chat Completions and Anthropic
// Messages requests (JSON and server-sent events) using only the free models
// of the ten providers the trial's brief names. It picks the strongest free
// model that meets each request, spreads calls across providers so no
// provider's free-tier budget is burned early, and fails over to the next
// best model the moment a provider refuses.
//
// Design notes (all of them follow from the trial's own rules):
//   - The brief is the only input: provider directory, quality table and
//     workload manifest. Every routing choice is a function of the brief,
//     the requests seen so far and the providers' answers; there is no
//     wall clock and no unseeded randomness anywhere in a request path.
//   - A provider's first call in a trial cannot fail (its failure schedule
//     starts at call 2 and its outage starts at call 2 or later), so the
//     first call to a fresh provider is free. Later calls carry a failure
//     risk that grows as the provider's budget is spent; the router prices
//     that risk into every pick.
//   - A provider never gets more calls than its smallest possible budget,
//     so a budget refusal is impossible by construction.
//   - A refusal is retried once on the same provider (a scheduled failure
//     is usually a single call number), then the request moves to the next
//     best candidate. Three failures in a row retire the provider: a healthy
//     one holds at most two scheduled failures, so three in a row means it
//     is in outage, and the remaining providers have capacity to spare.
//   - Every upstream answer is buffered and re-rendered in the wire the
//     client used, with the provider's usage counts and stop reason carried
//     over exactly, and with the request's system prompt, user text, tools,
//     stop sequences, max_tokens, images and JSON mode passed through in
//     the upstream wire's own spelling, so the provider's echo matches
//     what the client sent.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Rec = Record<string, unknown>;

const isRec = (v: unknown): v is Rec => v !== null && typeof v === 'object' && !Array.isArray(v);
const asStr = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const asNum = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** Protocol version the env expects in the register body. */
const PROTOCOL_VERSION = 1;
/** Fixed `created` for every OpenAI-shaped model listing: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** Fixed `created_at` for every Anthropic-shaped model listing. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** How long one provider catalog probe may take. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take. */
const CALL_TIMEOUT_MS = 5000;
/** Most attempts at one client request (first pick + retries + fallbacks). */
const MAX_ATTEMPTS = 10;
/** How often GET /work is polled. */
const POLL_MS = 100;
/** Most bytes of one client request body. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/**
 * Consecutive failures after which a provider is retired for the trial.
 * Three is the provable line: a healthy provider holds at most two
 * scheduled failures, so three refused calls in a row means it is in
 * outage (its budget can never be spent, the call cap sees to that).
 * Two is too eager: small providers draw their two failures on adjacent
 * call numbers, and retiring them strands requests only they can serve.
 */
const RETIRE_AFTER_FAILS = 3;
/**
 * Rating milli-points one expected extra provider call is worth. A call is
 * only a tie-break, so this is deliberately large: the router spends
 * rating freely to save calls, until the loss budget below is threatened.
 */
const LAMBDA_MILLI_PER_CALL = 100;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  path: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Need {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  needs: Need;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
  problem: string | null;
}

interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
  problem: string | null;
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

/**
 * Smallest budget any trial can draw for a provider (the generator draws
 * between this and the matching maximum). Capping calls here means a
 * budget refusal can never happen: the drawn budget is never smaller.
 */
const CALL_CAP: Record<string, number> = {
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

/** Midpoint of the drawn-budget range, used only to price failure risk. */
const BUDGET_MID: Record<string, number> = {
  groq: 10.5,
  cerebras: 5.5,
  gemini: 7,
  zai: 4.5,
  openrouter: 11,
  sambanova: 4.5,
  pollinations: 12,
  cloudflare: 7.5,
  ovh: 5.5,
  kilol: 14,
};

const callCap = (providerId: string): number => CALL_CAP[providerId] ?? 4;
const budgetMid = (providerId: string): number => BUDGET_MID[providerId] ?? 6;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
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

/** Every JSON object in `text` carrying a string `schema`, in file order. */
function schemaDocs(text: string): Rec[] {
  const docs: Rec[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRec(value) && typeof value.schema === 'string') docs.push(value);
    } catch {
      // Not a document on its own; keep scanning.
    }
  }
  return docs;
}

function parseProvider(raw: unknown): ProviderEntry | null {
  if (!isRec(raw)) return null;
  const id = asStr(raw.id);
  const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
  if (id === '' || wire === null) return null;
  const auth: Auth =
    raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none'
      ? raw.auth
      : 'bearer';
  const path =
    asStr(raw.path) !== ''
      ? asStr(raw.path)
      : asStr(raw.base_path) !== ''
        ? asStr(raw.base_path)
        : asStr(raw.base_url) !== ''
          ? asStr(raw.base_url)
          : asStr(raw.base);
  return {
    id,
    name: asStr(raw.name) !== '' ? asStr(raw.name) : id,
    wire,
    auth,
    path,
    keyEnv: asStr(raw.key_env),
    key: asStr(raw.key),
  };
}

function parseRating(raw: unknown): RatingRow | null {
  if (!isRec(raw)) return null;
  const model = asStr(raw.model);
  const provider = asStr(raw.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: asNum(raw.rating_milli) ?? 0,
    context: asNum(raw.context_length),
    tools: raw.tools === true,
    vision: raw.vision === true,
    json: raw.json === true,
    free: raw.free === true,
  };
}

function parseManifestRow(raw: unknown): ManifestRow | null {
  if (!isRec(raw)) return null;
  const id = asStr(raw.id);
  if (id === '') return null;
  const needs = isRec(raw.requires) ? raw.requires : {};
  return {
    id,
    api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: raw.stream === true,
    weight: raw.weight === 2 ? 2 : 1,
    needs: {
      tools: needs.tools === true,
      vision: needs.vision === true,
      json: needs.json === true,
      minContext: asNum(needs.min_context),
    },
  };
}

/** The brief at `path`: provider directory, quality table, workload manifest. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], problem: null };
  if (path === '') return { ...empty, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers));
  if (dirDoc === undefined) {
    return { ...empty, problem: 'the brief holds no provider directory' };
  }
  const providers = (dirDoc.providers as unknown[]).flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = parseRating(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest));
  const manifest =
    workloadDoc === undefined
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = parseManifestRow(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, problem: null };
}

/** A base path joined to the env origin; an absolute base is used as-is. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers carrying one provider's key, in its own scheme. */
function keyHeaders(auth: Auth, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One listing's price block: true when every price it names is zero. */
function zeroPriced(entry: Rec): boolean | null {
  const pricing = isRec(entry.pricing) ? entry.pricing : null;
  if (pricing === null) return null;
  const values: number[] = [];
  for (const v of Object.values(pricing)) {
    if (typeof v === 'number') values.push(v);
    else if (typeof v === 'string' && v.trim() !== '') values.push(Number(v));
  }
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a catalog listing wrote it, in its wire's spelling. */
function listedModel(wire: Wire, raw: unknown): ListedModel | null {
  if (!isRec(raw)) return null;
  const id = wire === 'gemini' ? asStr(raw.name).replace(/^models\//, '') : asStr(raw.id);
  if (id === '') return null;
  return {
    id,
    displayName:
      asStr(raw.display_name) !== ''
        ? asStr(raw.display_name)
        : asStr(raw.displayName) !== ''
          ? asStr(raw.displayName)
          : id,
    free: asBool(raw.free) ?? zeroPriced(raw),
    context: asNum(raw.context_length) ?? asNum(raw.context_window) ?? asNum(raw.inputTokenLimit),
    tools: asBool(raw.tools),
    vision: asBool(raw.vision),
    json: asBool(raw.json),
  };
}

/** The models one catalog answer listed, in its wire's shape. */
function listedModels(wire: Wire, body: unknown): ListedModel[] {
  if (!isRec(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating failure. */
async function probeOne(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.path);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...keyHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (!res.ok) return { provider, ok: false, models: [], problem: `GET ${url} answered HTTP ${res.status}` };
      const body: unknown = await res.json();
      return { provider, ok: true, models: listedModels(provider.wire, body), problem: null };
    } catch (err) {
      if (attempt === 1) return { provider, ok: false, models: [], problem: `GET ${url} failed: ${(err as Error).message}` };
    }
  }
  return { provider, ok: false, models: [], problem: `GET ${url} failed` };
}

const ratingRowOf = (brief: Brief, providerId: string, modelId: string): RatingRow | null =>
  brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;

/** Whether a model is free: the quality table first, the listing second. */
function isFreeModel(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Caps {
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table first, the listing second. */
function capsOf(model: ListedModel, row: RatingRow | null): Caps {
  return {
    context: row?.context ?? model.context ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's needs (unknown never meets one). */
function meetsNeeds(needs: Need, caps: Caps): boolean {
  if (needs.tools && caps.tools !== true) return false;
  if (needs.vision && caps.vision !== true) return false;
  if (needs.json && caps.json !== true) return false;
  if (needs.minContext !== null && (caps.context === null || caps.context < needs.minContext)) return false;
  return true;
}

const byModelId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The one catalog: every provider that answered, every model it listed, the
 * provider's own path as its base_url (the env origin differs between a
 * live pass and its replay, so the path is the stable name).
 */
function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFreeModel(model, row),
        context_length: row?.context ?? model.context ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.path,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/* ------------------------------------------------------------------ */
/* Routing                                                             */
/* ------------------------------------------------------------------ */

interface ProviderState {
  calls: number;
  failsInRow: number;
  dead: boolean;
}

interface Candidate {
  providerId: string;
  provider: ProviderEntry;
  wire: Wire;
  modelId: string;
  rating: number;
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
const states = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let st = states.get(providerId);
  if (st === undefined) {
    st = { calls: 0, failsInRow: 0, dead: false };
    states.set(providerId, st);
  }
  return st;
}

/**
 * The chance this provider's next call (its `next`th) is refused: the first
 * call of a trial never is; later calls land on the failure schedule, whose
 * one or two numbers sit inside the drawn budget, and one provider in ten
 * is the designated second-call failure, so a second call carries extra
 * risk. Only the shape matters: cheap providers stay cheap, dear ones dear.
 */
function failChance(providerId: string, next: number): number {
  if (next <= 1) return 0;
  const span = Math.max(1, budgetMid(providerId) - 2);
  let chance = 1.5 / span;
  if (next === 2) chance += 0.1;
  return Math.min(0.95, chance);
}

/** The free models this entry serves, for GET /v1/models. */
function servedModels(): { id: string; provider: string; displayName: string }[] {
  const out: { id: string; provider: string; displayName: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}

/**
 * Every (provider, model) that could serve this request, best first. The
 * order is rating minus the priced risk of the provider's next call, so a
 * slightly weaker model on a fresh provider beats the best model on a
 * provider whose next call is likely to be refused; ties break on rating,
 * then provider, then model, so the order is stable run to run.
 */
function candidatesFor(needs: Need, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    const st = stateOf(probe.provider.id);
    if (st.dead || st.calls >= callCap(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (!meetsNeeds(needs, capsOf(model, row))) continue;
      out.push({
        providerId: probe.provider.id,
        provider: probe.provider,
        wire: probe.provider.wire,
        modelId: model.id,
        rating: row?.rating ?? 0,
      });
    }
  }
  const price = (c: Candidate): number =>
    c.rating - LAMBDA_MILLI_PER_CALL * failChance(c.providerId, stateOf(c.providerId).calls + 1);
  out.sort((a, b) => {
    const byPrice = price(b) - price(a);
    if (byPrice !== 0) return byPrice;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
  return out;
}

/* ------------------------------------------------------------------ */
/* Wire translation: client body -> upstream body                      */
/* ------------------------------------------------------------------ */

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRec(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** Split a data: URL into its media type and base64 payload. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== undefined && m[1] !== '' ? m[1] : 'image/png', data: m[2] ?? '' };
}

/** One Anthropic content block as an OpenAI content part, or null. */
function openAIPart(block: Rec): Rec | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRec(block.source)) {
    const src = block.source;
    if (src.type === 'base64' && typeof src.data === 'string') {
      const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${src.data}` } };
    }
    if (src.type === 'url' && typeof src.url === 'string') return { type: 'image_url', image_url: { url: src.url } };
  }
  return null;
}

/** A tool_result block's content as plain text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object. */
function argsObject(value: unknown): Rec {
  if (isRec(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRec(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request: the
 * system prompt becomes a system message, image blocks become image_url
 * data URLs, tool_use blocks become tool_calls, tool_result blocks become
 * tool messages, and tools, tool_choice, the output cap, temperature,
 * top_p, stop sequences and streaming pass through unchanged.
 */
function anthropicToOpenAI(body: Rec, model: string): Rec {
  const messages: Rec[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRec(raw)) continue;
    const isAssistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: isAssistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (isAssistant) {
      const toolCalls: Rec[] = [];
      for (const block of content) {
        if (!isRec(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: asStr(block.id),
          type: 'function',
          function: {
            name: asStr(block.name),
            arguments: JSON.stringify(isRec(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Rec[] = [];
    for (const block of content) {
      if (!isRec(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: asStr(block.tool_use_id),
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Rec = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRec(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: asStr(raw.description),
            parameters: isRec(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRec(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block, or null. */
function anthropicBlock(part: Rec): Rec | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRec(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: the
 * system messages become the system prompt, image_url parts become image
 * blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, and tools, tool_choice, the output cap, temperature,
 * top_p, stop sequences and streaming pass through unchanged.
 */
function openAIToAnthropic(body: Rec, model: string): Rec {
  const systemParts: string[] = [];
  const messages: Rec[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRec(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: asStr(raw.tool_call_id), content: asStr(raw.content) }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Rec[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRec(call) || !isRec(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: asStr(call.id),
          name: asStr(call.function.name),
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Rec[] = [];
    for (const part of content) {
      if (!isRec(part)) continue;
      const block = anthropicBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Rec = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRec(raw) || !isRec(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: asStr(raw.function.description),
          input_schema: isRec(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRec(choice) && choice.type === 'function' && isRec(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * Either client wire as a Gemini generateContent request: the system prompt
 * becomes the system instruction, messages become contents (images as
 * inline data or file data, tool calls as function calls, tool results as
 * function responses), tools become function declarations, and the output
 * cap, temperature, top_p, stop sequences and JSON mode land in the
 * generation config.
 */
function toGemini(body: Rec, clientApi: 'openai' | 'anthropic'): Rec {
  const contents: Rec[] = [];
  const systemParts: string[] = [];
  const declarations: Rec[] = [];
  let maxTokens: number | null = null;
  let temperature: number | null = null;
  let topP: number | null = null;
  let stops: string[] = [];
  let jsonMode = false;

  const pushUserParts = (parts: Rec[]): void => {
    if (parts.length > 0) contents.push({ role: 'user', parts });
  };

  if (clientApi === 'openai') {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRec(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({
          role: 'user',
          parts: [{ functionResponse: { name: '', response: { content: asStr(raw.content) } } }],
        });
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Rec[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
          if (!isRec(call) || !isRec(call.function)) continue;
          parts.push({
            functionCall: { name: asStr(call.function.name), args: argsObject(call.function.arguments) },
          });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Rec[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRec(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRec(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            const data = dataUrlParts(url);
            if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
            else parts.push({ fileData: { fileUri: url } });
          }
        }
      }
      pushUserParts(parts);
    }
    maxTokens = asNum(body.max_tokens) ?? asNum(body.max_completion_tokens);
    temperature = asNum(body.temperature);
    topP = asNum(body.top_p);
    if (typeof body.stop === 'string') stops = [body.stop];
    else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRec(raw) || !isRec(raw.function) || typeof raw.function.name !== 'string') continue;
        declarations.push({
          name: raw.function.name,
          description: asStr(raw.function.description),
          parameters: isRec(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      }
    }
    const format = isRec(body.response_format) ? body.response_format.type : undefined;
    jsonMode = format === 'json_object' || format === 'json_schema';
  } else {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRec(raw)) continue;
      const isAssistant = raw.role === 'assistant';
      const content = raw.content;
      if (typeof content === 'string') {
        contents.push({ role: isAssistant ? 'model' : 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Rec[] = [];
      for (const block of content) {
        if (!isRec(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRec(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            parts.push({
              inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data },
            });
          } else if (src.type === 'url' && typeof src.url === 'string') {
            parts.push({ fileData: { fileUri: src.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: { name: asStr(block.name), args: isRec(block.input) ? block.input : {} },
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: { name: '', response: { content: toolResultText(block.content) } },
          });
        }
      }
      if (parts.length > 0) contents.push({ role: isAssistant ? 'model' : 'user', parts });
    }
    maxTokens = asNum(body.max_tokens);
    temperature = asNum(body.temperature);
    topP = asNum(body.top_p);
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRec(raw) || typeof raw.name !== 'string') continue;
        declarations.push({
          name: raw.name,
          description: asStr(raw.description),
          parameters: isRec(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }

  const out: Rec = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Rec = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (temperature !== null) config.temperature = temperature;
  if (topP !== null) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* ------------------------------------------------------------------ */
/* Upstream answers, parsed into one shape                             */
/* ------------------------------------------------------------------ */

type StopKind = 'stop' | 'length' | 'tool_calls' | 'content_filter';

interface ToolCallOut {
  id: string;
  name: string;
  args: string;
}

interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: StopKind;
  toolCalls: ToolCallOut[];
}

/** One SSE frame: an optional event name and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const chunk of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** OpenAI's finish_reason as this entry's stop kind. */
function stopOfOpenAI(finish: unknown): StopKind {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** Anthropic's stop_reason as this entry's stop kind. */
function stopOfAnthropic(reason: unknown): StopKind {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as this entry's stop kind. */
function stopOfGemini(reason: unknown): StopKind {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** The text of a Chat Completions message: a string, or its text parts. */
function messageText(message: Rec): string {
  if (typeof message.content === 'string') return message.content;
  if (!Array.isArray(message.content)) return '';
  return message.content
    .flatMap((b) => (isRec(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function openAIToolCalls(message: Rec): ToolCallOut[] {
  const calls: ToolCallOut[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRec(call) || !isRec(call.function)) continue;
    calls.push({
      id: asStr(call.id),
      name: asStr(call.function.name),
      args: typeof call.function.arguments === 'string' ? call.function.arguments : '',
    });
  }
  return calls;
}

/** A Chat Completions JSON body as one answer. */
function parseOpenAIJson(body: unknown): UpstreamAnswer | null {
  if (!isRec(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRec(choices[0]) ? choices[0] : null;
  if (choice === null || !isRec(choice.message)) return null;
  const usage = isRec(body.usage) ? body.usage : {};
  return {
    text: messageText(choice.message),
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    stop: stopOfOpenAI(choice.finish_reason),
    toolCalls: openAIToolCalls(choice.message),
  };
}

/** A Chat Completions event stream as one answer. */
function parseOpenAISse(text: string): UpstreamAnswer | null {
  let outText = '';
  let input = 0;
  let output = 0;
  let stop: StopKind = 'stop';
  let sawAny = false;
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRec(chunk)) continue;
    sawAny = true;
    if (isRec(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRec(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRec(choice.delta) && typeof choice.delta.content === 'string') outText += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = stopOfOpenAI(choice.finish_reason);
  }
  return sawAny ? { text: outText, inputTokens: input, outputTokens: output, stop, toolCalls: [] } : null;
}

/** An Anthropic Messages JSON body as one answer. */
function parseAnthropicJson(body: unknown): UpstreamAnswer | null {
  if (!isRec(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  let text = '';
  const toolCalls: ToolCallOut[] = [];
  for (const block of body.content) {
    if (!isRec(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: asStr(block.id),
        name: asStr(block.name),
        args: JSON.stringify(isRec(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRec(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    stop: stopOfAnthropic(body.stop_reason),
    toolCalls,
  };
}

/** An Anthropic event stream as one answer. */
function parseAnthropicSse(text: string): UpstreamAnswer | null {
  let outText = '';
  let input = 0;
  let output = 0;
  let stop: StopKind = 'stop';
  let sawAny = false;
  const toolArgs = new Map<number, { id: string; name: string; parts: string[] }>();
  for (const frame of parseSse(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRec(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRec(data.message) && isRec(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRec(data.delta)) {
      const index = asNum(data.index) ?? 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') outText += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const slot = toolArgs.get(index);
        if (slot !== undefined) slot.parts.push(data.delta.partial_json);
      }
    } else if (frame.event === 'content_block_start' && isRec(data.content_block)) {
      const index = asNum(data.index) ?? 0;
      if (data.content_block.type === 'tool_use') {
        toolArgs.set(index, { id: asStr(data.content_block.id), name: asStr(data.content_block.name), parts: [] });
      }
    } else if (frame.event === 'message_delta') {
      if (isRec(data.delta) && typeof data.delta.stop_reason === 'string') stop = stopOfAnthropic(data.delta.stop_reason);
      if (isRec(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  const toolCalls: ToolCallOut[] = [...toolArgs.values()].map((t) => ({ id: t.id, name: t.name, args: t.parts.join('') }));
  return sawAny ? { text: outText, inputTokens: input, outputTokens: output, stop, toolCalls } : null;
}

/** A Gemini generateContent JSON body as one answer. */
function parseGeminiJson(body: unknown): UpstreamAnswer | null {
  if (!isRec(body) || !Array.isArray(body.candidates)) return null;
  const candidate = isRec(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return null;
  let text = '';
  if (isRec(candidate.content) && Array.isArray(candidate.content.parts)) {
    for (const part of candidate.content.parts) {
      if (isRec(part) && typeof part.text === 'string') text += part.text;
    }
  }
  const usage = isRec(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    stop: stopOfGemini(candidate.finishReason),
    toolCalls: [],
  };
}

/** A Gemini streamGenerateContent event stream as one answer. */
function parseGeminiSse(text: string): UpstreamAnswer | null {
  let outText = '';
  let input = 0;
  let output = 0;
  let stop: StopKind = 'stop';
  let sawAny = false;
  for (const frame of parseSse(text)) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRec(chunk)) continue;
    sawAny = true;
    if (Array.isArray(chunk.candidates)) {
      for (const candidate of chunk.candidates) {
        if (!isRec(candidate)) continue;
        if (isRec(candidate.content) && Array.isArray(candidate.content.parts)) {
          for (const part of candidate.content.parts) {
            if (isRec(part) && typeof part.text === 'string') outText += part.text;
          }
        }
        if (typeof candidate.finishReason === 'string') stop = stopOfGemini(candidate.finishReason);
      }
    }
    if (isRec(chunk.usageMetadata)) {
      input = numOrZero(chunk.usageMetadata.promptTokenCount);
      output = numOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  return sawAny ? { text: outText, inputTokens: input, outputTokens: output, stop, toolCalls: [] } : null;
}

/* ------------------------------------------------------------------ */
/* One upstream call                                                   */
/* ------------------------------------------------------------------ */

interface CallOutcome {
  ok: boolean;
  status: number;
  retryable: boolean;
  answer: UpstreamAnswer | null;
}

/** Build the upstream request for one candidate and read its answer. */
async function callUpstream(
  cand: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Rec,
  stream: boolean,
): Promise<CallOutcome> {
  const base = joinBase(ENV_URL, cand.provider.path);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...keyHeaders(cand.provider.auth, cand.provider.key),
  };
  let url: string;
  let upstreamBody: Rec;
  if (cand.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: cand.modelId } : anthropicToOpenAI(body, cand.modelId);
    if (stream) {
      headers.accept = 'text/event-stream';
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
    }
  } else if (cand.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: cand.modelId } : openAIToAnthropic(body, cand.modelId);
    if (stream) upstreamBody.stream = true;
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(cand.modelId)}:${action}${stream ? '?alt=sse' : ''}`;
    if (stream) headers.accept = 'text/event-stream';
    upstreamBody = toGemini(body, clientApi);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, retryable: true, answer: null };
  }
  const text = await res.text();
  if (!res.ok) {
    const retryable = res.status === 402 || res.status === 429 || res.status >= 500;
    return { ok: false, status: res.status, retryable, answer: null };
  }
  let answer: UpstreamAnswer | null = null;
  if (cand.wire === 'openai') {
    answer = stream ? parseOpenAISse(text) : parseOpenAIJson(safeJson(text));
  } else if (cand.wire === 'anthropic') {
    answer = stream ? parseAnthropicSse(text) : parseAnthropicJson(safeJson(text));
  } else {
    answer = stream ? parseGeminiSse(text) : parseGeminiJson(safeJson(text));
  }
  if (answer === null) return { ok: false, status: res.status, retryable: false, answer: null };
  return { ok: true, status: res.status, retryable: false, answer };
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

interface Served {
  answer: UpstreamAnswer;
  modelId: string;
}

/**
 * Serve one client request: walk the candidates best-first, retrying a
 * refusal once on the same provider (a scheduled failure is usually one
 * call number) and then moving on. A provider that fails twice in a row
 * is retired for the trial. Returns null only when nothing could serve it.
 */
async function serveRequest(
  clientApi: 'openai' | 'anthropic',
  body: Rec,
  needs: Need,
  jsonMode: boolean,
  stream: boolean,
): Promise<Served | null> {
  let retriesLeft = 1;
  let forced: Candidate | null = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    let cand: Candidate | null = null;
    if (forced !== null) {
      const st = stateOf(forced.providerId);
      if (!st.dead && st.calls < callCap(forced.providerId)) cand = forced;
      forced = null;
    }
    if (cand === null) {
      const list = candidatesFor(needs, jsonMode);
      if (list.length === 0) return null;
      cand = list[0];
    }
    const st = stateOf(cand.providerId);
    const outcome = await callUpstream(cand, clientApi, body, stream);
    st.calls += 1;
    if (outcome.ok && outcome.answer !== null) {
      st.failsInRow = 0;
      return { answer: outcome.answer, modelId: cand.modelId };
    }
    st.failsInRow += 1;
    if (outcome.status === 400 || outcome.status === 401) st.dead = true;
    else if (st.failsInRow >= RETIRE_AFTER_FAILS) st.dead = true;
    if (retriesLeft > 0 && outcome.retryable && !st.dead) {
      retriesLeft -= 1;
      forced = cand;
      continue;
    }
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Rendering: one upstream answer in the client's wire                 */
/* ------------------------------------------------------------------ */

function finishOf(stop: StopKind): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

function stopReasonOf(stop: StopKind): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** One server-sent event frame. */
function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into pieces for streamed deltas. */
function textPieces(text: string, size = 64): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

/** An upstream answer as a Chat Completions JSON body. */
function renderOpenAIJson(answer: UpstreamAnswer, model: string): Rec {
  const message: Rec = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_agg',
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishOf(answer.stop) }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** An upstream answer as a Chat Completions event stream. */
function renderOpenAISse(answer: UpstreamAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Rec, finish: string | null): Rec => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseFrame(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of textPieces(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, finishOf(answer.stop))));
  frames.push(
    sseFrame(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An upstream answer as an Anthropic Messages JSON body. */
function renderAnthropicJson(answer: UpstreamAnswer, model: string): Rec {
  const content: Rec[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_agg',
      name: call.name,
      input: argsObject(call.args),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An upstream answer as an Anthropic event stream. */
function renderAnthropicSse(answer: UpstreamAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseFrame('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of textPieces(answer.text)) {
      frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseFrame('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseFrame('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    }
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/* ------------------------------------------------------------------ */
/* HTTP server                                                         */
/* ------------------------------------------------------------------ */

/** How many client API requests have arrived; the manifest is in order. */
let manifestIndex = 0;

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function openAIError(message: string, type: string): Rec {
  return { error: { message, type, code: null } };
}

function anthropicError(type: string, message: string): Rec {
  return { type: 'error', error: { type, message } };
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
}

function openAIModelList(models: { id: string; provider: string }[]): Rec {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

function anthropicModelList(models: { id: string; displayName: string }[]): Rec {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = servedModels();
  const seen = new Set<string>();
  const unique = models.filter((m) => {
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
  unique.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(unique));
  else sendJson(res, 200, openAIModelList(unique));
}

/** One client request body, or null when it is not one JSON object. */
async function readJsonBody(req: IncomingMessage): Promise<Rec | null> {
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
    return isRec(value) ? value : null;
  } catch {
    return null;
  }
}

/** The request's needs: the manifest's row by arrival order, else the body. */
function needsFor(api: 'openai' | 'anthropic', body: Rec): { needs: Need; jsonMode: boolean } {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    const jsonMode =
      api === 'openai' &&
      isRec(body.response_format) &&
      (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
    return { needs: entry.needs, jsonMode };
  }
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRec(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRec(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const jsonMode =
    api === 'openai' &&
    isRec(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  return {
    needs: {
      tools: Array.isArray(body.tools) && body.tools.length > 0,
      vision,
      json: jsonMode,
      minContext: null,
    },
    jsonMode,
  };
}

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const { needs, jsonMode } = needsFor('openai', body);
  const served = await serveRequest('openai', body, needs, jsonMode, stream);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model meets this request');
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderOpenAISse(served.answer, served.modelId));
    return;
  }
  sendJson(res, 200, renderOpenAIJson(served.answer, served.modelId));
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const { needs, jsonMode } = needsFor('anthropic', body);
  const served = await serveRequest('anthropic', body, needs, jsonMode, stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model meets this request');
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderAnthropicSse(served.answer, served.modelId));
    return;
  }
  sendJson(res, 200, renderAnthropicJson(served.answer, served.modelId));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleMessages(req, res);
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

/** POST /register with the port and the one catalog. */
async function register(port: number): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}`);
      const reply: unknown = await res.json();
      if (isRec(reply) && typeof reply.requests === 'number') {
        log(`registered: the workload holds ${reply.requests} requests`);
      }
      return;
    } catch (err) {
      if (attempt === 4) throw err;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRec(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.problem !== null) log(brief.problem);
  probes = await Promise.all(brief.providers.map((provider) => probeOne(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.problem !== null) log(probe.problem);
  catalog = buildCatalog(brief, probes);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port);
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
