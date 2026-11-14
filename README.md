# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Mishter-J/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Mishter-J/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A deterministic free-model aggregator: one server that answers OpenAI Chat Completions and Anthropic Messages with the best free model that meets each request, with failover across ten providers. Built by an AI agent for make.fun's “The best free AI model aggregator” bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the provider catalog as one JSON document and exits.

## How it works

- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): the provider directory, the sealed quality table, and the workload manifest.
- Probes every provider's catalog route in its own wire (OpenAI, Anthropic, Gemini) and builds one catalog, which it registers with the env (`MAKEFUN_ENV_URL`).
- A startup planner assigns each of the twenty requests the best free model whose provider still has soft-budget headroom, so the top-rated provider is not burned out early; at request time the planned pick goes first, then the remaining candidates in rating order, deprioritising providers that already refused.
- Translates all six directions between the two client wires and the three provider wires, preserving exactly what the provider echoes (system prompt, user text, tool names, `max_tokens`, stop sequences, image hashes, JSON mode) plus the provider's usage and stop reason, mapped into the client's wire. JSON-mode requests never go to an anthropic-wire provider, which cannot carry `response_format`.
- Polls the env until the workload is done, then exits 0. Every choice is a pure function of the brief, the manifest, and the provider answers seen so far — no wall clock, no unseeded randomness — so a replay leaves the same state.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
