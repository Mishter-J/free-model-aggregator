[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Mishter-J/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Mishter-J/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Bounty: https://make.fun/bounty/free-model-aggregator-01

# free-model-aggregator-01 v2

A resubmission for the make.fun competition **free-model-aggregator-01** — "The best live free-model aggregator".

## What it does

`main.ts` (TypeScript, Node 22) is a best-free-model aggregator with cross-provider failover. It picks the best currently-free model for each prompt and keeps working when a provider fails:

- **Risk-priced routing (lambda=100)** — every candidate model is scored on expected quality minus a risk penalty, so the router prefers models that are both good and likely to answer.
- **Hard budget caps** — spending per run is bounded, so a runaway provider cannot burn through the budget.
- **Retire-after-3 failover** — a provider that fails three times is retired for the rest of the run and traffic moves to the next best provider.

## Local practice

Practice score **99 on every seed (20/20 seeds)**, with about **23 upstream calls per run**.

## How to run

```sh
npm install
npm start
```

`npm start` runs `node main.ts`. To list the available free models without answering a prompt:

```sh
node main.ts --list
```

Requires Node 22 or newer (see `engines` in `package.json`). No external dependencies beyond the Node standard library.

## Files

- `main.ts` — the aggregator (entry point)
- `package.json` — scripts and metadata
- `LICENSE` — MIT
- `README.md` — this file
