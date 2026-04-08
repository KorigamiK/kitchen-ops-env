---
title: Kitchen Ops Env
sdk: docker
app_port: 8000
tags:
  - openenv
  - restaurant
  - operations
  - simulation
---

# Kitchen Ops Env

`kitchen_ops_env` is a real-world OpenEnv environment for restaurant order fulfillment. The agent operates a compact kitchen where it must manage ingredient quantities, prepare recipe components, recover from stockouts, cook and assemble dishes, and serve orders before deadlines.

## Why this fits the problem statement

- Real-world task: restaurant kitchen execution instead of a game or toy puzzle.
- Full OpenEnv contract: typed models, `reset()` / `step()` / `state()`, and `openenv.yaml`.
- Three graded tasks: easy, medium, hard.
- Meaningful reward: partial progress for prep, cooking, assembly, service, substitutions, and cost-aware recovery.
- Deterministic local bundle: no external dataset fetches or live APIs required for the environment itself.

## Task Set

| Task ID | Difficulty | Scenario |
|---|---|---|
| `breakfast_omelette` | Easy | One breakfast omelette ticket with a tight counter-service deadline |
| `lunch_combo` | Medium | A salad and wrap share fresh produce and must both leave the pass cleanly |
| `dinner_rush_stockout` | Hard | Three dinner orders arrive together while paneer is short, forcing a recovery decision |

## Action Space

The agent emits a typed `KitchenAction` with one of these actions:

- `PREP_COMPONENT`: Prepare the next recipe component for an order using exact bundled ingredient quantities.
- `COOK_DISH`: Cook the dish once its prep components are ready.
- `ASSEMBLE_DISH`: Assemble plated or wrapped dishes after prep/cook prerequisites are satisfied.
- `SERVE_ORDER`: Send the finished order to the guest.
- `RESTOCK_INGREDIENT`: Buy missing inventory from a rush supplier.
- `SUBSTITUTE_INGREDIENT`: Swap an allowed substitute, for example tofu for paneer.
- `CHECK_PROGRESS`: No-op fallback.

## Observation Space

Each `KitchenObservation` contains:

- `service_board`: per-order status, due time, substitutions, missing ingredients, and next component.
- `inventory`: ingredient quantities, units, low-stock flags, and unit costs.
- `prepared_components`: staged items with freshness windows.
- `available_actions`: legal concrete actions the agent can pick immediately.
- `kpis`: orders served, on-time rate, total cost, waste cost, revenue, and action counts.

## Reward and Grading

Step rewards stay within `0.0-1.0` and expose partial progress:

- Successful prep: `0.12`
- Successful cook: `0.18`
- Successful assembly: `0.16`
- Useful restock: `0.08`
- Valid substitution: `0.10`
- Successful service: `0.18-0.32` depending on timeliness and quality penalty

Episode grading stays within `0.0-1.0` and weights:

- Completion rate
- On-time service rate
- Production progress on unfinished tasks
- Cost efficiency
- Waste efficiency
- Substitution quality on the hard task

## Data Grounding

The environment ships with a compact ingredient and recipe bundle inside [`kitchen_ops_env/data`](./kitchen_ops_env/data). Ingredient units and inventory choices are grounded in public food-data conventions, while the recipe graphs themselves are authored for determinism and fast evaluation. More detail is in [DATA_SOURCES.md](./DATA_SOURCES.md).

## Local Setup with UV

```bash
uv python install 3.11
uv sync --managed-python --python 3.11 --extra dev
```

Run the server:

```bash
uv run --managed-python --python 3.11 python -m uvicorn kitchen_ops_env.server.app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
uv run --managed-python --python 3.11 python -m pytest
```

## Inference

The baseline script is named exactly `inference.py` and lives in the repo root as required by the hackathon.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `KITCHEN_ENV_URL` defaults to `http://localhost:8000`

For local development you can copy [`.env.example`](/home/origami/Dev/projects/openenv/kitchen_ops_env/.env.example) to `.env`, but keep the real file untracked.

Run it:

```bash
export API_BASE_URL=https://models.github.ai/inference
export MODEL_NAME=openai/gpt-4o
export HF_TOKEN=...
uv run --managed-python --python 3.11 python inference.py
```

The script prints structured logs in the required format:

- `[START]`
- `[STEP]`
- `[END]`

## Docker and Hugging Face Spaces

Build locally:

```bash
docker build -t kitchen-ops-env:latest .
docker run -p 8000:8000 kitchen-ops-env:latest
```

This repo is ready for Docker-based Hugging Face Spaces deployment because:

- `README.md` contains Space frontmatter
- the root `Dockerfile` starts the FastAPI app
- `openenv.yaml` is included at the repo root

### Hugging Face Space secrets

Do not commit production secrets. In the Space settings, add these as repository secrets or variables instead:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

That keeps the runtime configuration outside the repo while preserving the same `inference.py` contract the hackathon expects.

## Observed Baseline Scores

Validated locally against `https://models.github.ai/inference` with `MODEL_NAME=openai/gpt-4o` on April 8, 2026:

| Task | Steps | Score |
|---|---:|---:|
| `breakfast_omelette` | 5 | 1.000 |
| `lunch_combo` | 9 | 1.000 |
| `dinner_rush_stockout` | 15 | 0.945 |

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grade`
- `GET /health`
- `GET /metadata`

## Notes

- The environment is deterministic for a fixed policy.
- Ingredient quantities are tracked in `piece`, `g`, and `ml`.
- Prepared components can expire before use, turning into waste and lowering the final score.
