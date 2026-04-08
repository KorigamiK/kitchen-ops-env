---
title: Kitchen Ops Env
sdk: docker
app_port: 8000
base_path: /web
---

# Kitchen Ops Env

OpenEnv environment for restaurant order fulfillment with inventory quantities, prep expiry, substitutions, rush restocking, and deadline-based grading.

## Tasks

| Task ID | Difficulty | Summary |
|---|---|---|
| `breakfast_omelette` | easy | Single fast breakfast ticket |
| `salad_opening` | easy | One cold-station salad with a short freshness window |
| `lunch_combo` | medium | Salad and wrap share produce and deadlines |
| `wrap_restock_window` | medium | Two wraps force a tortilla restock decision |
| `dinner_rush_stockout` | hard | Three concurrent orders with a paneer shortage |
| `double_shortage_service` | hard | Four-ticket rush with paneer and oil pressure |

## Action Space

`KitchenAction` supports:

- `PREP_COMPONENT`
- `COOK_DISH`
- `ASSEMBLE_DISH`
- `SERVE_ORDER`
- `RESTOCK_INGREDIENT`
- `SUBSTITUTE_INGREDIENT`
- `CHECK_PROGRESS`

## Observation Space

`KitchenObservation` exposes:

- `service_board`
- `inventory`
- `prepared_components`
- `available_actions`
- `kpis`

## Reward / Grading

- Step rewards stay in `0.0-1.0`
- Episode scores stay in `0.0-1.0`
- Grading combines completion, timeliness, progress, cost, waste, execution efficiency, and substitution quality
- Idle/no-progress play is heavily penalized
- Tasks are generated from reusable templates instead of fixed full-scenario blobs

## Setup

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

Run the baseline:

```bash
export API_BASE_URL=https://models.github.ai/inference
export MODEL_NAME=openai/gpt-4o
export HF_TOKEN=...
uv run --managed-python --python 3.11 python inference.py
```

`inference.py` is deterministic by default. Set `USE_LLM_BASELINE=1` to let the script ask the configured model to break ties among top heuristic actions.

## Environment Variables

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `KITCHEN_ENV_URL` optional, defaults to `http://localhost:8000`
- `USE_LLM_BASELINE` optional, defaults to `0`

## API

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grade`
- `GET /health`
- `GET /metadata`
