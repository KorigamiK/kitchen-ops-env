#!/usr/bin/env python3
"""Baseline inference script for the kitchen operations environment."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any
from urllib import request


@dataclass(slots=True)
class KitchenAction:
    action_type: str
    order_id: str = ""
    component_id: str = ""
    ingredient_id: str = ""
    quantity: float = 0.0
    source_id: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://models.github.ai/inference")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
BENCHMARK = "kitchen_ops_env"
ENV_URL = (
    os.getenv("ENV_URL") or os.getenv("KITCHEN_ENV_URL", "https://korigamik-kitchen-ops-env.hf.space")
).rstrip("/")
SUCCESS_SCORE_THRESHOLD = 0.1

client: Any | None = None
_client_initialized = False


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object response from {url}")
    return data


def _env_endpoint(path: str) -> str:
    return f"{ENV_URL}/{path.lstrip('/')}"


def _score_from_components(task_id: str, components: dict[str, float]) -> float:
    if task_id == "breakfast_omelette":
        score = (
            0.45 * components.get("completion_ratio", 0.0)
            + 0.25 * components.get("on_time_ratio", 0.0)
            + 0.15 * components.get("execution_accuracy", 0.0)
            + 0.10 * components.get("waste_efficiency", 0.0)
            + 0.05 * components.get("lateness_efficiency", 0.0)
        )
    elif task_id == "lunch_combo":
        score = (
            0.30 * components.get("completion_ratio", 0.0)
            + 0.20 * components.get("on_time_ratio", 0.0)
            + 0.15 * components.get("production_progress", 0.0)
            + 0.10 * components.get("cost_efficiency", 0.0)
            + 0.10 * components.get("waste_efficiency", 0.0)
            + 0.10 * components.get("execution_accuracy", 0.0)
            + 0.05 * components.get("lateness_efficiency", 0.0)
        )
    else:
        score = (
            0.25 * components.get("completion_ratio", 0.0)
            + 0.15 * components.get("on_time_ratio", 0.0)
            + 0.15 * components.get("production_progress", 0.0)
            + 0.15 * components.get("substitution_quality", 0.0)
            + 0.10 * components.get("cost_efficiency", 0.0)
            + 0.10 * components.get("waste_efficiency", 0.0)
            + 0.05 * components.get("execution_accuracy", 0.0)
            + 0.05 * components.get("lateness_efficiency", 0.0)
        )
    return max(0.001, min(round(score, 3), 0.999))


def _score_from_state(task_id: str, env: Any) -> float:
    state = env.state()
    components = state.score_components
    if not components:
        raise RuntimeError(
            f"Environment state did not include score components for task '{task_id}'"
        )
    return _score_from_components(task_id, components)


def _get_client() -> Any | None:
    global client, _client_initialized
    if _client_initialized:
        return client

    _client_initialized = True
    if not HF_TOKEN:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return client


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _clean_action_payload(candidate: dict[str, Any]) -> KitchenAction:
    return KitchenAction(
        action_type=candidate.get("action_type", "CHECK_PROGRESS"),
        order_id=candidate.get("order_id", ""),
        component_id=candidate.get("component_id", ""),
        ingredient_id=candidate.get("ingredient_id", ""),
        quantity=float(candidate.get("quantity", 0.0) or 0.0),
        source_id=candidate.get("source_id", ""),
        notes=candidate.get("notes", ""),
    )


def _action_identity(candidate: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        candidate.get("action_type", ""),
        candidate.get("order_id", ""),
        candidate.get("component_id", ""),
        candidate.get("ingredient_id", ""),
        candidate.get("source_id", ""),
    )


def _format_action_for_log(action: KitchenAction) -> str:
    if action.action_type == "PREP_COMPONENT":
        return f"prep_component('{action.order_id}','{action.component_id}')"
    if action.action_type == "COOK_DISH":
        return f"cook_dish('{action.order_id}')"
    if action.action_type == "ASSEMBLE_DISH":
        return f"assemble_dish('{action.order_id}')"
    if action.action_type == "SERVE_ORDER":
        return f"serve_order('{action.order_id}')"
    if action.action_type == "RESTOCK_INGREDIENT":
        return (
            f"restock_ingredient('{action.order_id}','{action.ingredient_id}',"
            f"{action.quantity:.2f},'{action.source_id}')"
        )
    if action.action_type == "SUBSTITUTE_INGREDIENT":
        return (
            f"substitute_ingredient('{action.order_id}','{action.ingredient_id}',"
            f"'{action.source_id}')"
        )
    return "check_progress()"


def _briefing_context(briefing: str) -> dict[str, Any]:
    urgent_order = ""
    match = re.search(r"most urgent is (ord_[A-Za-z0-9_]+)", briefing)
    if match:
        urgent_order = match.group(1)

    order_stages: dict[str, str] = {}
    time_left: dict[str, int] = {}
    blocked_orders: set[str] = set()
    expiring_orders: set[str] = set()
    swapped_orders: set[str] = set()
    in_prepared_section = False

    for raw_line in briefing.splitlines():
        line = raw_line.strip()
        if line == "Prepared items waiting on the line:":
            in_prepared_section = True
            continue
        if line == "Legal moves right now:":
            in_prepared_section = False
            continue

        order_match = re.match(
            r"- (ord_[^:]+): .*?, ([^,]+), prep [^,]+, (\d+) steps left\.",
            line,
        )
        if order_match:
            order_id, stage, steps_left = order_match.groups()
            order_stages[order_id] = stage.strip().lower()
            time_left[order_id] = int(steps_left)
            if "Blocked:" in line:
                blocked_orders.add(order_id)
            if "Current swaps:" in line:
                swapped_orders.add(order_id)
            continue

        if in_prepared_section:
            prepared_match = re.match(r"- (ord_[^:]+): ", line)
            if prepared_match:
                expiring_orders.add(prepared_match.group(1))

    return {
        "urgent_order": urgent_order,
        "order_stages": order_stages,
        "time_left": time_left,
        "blocked_orders": blocked_orders,
        "expiring_orders": expiring_orders,
        "swapped_orders": swapped_orders,
    }


def _heuristic_score(action: dict[str, Any], context: dict[str, Any]) -> float:
    action_type = action.get("action_type", "CHECK_PROGRESS")
    if action_type == "CHECK_PROGRESS":
        return -1_000_000.0

    order_id = action.get("order_id", "")
    order_stages = context["order_stages"]
    time_left = context["time_left"]
    blocked_orders = context["blocked_orders"]
    expiring_orders = context["expiring_orders"]
    swapped_orders = context["swapped_orders"]
    ready_to_serve_orders = {
        item_id for item_id, stage in order_stages.items() if stage == "ready to serve"
    }
    ready_to_assemble_orders = {
        item_id
        for item_id, stage in order_stages.items()
        if stage == "ready to assemble"
    }
    ready_to_cook_orders = {
        item_id for item_id, stage in order_stages.items() if stage == "ready to cook"
    }

    score = {
        "SERVE_ORDER": 100.0,
        "ASSEMBLE_DISH": 80.0,
        "COOK_DISH": 70.0,
        "SUBSTITUTE_INGREDIENT": 68.0,
        "RESTOCK_INGREDIENT": 58.0,
        "PREP_COMPONENT": 55.0,
    }.get(action_type, 0.0)

    if order_id:
        score += max(0, 6 - time_left.get(order_id, 99)) * 6.0
        if order_id == context["urgent_order"]:
            score += 15.0

        stage = order_stages.get(order_id, "")
        if action_type == "SERVE_ORDER" and stage == "ready to serve":
            score += 25.0
        if action_type == "ASSEMBLE_DISH" and stage == "ready to assemble":
            score += 18.0
        if action_type == "COOK_DISH" and stage == "ready to cook":
            score += 12.0
        if action_type == "PREP_COMPONENT" and stage in {"waiting to start", "in prep"}:
            score += 8.0
        if order_id in swapped_orders and action_type in {
            "PREP_COMPONENT",
            "COOK_DISH",
            "ASSEMBLE_DISH",
            "SERVE_ORDER",
        }:
            score += 6.0
        if order_id in expiring_orders:
            if action_type in {
                "SERVE_ORDER",
                "ASSEMBLE_DISH",
                "COOK_DISH",
                "RESTOCK_INGREDIENT",
            }:
                score += 16.0
            elif action_type == "PREP_COMPONENT":
                score -= 8.0
        if order_id in blocked_orders:
            if action_type == "SUBSTITUTE_INGREDIENT":
                score += 18.0
            elif action_type == "RESTOCK_INGREDIENT":
                score += 12.0
            elif action_type == "PREP_COMPONENT":
                score -= 18.0

    if ready_to_serve_orders and not (
        action_type == "SERVE_ORDER"
        and action.get("order_id", "") in ready_to_serve_orders
    ):
        score -= 35.0
    elif ready_to_assemble_orders and not (
        action_type in {"ASSEMBLE_DISH", "SERVE_ORDER"}
        and action.get("order_id", "") in ready_to_assemble_orders
    ):
        score -= 18.0
    elif ready_to_cook_orders and not (
        action_type in {"COOK_DISH", "RESTOCK_INGREDIENT"}
        and action.get("order_id", "") in ready_to_cook_orders
    ):
        score -= 10.0

    if action_type == "SUBSTITUTE_INGREDIENT":
        score += 4.0
    elif action_type == "RESTOCK_INGREDIENT":
        score -= float(action.get("quantity", 0.0) or 0.0) * 0.02
        if action.get("source_id") == "rush_supplier":
            score -= 2.0

    return score


def _heuristic_action(observation: Any) -> KitchenAction:
    available_actions = observation.available_actions
    if not available_actions:
        return KitchenAction(action_type="CHECK_PROGRESS")
    context = _briefing_context(observation.briefing)
    ranked = sorted(
        available_actions,
        key=lambda action: (
            -_heuristic_score(action, context),
            _action_identity(action),
        ),
    )
    return _clean_action_payload(ranked[0])


def _llm_action(
    step: int, observation: Any, history: list[str]
) -> KitchenAction | None:
    llm_client = _get_client()
    if llm_client is None:
        return None
    del step, history

    available_actions = observation.available_actions
    if not available_actions:
        return KitchenAction(action_type="CHECK_PROGRESS")

    system_prompt = (
        "You are the kitchen operator. Read the briefing, choose one legal move, "
        "and reply with JSON only using these keys: "
        "action_type, order_id, component_id, ingredient_id, quantity, source_id, notes."
    )
    user_prompt = (
        observation.briefing or "Choose one legal action and reply with JSON only."
    )

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        candidate = json.loads(content)
        chosen = _clean_action_payload(candidate)
        chosen_id = _action_identity(candidate)
        if any(_action_identity(action) == chosen_id for action in available_actions):
            return chosen
    except Exception:
        return None
    return None


def choose_action(step: int, observation: Any, history: list[str]) -> KitchenAction:
    heuristic = _heuristic_action(observation)
    llm_choice = _llm_action(step, observation, history)
    return llm_choice or heuristic


def fetch_tasks() -> list[str]:
    try:
        tasks = _get_json(_env_endpoint("/tasks"), timeout=10.0).get("tasks", [])
        if tasks:
            return [str(task) for task in tasks]
    except Exception:
        pass
    return [
        "breakfast_omelette",
        "salad_opening",
        "lunch_combo",
        "wrap_restock_window",
        "dinner_rush_stockout",
        "double_shortage_service",
    ]


def run_task(task_id: str) -> tuple[bool, int, float, list[float]]:
    try:
        from kitchen_ops_env import KitchenOpsEnv
    except ImportError as exc:
        raise RuntimeError(
            "KitchenOpsEnv runtime dependencies are not installed; install the project requirements to run benchmark tasks"
        ) from exc

    rewards: list[float] = []
    history: list[str] = []
    step_count = 0
    score = 0.0
    success = False
    result = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with KitchenOpsEnv(base_url=ENV_URL).sync() as env:
            result = env.reset(task_id=task_id)

            while not result.done:
                step_count += 1
                action = choose_action(step_count, result.observation, history)
                action_text = _format_action_for_log(action)
                error = None

                try:
                    result = env.step(action)
                except Exception as exc:
                    error = str(exc)
                    action = KitchenAction(action_type="CHECK_PROGRESS")
                    action_text = _format_action_for_log(action)
                    result = env.step(action)

                reward = float(result.reward or 0.0)
                rewards.append(reward)
                history.append(
                    f"step={step_count} action={action.action_type} reward={reward:.2f} task={task_id}"
                )
                log_step(
                    step=step_count,
                    action=action_text,
                    reward=reward,
                    done=result.done,
                    error=error or result.observation.metadata.get("last_action_error"),
                )

            score = _score_from_state(task_id, env)
            success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=step_count, score=score, rewards=rewards)

    return success, step_count, score, rewards


def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN or API_KEY must be set", file=sys.stderr)
        sys.exit(1)
    if _get_client() is None:
        print(
            "ERROR: openai dependencies are not installed; install the project requirements to enable model inference",
            file=sys.stderr,
        )
        sys.exit(1)
    for task_id in fetch_tasks():
        run_task(task_id)


if __name__ == "__main__":
    main()
