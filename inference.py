#!/usr/bin/env python3
"""Baseline inference script for the kitchen operations environment."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

import httpx
from openai import OpenAI

from kitchen_ops_env import KitchenAction, KitchenOpsEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://models.github.ai/inference")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("KITCHEN_ENV_URL", "http://localhost:8000")
USE_LLM_BASELINE = os.getenv("USE_LLM_BASELINE", "0").lower() in {"1", "true", "yes"}
ENV_NAME = "kitchen_ops_env"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if USE_LLM_BASELINE and HF_TOKEN else None


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
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


def _find_shortage(order: dict[str, Any] | None, ingredient_id: str) -> dict[str, Any] | None:
    if order is None:
        return None
    return next(
        (
            shortage
            for shortage in order.get("missing_ingredients", [])
            if shortage.get("ingredient_id") == ingredient_id
            or shortage.get("actual_ingredient_id") == ingredient_id
        ),
        None,
    )


def _heuristic_score(action: dict[str, Any], observation: Any) -> float:
    action_type = action.get("action_type", "CHECK_PROGRESS")
    if action_type == "CHECK_PROGRESS":
        return -1_000_000.0

    order_map = {order["order_id"]: order for order in observation.service_board}
    inventory_map = {item["ingredient_id"]: item for item in observation.inventory}
    prepared_expiry: dict[str, int] = {}
    for prepared in observation.prepared_components:
        order_id = prepared.get("order_id", "")
        expires_in = int(prepared.get("expires_in_steps", 99))
        prepared_expiry[order_id] = min(prepared_expiry.get(order_id, expires_in), expires_in)
    active_prepared_orders = set(prepared_expiry)

    ready_to_serve_orders = {
        order["order_id"] for order in observation.service_board if order.get("status") == "ready_to_serve"
    }
    ready_to_assemble_orders = {
        order["order_id"] for order in observation.service_board if order.get("status") == "ready_to_assemble"
    }
    prep_complete_orders = {
        order["order_id"] for order in observation.service_board if order.get("status") == "prep_complete"
    }
    order = order_map.get(action.get("order_id", ""))

    score = {
        "SERVE_ORDER": 100.0,
        "ASSEMBLE_DISH": 80.0,
        "COOK_DISH": 70.0,
        "SUBSTITUTE_INGREDIENT": 68.0,
        "RESTOCK_INGREDIENT": 58.0,
        "PREP_COMPONENT": 55.0,
    }.get(action_type, 0.0)

    if order is not None:
        order_id = order["order_id"]
        slack = int(order.get("slack_steps", 0))
        score += max(0, 6 - slack) * 6.0
        score += len(order.get("completed_components", [])) * 2.0
        if action_type == "SERVE_ORDER" and order.get("status") == "ready_to_serve":
            score += 25.0
        if action_type == "ASSEMBLE_DISH" and order.get("status") == "ready_to_assemble":
            score += 18.0
        if action_type == "COOK_DISH" and order.get("status") in {"prep_complete", "ready_to_cook"}:
            score += 12.0
        if action_type == "RESTOCK_INGREDIENT" and order.get("status") == "prep_complete":
            score += 16.0
        if action_type == "PREP_COMPONENT" and action.get("component_id") == order.get("next_component_id"):
            score += 8.0
        if order_id in active_prepared_orders:
            score += 24.0
            if action_type == "PREP_COMPONENT" and action.get("component_id") == order.get("next_component_id"):
                score += 12.0
        if order_id in prepared_expiry:
            expiry_pressure = max(0, 4 - prepared_expiry[order_id]) * 10.0
            score += expiry_pressure
            if action_type in {"SERVE_ORDER", "ASSEMBLE_DISH", "COOK_DISH", "RESTOCK_INGREDIENT"}:
                score += 16.0
            elif action_type == "PREP_COMPONENT":
                score -= 8.0
        elif active_prepared_orders and action_type == "PREP_COMPONENT":
            score -= 20.0

    if ready_to_serve_orders and not (
        action_type == "SERVE_ORDER" and action.get("order_id", "") in ready_to_serve_orders
    ):
        score -= 35.0
    elif ready_to_assemble_orders and not (
        action_type in {"ASSEMBLE_DISH", "SERVE_ORDER"} and action.get("order_id", "") in ready_to_assemble_orders
    ):
        score -= 18.0
    elif prep_complete_orders and not (
        action_type in {"COOK_DISH", "RESTOCK_INGREDIENT"} and action.get("order_id", "") in prep_complete_orders
    ):
        score -= 10.0

    if action_type == "SUBSTITUTE_INGREDIENT":
        shortage = _find_shortage(order, action.get("ingredient_id", ""))
        if shortage is not None:
            score += 12.0
            option = next(
                (
                    item
                    for item in shortage.get("allowed_substitutes", [])
                    if item.get("ingredient_id") == action.get("source_id", "")
                ),
                None,
            )
            if option is not None:
                source_cost = float(inventory_map.get(action.get("source_id", ""), {}).get("unit_cost", 0.0))
                target_cost = float(
                    inventory_map.get(action.get("ingredient_id", ""), {}).get("unit_cost", 0.0)
                )
                score += max(0.0, target_cost - source_cost) * 40.0
                score -= float(option.get("quality_penalty", 0.0)) * 100.0
    elif action_type == "RESTOCK_INGREDIENT":
        shortage = _find_shortage(order, action.get("ingredient_id", ""))
        if shortage is not None:
            unit_cost = float(
                inventory_map.get(action.get("ingredient_id", ""), {}).get("unit_cost", 0.0)
            )
            score -= unit_cost * float(action.get("quantity", 0.0)) * 1.5
            if shortage.get("allowed_substitutes"):
                score -= 8.0
        elif order is not None and order.get("status") == "prep_complete":
            score += 10.0
        else:
            score -= 12.0

    return score


def _heuristic_action(observation: Any) -> KitchenAction:
    available_actions = observation.available_actions
    if not available_actions:
        return KitchenAction(action_type="CHECK_PROGRESS")
    ranked = sorted(
        available_actions,
        key=lambda action: (-_heuristic_score(action, observation), _action_identity(action)),
    )
    return _clean_action_payload(ranked[0])


def _llm_action(step: int, observation: Any, history: list[str]) -> KitchenAction | None:
    if client is None:
        return None

    available_actions = observation.available_actions
    if not available_actions:
        return KitchenAction(action_type="CHECK_PROGRESS")

    ranked = sorted(
        available_actions,
        key=lambda action: (-_heuristic_score(action, observation), _action_identity(action)),
    )
    shortlist = ranked[: min(3, len(ranked))]

    system_prompt = textwrap.dedent(
        """
        You are controlling a restaurant kitchen in an OpenEnv benchmark.
        Pick exactly one action from shortlist.
        Prefer:
        1. Serving ready orders before they go late.
        2. The tightest deadline under real slack pressure.
        3. Lower-cost recovery when a substitute preserves service.

        Reply with JSON only:
        {
          "action_type": "...",
          "order_id": "...",
          "component_id": "...",
          "ingredient_id": "...",
          "quantity": 0.0,
          "source_id": "...",
          "notes": ""
        }
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Scenario: {observation.scenario_id}
        KPIs: {json.dumps(observation.kpis, ensure_ascii=True)}
        Service board: {json.dumps(observation.service_board, ensure_ascii=True)}
        Prepared components: {json.dumps(observation.prepared_components, ensure_ascii=True)}
        Shortlist: {json.dumps(shortlist, ensure_ascii=True)}
        Recent history: {history[-4:] if history else []}
        """
    ).strip()

    try:
        response = client.chat.completions.create(
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
        if any(_action_identity(action) == chosen_id for action in shortlist):
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
        response = httpx.get(f"{ENV_URL}/tasks", timeout=10.0)
        response.raise_for_status()
        tasks = response.json().get("tasks", [])
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
    rewards: list[float] = []
    history: list[str] = []

    with KitchenOpsEnv(base_url=ENV_URL).sync() as env:
        result = env.reset(task_id=task_id)
        step_count = 0

        while not result.done:
            step_count += 1
            action = choose_action(step_count, result.observation, history)
            error = None
            try:
                result = env.step(action)
                reward = float(result.reward or 0.0)
            except Exception as exc:
                error = str(exc)
                action = KitchenAction(action_type="CHECK_PROGRESS")
                result = env.step(action)
                reward = float(result.reward or 0.0)

            rewards.append(reward)
            history.append(
                f"step={step_count} action={action.action_type} reward={reward:.2f} task={task_id}"
            )
            log_step(
                step=step_count,
                action=action.action_type,
                reward=reward,
                done=result.done,
                error=error or result.observation.metadata.get("last_error"),
            )

        grade = httpx.get(f"{ENV_URL}/grade", timeout=10.0).json()
        score = float(grade.get("score", 0.0))
        return score > 0.0, step_count, score, rewards


def main() -> None:
    if USE_LLM_BASELINE and not HF_TOKEN:
        print("ERROR: HF_TOKEN must be set when USE_LLM_BASELINE=1", file=sys.stderr)
        sys.exit(1)
    for task_id in fetch_tasks():
        log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)
        success, steps, score, rewards = run_task(task_id)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
