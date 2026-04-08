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
TASKS = ["breakfast_omelette", "lunch_combo", "dinner_rush_stockout"]
ENV_NAME = "kitchen_ops_env"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN must be set", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


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


def _fallback_action(observation: Any) -> KitchenAction:
    available = observation.available_actions
    if not available:
        return KitchenAction(action_type="CHECK_PROGRESS")
    return _clean_action_payload(available[0])


def choose_action(step: int, observation: Any, history: list[str]) -> KitchenAction:
    available_actions = observation.available_actions
    if not available_actions:
        return KitchenAction(action_type="CHECK_PROGRESS")

    system_prompt = textwrap.dedent(
        """
        You are controlling a restaurant kitchen in an OpenEnv benchmark.
        Choose exactly one action from the available_actions list.
        Priorities:
        1. Serve ready orders before they go late.
        2. Finish the nearest due order first.
        3. Avoid unnecessary restocks and extra waste.
        4. Use a valid substitute when it is cheaper and keeps the order on time.

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
        Available actions: {json.dumps(available_actions, ensure_ascii=True)}
        Recent history: {history[-4:] if history else []}
        """
    ).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
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
        for action in available_actions:
            if (
                action.get("action_type") == chosen.action_type
                and action.get("order_id", "") == chosen.order_id
                and action.get("component_id", "") == chosen.component_id
                and action.get("ingredient_id", "") == chosen.ingredient_id
                and action.get("source_id", "") == chosen.source_id
            ):
                return chosen
    except Exception:
        pass

    return _fallback_action(observation)


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
    for task_id in TASKS:
        log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)
        success, steps, score, rewards = run_task(task_id)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
