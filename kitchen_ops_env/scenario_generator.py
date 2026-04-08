"""Scenario templates and generator for kitchen operations tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
INGREDIENTS: dict[str, dict[str, Any]] = json.loads((DATA_DIR / "ingredients.json").read_text())
RECIPES: dict[str, dict[str, Any]] = json.loads((DATA_DIR / "recipes.json").read_text())


TASK_TEMPLATES: dict[str, dict[str, Any]] = {
    "breakfast_omelette": {
        "title": "Breakfast Omelette Ticket",
        "difficulty": "easy",
        "description": "A single breakfast ticket must leave the pass quickly with minimal waste.",
        "orders": [
            {
                "order_id": "ord_easy_1",
                "dish_id": "masala_omelette",
                "guest": "Counter seat 4",
                "due_by_step": 6,
            }
        ],
        "buffer_multiplier": 1.55,
        "ingredient_multipliers": {"egg": 1.3},
        "step_buffer": 3,
    },
    "salad_opening": {
        "title": "Salad Station Opening",
        "difficulty": "easy",
        "description": "Open the cold station cleanly and get one fresh salad out before produce sits too long.",
        "orders": [
            {
                "order_id": "ord_easy_2",
                "dish_id": "garden_salad",
                "guest": "Patio 2",
                "due_by_step": 5,
            }
        ],
        "buffer_multiplier": 1.45,
        "step_buffer": 3,
    },
    "lunch_combo": {
        "title": "Lunch Combo Window",
        "difficulty": "medium",
        "description": "Two lunch orders share fresh produce. Finish both without over-prepping the cold station.",
        "orders": [
            {
                "order_id": "ord_med_1",
                "dish_id": "garden_salad",
                "guest": "Table 7",
                "due_by_step": 7,
            },
            {
                "order_id": "ord_med_2",
                "dish_id": "paneer_wrap",
                "guest": "Table 9",
                "due_by_step": 9,
            },
        ],
        "buffer_multiplier": 1.2,
        "ingredient_multipliers": {"paneer": 1.2, "tortilla": 1.15},
        "step_buffer": 4,
    },
    "wrap_restock_window": {
        "title": "Wrap Restock Window",
        "difficulty": "medium",
        "description": "Two wraps stack up with tight deadlines, but tortillas are short enough to force a clean restock decision.",
        "orders": [
            {
                "order_id": "ord_med_3",
                "dish_id": "paneer_wrap",
                "guest": "Table 11",
                "due_by_step": 8,
            },
            {
                "order_id": "ord_med_4",
                "dish_id": "paneer_wrap",
                "guest": "Table 14",
                "due_by_step": 10,
            },
        ],
        "buffer_multiplier": 1.05,
        "ingredient_multipliers": {"tortilla": 0.55, "paneer": 1.05},
        "step_buffer": 5,
    },
    "dinner_rush_stockout": {
        "title": "Dinner Rush Stockout",
        "difficulty": "hard",
        "description": "Three orders arrive together and the paneer line is short. Recover the rush without blowing food cost.",
        "orders": [
            {
                "order_id": "ord_hard_1",
                "dish_id": "paneer_wrap",
                "guest": "Table 2",
                "due_by_step": 10,
            },
            {
                "order_id": "ord_hard_2",
                "dish_id": "veg_fried_rice",
                "guest": "Table 3",
                "due_by_step": 11,
            },
            {
                "order_id": "ord_hard_3",
                "dish_id": "lentil_soup",
                "guest": "Table 5",
                "due_by_step": 13,
            },
        ],
        "buffer_multiplier": 1.05,
        "ingredient_multipliers": {"paneer": 0.45, "tofu": 1.0},
        "ingredient_additions": {"tofu": 160.0},
        "preferred_recovery": {
            "order_id": "ord_hard_1",
            "ingredient_id": "paneer",
            "preferred_substitute": "tofu",
        },
        "step_buffer": 5,
    },
    "double_shortage_service": {
        "title": "Double Shortage Service",
        "difficulty": "hard",
        "description": "A four-ticket push mixes a protein shortage with low cooking oil, so the line must substitute and restock without slipping deadlines.",
        "orders": [
            {
                "order_id": "ord_hard_4",
                "dish_id": "paneer_wrap",
                "guest": "Table 16",
                "due_by_step": 9,
            },
            {
                "order_id": "ord_hard_5",
                "dish_id": "veg_fried_rice",
                "guest": "Table 18",
                "due_by_step": 10,
            },
            {
                "order_id": "ord_hard_6",
                "dish_id": "garden_salad",
                "guest": "Table 19",
                "due_by_step": 11,
            },
            {
                "order_id": "ord_hard_7",
                "dish_id": "lentil_soup",
                "guest": "Table 22",
                "due_by_step": 13,
            },
        ],
        "buffer_multiplier": 1.0,
        "ingredient_multipliers": {"paneer": 0.4, "oil": 0.35, "tofu": 1.0},
        "ingredient_additions": {"tofu": 200.0},
        "preferred_recovery": {
            "order_id": "ord_hard_4",
            "ingredient_id": "paneer",
            "preferred_substitute": "tofu",
        },
        "step_buffer": 6,
    },
}


def _unit_cost(ingredient_id: str) -> float:
    item = INGREDIENTS[ingredient_id]
    return float(item["restock_cost"]) / float(item["restock_pack"])


def _normalize_quantity(ingredient_id: str, quantity: float) -> float:
    unit = INGREDIENTS[ingredient_id]["unit"]
    quantity = max(0.0, float(quantity))
    if unit == "piece":
        return float(max(0, round(quantity)))
    return round(quantity, 2)


def _recipe_requirements(dish_id: str) -> dict[str, float]:
    recipe = RECIPES[dish_id]
    requirements: dict[str, float] = {}
    for component in recipe["components"]:
        for ingredient in component["ingredients"]:
            ingredient_id = ingredient["ingredient_id"]
            requirements[ingredient_id] = requirements.get(ingredient_id, 0.0) + float(ingredient["quantity"])
    cook_step = recipe.get("cook_step") or {}
    for ingredient in cook_step.get("ingredients", []):
        ingredient_id = ingredient["ingredient_id"]
        requirements[ingredient_id] = requirements.get(ingredient_id, 0.0) + float(ingredient["quantity"])
    return requirements


RECIPE_REQUIREMENTS = {dish_id: _recipe_requirements(dish_id) for dish_id in RECIPES}


def _required_inventory(orders: list[dict[str, Any]]) -> dict[str, float]:
    requirements: dict[str, float] = {}
    for order in orders:
        for ingredient_id, quantity in RECIPE_REQUIREMENTS[order["dish_id"]].items():
            requirements[ingredient_id] = requirements.get(ingredient_id, 0.0) + quantity
    return requirements


def _ideal_food_cost(orders: list[dict[str, Any]]) -> float:
    requirements = _required_inventory(orders)
    return round(
        sum(_unit_cost(ingredient_id) * quantity for ingredient_id, quantity in requirements.items()),
        2,
    )


def _minimum_steps(orders: list[dict[str, Any]]) -> int:
    steps = 0
    for order in orders:
        recipe = RECIPES[order["dish_id"]]
        steps += len(recipe["components"]) + 1
        if recipe.get("cook_step") is not None:
            steps += 1
        if recipe.get("assemble_step") is not None:
            steps += 1
    return steps


def _build_inventory(template: dict[str, Any]) -> dict[str, float]:
    orders = template["orders"]
    required = _required_inventory(orders)
    inventory: dict[str, float] = {}
    buffer_multiplier = float(template.get("buffer_multiplier", 1.1))
    ingredient_multipliers = template.get("ingredient_multipliers", {})
    ingredient_additions = template.get("ingredient_additions", {})
    inventory_overrides = template.get("inventory_overrides", {})

    for ingredient_id, quantity in required.items():
        multiplier = float(ingredient_multipliers.get(ingredient_id, 1.0))
        inventory[ingredient_id] = _normalize_quantity(
            ingredient_id,
            quantity * buffer_multiplier * multiplier,
        )

    for ingredient_id, quantity in ingredient_additions.items():
        inventory[ingredient_id] = _normalize_quantity(
            ingredient_id,
            inventory.get(ingredient_id, 0.0) + float(quantity),
        )

    for ingredient_id, quantity in inventory_overrides.items():
        inventory[ingredient_id] = _normalize_quantity(ingredient_id, float(quantity))

    return {ingredient_id: quantity for ingredient_id, quantity in inventory.items() if quantity > 0}


def _build_targets(template: dict[str, Any]) -> tuple[float, float]:
    ideal_cost = _ideal_food_cost(template["orders"])
    target_total_cost = round(ideal_cost * float(template.get("target_cost_multiplier", 1.18)), 2)
    target_waste_cost = round(max(0.2, ideal_cost * float(template.get("target_waste_multiplier", 0.12))), 2)
    return target_total_cost, target_waste_cost


def generate_scenarios() -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}

    for task_id, template in TASK_TEMPLATES.items():
        target_total_cost, target_waste_cost = _build_targets(template)
        scenarios[task_id] = {
            "title": template["title"],
            "difficulty": template["difficulty"],
            "description": template["description"],
            "max_steps": _minimum_steps(template["orders"]) + int(template.get("step_buffer", 4)),
            "target_total_cost": target_total_cost,
            "target_waste_cost": target_waste_cost,
            "orders": list(template["orders"]),
            "inventory": _build_inventory(template),
        }
        if "preferred_recovery" in template:
            scenarios[task_id]["preferred_recovery"] = dict(template["preferred_recovery"])

    return scenarios


SCENARIOS = generate_scenarios()
TASK_IDS = list(SCENARIOS.keys())
