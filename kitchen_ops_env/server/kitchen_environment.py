"""Restaurant kitchen environment implementation."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import KitchenAction, KitchenObservation, KitchenState
    from ..scenario_generator import INGREDIENTS, RECIPES, SCENARIOS, TASK_IDS
except ImportError:
    from kitchen_ops_env.models import KitchenAction, KitchenObservation, KitchenState
    from kitchen_ops_env.scenario_generator import INGREDIENTS, RECIPES, SCENARIOS, TASK_IDS


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round2(value: float) -> float:
    return round(float(value), 2)


class KitchenOpsEnvironment(Environment[KitchenAction, KitchenObservation, KitchenState]):
    """A deterministic restaurant-kitchen simulation with ingredient quantities."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._scenario_id = TASK_IDS[0]
        self._scenario: dict[str, Any] = {}
        self._inventory: dict[str, dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._prepared_components: list[dict[str, Any]] = []
        self._food_cost = 0.0
        self._procurement_cost = 0.0
        self._waste_cost = 0.0
        self._revenue = 0.0
        self._idle_actions = 0
        self._productive_actions = 0
        self._valid_actions = 0
        self._invalid_actions = 0
        self._state = KitchenState(episode_id=str(uuid4()), step_count=0)
        self._last_error = ""
        self._last_action: dict[str, Any] = {}
        self._last_reward = 0.0
        self.reset(task_id=self._scenario_id)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = TASK_IDS[0],
        **kwargs: Any,
    ) -> KitchenObservation:
        del seed, kwargs
        self._scenario_id = task_id if task_id in SCENARIOS else TASK_IDS[0]
        self._scenario = deepcopy(SCENARIOS[self._scenario_id])
        self._state = KitchenState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_id=self._scenario_id,
            scenario_description=self._scenario["description"],
            max_steps=self._scenario["max_steps"],
        )
        self._inventory = {
            ingredient_id: {
                **meta,
                "ingredient_id": ingredient_id,
                "quantity": float(self._scenario.get("inventory", {}).get(ingredient_id, 0.0)),
            }
            for ingredient_id, meta in INGREDIENTS.items()
        }
        self._orders = {}
        for raw_order in self._scenario["orders"]:
            self._orders[raw_order["order_id"]] = {
                **deepcopy(raw_order),
                "status": "pending",
                "completed_components": [],
                "cooked": False,
                "assembled": False,
                "served_at": None,
                "substitutions": {},
                "quality_penalty": 0.0,
            }
        self._prepared_components = []
        self._food_cost = 0.0
        self._procurement_cost = 0.0
        self._waste_cost = 0.0
        self._revenue = 0.0
        self._idle_actions = 0
        self._productive_actions = 0
        self._valid_actions = 0
        self._invalid_actions = 0
        self._last_error = ""
        self._last_action = {"action_type": "RESET", "task_id": self._scenario_id}
        self._last_reward = 0.0
        return self._get_observation(reward=0.0, done=False, error="")

    def step(
        self,
        action: KitchenAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> KitchenObservation:
        del timeout_s, kwargs
        self._state.step_count += 1
        self._expire_prepared_components()
        self._last_action = action.model_dump()

        reward = 0.0
        error = ""
        progressed = False
        self._last_error = ""

        if action.action_type == "PREP_COMPONENT":
            reward, error, progressed = self._handle_prep_component(action)
        elif action.action_type == "COOK_DISH":
            reward, error, progressed = self._handle_cook_dish(action)
        elif action.action_type == "ASSEMBLE_DISH":
            reward, error, progressed = self._handle_assemble_dish(action)
        elif action.action_type == "SERVE_ORDER":
            reward, error, progressed = self._handle_serve_order(action)
        elif action.action_type == "RESTOCK_INGREDIENT":
            reward, error, progressed = self._handle_restock_ingredient(action)
        elif action.action_type == "SUBSTITUTE_INGREDIENT":
            reward, error, progressed = self._handle_substitute_ingredient(action)
        elif action.action_type == "CHECK_PROGRESS":
            reward = 0.0
            self._idle_actions += 1
        else:
            error = f"Unsupported action: {action.action_type}"

        if error:
            self._invalid_actions += 1
            self._last_error = error
        else:
            self._valid_actions += 1
            if progressed:
                self._productive_actions += 1

        self._last_reward = reward
        done = self._check_done()
        return self._get_observation(reward=reward, done=done, error=error)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="kitchen_ops_env",
            description="Restaurant kitchen simulation with inventory, prep, cooking, substitutions, and graded service outcomes.",
            version="0.1.0",
        )

    def _handle_prep_component(self, action: KitchenAction) -> tuple[float, str, bool]:
        order, recipe, error = self._get_order_and_recipe(action.order_id)
        if error:
            return 0.0, error, False
        component = self._component_lookup(recipe, action.component_id)
        if component is None:
            return 0.0, f"Unknown component_id: {action.component_id}", False
        if component["component_id"] in order["completed_components"]:
            return 0.0, f"Component already prepared: {component['component_id']}", False
        next_component = self._next_component(order, recipe)
        if next_component is None or next_component["component_id"] != component["component_id"]:
            return 0.0, "Components must be prepared in recipe order", False

        shortages = self._component_shortages(order, recipe, component)
        if shortages:
            return (
                0.0,
                f"Missing ingredient for component: {shortages[0]['actual_ingredient_id']}",
                False,
            )

        prepared_ingredients: list[dict[str, Any]] = []
        component_cost = 0.0
        shelf_life = 99
        for ingredient_req in component["ingredients"]:
            ingredient_use = self._resolved_ingredient_use(order, recipe, ingredient_req)
            ingredient_id = ingredient_use["actual_ingredient_id"]
            quantity = ingredient_use["required_quantity"]
            unit_cost = self._unit_cost(ingredient_id)
            self._inventory[ingredient_id]["quantity"] -= quantity
            component_cost += unit_cost * quantity
            shelf_life = min(shelf_life, self._inventory[ingredient_id]["prep_shelf_life_steps"])
            prepared_ingredients.append(
                {
                    "ingredient_id": ingredient_id,
                    "display_name": self._inventory[ingredient_id]["display_name"],
                    "quantity": _round2(quantity),
                    "unit": self._inventory[ingredient_id]["unit"],
                    "operation": ingredient_req["operation"],
                    "source_ingredient_id": ingredient_req["ingredient_id"],
                }
            )

        self._food_cost += component_cost
        order["completed_components"].append(component["component_id"])
        order["status"] = "prepping"
        self._prepared_components.append(
            {
                "order_id": order["order_id"],
                "dish_id": order["dish_id"],
                "component_id": component["component_id"],
                "display_name": component["display_name"],
                "created_at": self._state.step_count,
                "expires_at": self._state.step_count + shelf_life,
                "component_cost": _round2(component_cost),
                "ingredients": prepared_ingredients,
            }
        )
        self._refresh_order_status(order, recipe)
        return 0.12, "", True

    def _handle_cook_dish(self, action: KitchenAction) -> tuple[float, str, bool]:
        order, recipe, error = self._get_order_and_recipe(action.order_id)
        if error:
            return 0.0, error, False
        if recipe.get("cook_step") is None:
            return 0.0, f"Dish {order['dish_id']} does not require cooking", False
        if order["cooked"]:
            return 0.0, f"Dish already cooked for order {order['order_id']}", False
        if self._next_component(order, recipe) is not None:
            return 0.0, "Finish prep components before cooking", False

        cook_shortages = self._cook_shortages(recipe)
        if cook_shortages:
            return 0.0, f"Missing cook-stage ingredient: {cook_shortages[0]['ingredient_id']}", False

        cook_cost = 0.0
        for cook_ingredient in recipe["cook_step"].get("ingredients", []):
            ingredient_id = cook_ingredient["ingredient_id"]
            quantity = float(cook_ingredient["quantity"])
            self._inventory[ingredient_id]["quantity"] -= quantity
            cook_cost += self._unit_cost(ingredient_id) * quantity
        self._food_cost += cook_cost
        self._prepared_components = [
            component
            for component in self._prepared_components
            if component["order_id"] != order["order_id"]
        ]
        order["cooked"] = True
        self._refresh_order_status(order, recipe)
        return 0.18, "", True

    def _handle_assemble_dish(self, action: KitchenAction) -> tuple[float, str, bool]:
        order, recipe, error = self._get_order_and_recipe(action.order_id)
        if error:
            return 0.0, error, False
        if recipe.get("assemble_step") is None:
            return 0.0, f"Dish {order['dish_id']} does not require assembly", False
        if order["assembled"]:
            return 0.0, f"Dish already assembled for order {order['order_id']}", False
        if self._next_component(order, recipe) is not None:
            return 0.0, "Finish prep components before assembly", False
        if recipe.get("cook_step") is not None and not order["cooked"]:
            return 0.0, "Cook the dish before assembly", False

        self._prepared_components = [
            component
            for component in self._prepared_components
            if component["order_id"] != order["order_id"]
        ]
        order["assembled"] = True
        self._refresh_order_status(order, recipe)
        return 0.16, "", True

    def _handle_serve_order(self, action: KitchenAction) -> tuple[float, str, bool]:
        order, recipe, error = self._get_order_and_recipe(action.order_id)
        if error:
            return 0.0, error, False
        if order["served_at"] is not None:
            return 0.0, f"Order already served: {order['order_id']}", False
        if recipe.get("assemble_step") is not None:
            if not order["assembled"]:
                return 0.0, "Assemble the dish before serving", False
        elif recipe.get("cook_step") is not None:
            if not order["cooked"]:
                return 0.0, "Cook the dish before serving", False
        elif self._next_component(order, recipe) is not None:
            return 0.0, "Prep components before serving", False

        order["served_at"] = self._state.step_count
        order["status"] = "served"
        self._revenue += float(recipe["sell_price"])
        on_time = order["served_at"] <= int(order["due_by_step"])
        base_reward = 0.32 if on_time else 0.18
        reward = max(0.1, base_reward - float(order["quality_penalty"]))
        self._refresh_order_status(order, recipe)
        return _round2(reward), "", True

    def _handle_restock_ingredient(self, action: KitchenAction) -> tuple[float, str, bool]:
        ingredient_id = action.ingredient_id
        if ingredient_id not in self._inventory:
            return 0.0, f"Unknown ingredient_id: {ingredient_id}", False
        if action.quantity <= 0:
            return 0.0, "Restock quantity must be positive", False

        was_short = self._ingredient_is_short(ingredient_id)
        self._inventory[ingredient_id]["quantity"] += action.quantity
        self._procurement_cost += self._unit_cost(ingredient_id) * action.quantity * 0.3
        return (0.08 if was_short else 0.03), "", was_short

    def _handle_substitute_ingredient(self, action: KitchenAction) -> tuple[float, str, bool]:
        order, recipe, error = self._get_order_and_recipe(action.order_id)
        if error:
            return 0.0, error, False
        ingredient_id = action.ingredient_id
        if ingredient_id in order["substitutions"]:
            return 0.0, f"Ingredient already substituted for order {order['order_id']}", False

        allowed = recipe.get("substitutions", {}).get(ingredient_id, [])
        selected = next(
            (option for option in allowed if option["ingredient_id"] == action.source_id),
            None,
        )
        if selected is None:
            return 0.0, f"Invalid substitute {action.source_id} for {ingredient_id}", False

        required_qty = self._required_quantity_for_ingredient(order, recipe, ingredient_id)
        substitute_qty = required_qty * float(selected.get("ratio", 1.0))
        if self._inventory[selected["ingredient_id"]]["quantity"] < substitute_qty:
            return 0.0, f"Insufficient substitute stock: {selected['ingredient_id']}", False

        order["substitutions"][ingredient_id] = {
            "ingredient_id": selected["ingredient_id"],
            "ratio": float(selected.get("ratio", 1.0)),
            "quality_penalty": float(selected.get("quality_penalty", 0.0)),
        }
        order["quality_penalty"] += float(selected.get("quality_penalty", 0.0))
        self._refresh_order_status(order, recipe)
        return 0.1, "", True

    def _get_order_and_recipe(
        self, order_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
        if order_id not in self._orders:
            return None, None, f"Unknown order_id: {order_id}"
        order = self._orders[order_id]
        recipe = RECIPES[order["dish_id"]]
        if order["served_at"] is not None:
            return None, None, f"Order already served: {order_id}"
        return order, recipe, ""

    def _component_lookup(
        self, recipe: dict[str, Any], component_id: str
    ) -> dict[str, Any] | None:
        return next(
            (component for component in recipe["components"] if component["component_id"] == component_id),
            None,
        )

    def _next_component(
        self, order: dict[str, Any], recipe: dict[str, Any]
    ) -> dict[str, Any] | None:
        for component in recipe["components"]:
            if component["component_id"] not in order["completed_components"]:
                return component
        return None

    def _resolved_ingredient_use(
        self, order: dict[str, Any], recipe: dict[str, Any], ingredient_req: dict[str, Any]
    ) -> dict[str, Any]:
        original_id = ingredient_req["ingredient_id"]
        substitution = order["substitutions"].get(original_id)
        if substitution is None:
            return {
                "actual_ingredient_id": original_id,
                "required_quantity": float(ingredient_req["quantity"]),
                "ratio": 1.0,
            }
        return {
            "actual_ingredient_id": substitution["ingredient_id"],
            "required_quantity": float(ingredient_req["quantity"]) * float(substitution["ratio"]),
            "ratio": float(substitution["ratio"]),
        }

    def _required_quantity_for_ingredient(
        self, order: dict[str, Any], recipe: dict[str, Any], ingredient_id: str
    ) -> float:
        total = 0.0
        for component in recipe["components"]:
            if component["component_id"] in order["completed_components"]:
                continue
            for ingredient_req in component["ingredients"]:
                if ingredient_req["ingredient_id"] == ingredient_id:
                    total += float(ingredient_req["quantity"])
        return total

    def _component_shortages(
        self, order: dict[str, Any], recipe: dict[str, Any], component: dict[str, Any]
    ) -> list[dict[str, Any]]:
        shortages: list[dict[str, Any]] = []
        for ingredient_req in component["ingredients"]:
            resolved = self._resolved_ingredient_use(order, recipe, ingredient_req)
            actual_ingredient_id = resolved["actual_ingredient_id"]
            required_quantity = float(resolved["required_quantity"])
            available_quantity = float(self._inventory[actual_ingredient_id]["quantity"])
            if available_quantity + 1e-9 < required_quantity:
                allowed_substitutes = []
                for option in recipe.get("substitutions", {}).get(ingredient_req["ingredient_id"], []):
                    substitute_qty = float(ingredient_req["quantity"]) * float(option.get("ratio", 1.0))
                    if self._inventory[option["ingredient_id"]]["quantity"] >= substitute_qty:
                        allowed_substitutes.append(option)
                shortages.append(
                    {
                        "ingredient_id": ingredient_req["ingredient_id"],
                        "actual_ingredient_id": actual_ingredient_id,
                        "required_quantity": _round2(required_quantity),
                        "available_quantity": _round2(available_quantity),
                        "missing_quantity": _round2(required_quantity - available_quantity),
                        "allowed_substitutes": allowed_substitutes,
                    }
                )
        return shortages

    def _cook_shortages(self, recipe: dict[str, Any]) -> list[dict[str, Any]]:
        shortages: list[dict[str, Any]] = []
        for ingredient_req in recipe.get("cook_step", {}).get("ingredients", []):
            ingredient_id = ingredient_req["ingredient_id"]
            required_quantity = float(ingredient_req["quantity"])
            available_quantity = float(self._inventory[ingredient_id]["quantity"])
            if available_quantity + 1e-9 < required_quantity:
                shortages.append(
                    {
                        "ingredient_id": ingredient_id,
                        "required_quantity": _round2(required_quantity),
                        "available_quantity": _round2(available_quantity),
                        "missing_quantity": _round2(required_quantity - available_quantity),
                    }
                )
        return shortages

    def _ingredient_is_short(self, ingredient_id: str) -> bool:
        for order in self._orders.values():
            if order["served_at"] is not None:
                continue
            recipe = RECIPES[order["dish_id"]]
            next_component = self._next_component(order, recipe)
            shortages = []
            if next_component is not None:
                shortages = self._component_shortages(order, recipe, next_component)
            elif recipe.get("cook_step") is not None and not order["cooked"]:
                shortages = self._cook_shortages(recipe)
            if any(
                shortage.get("actual_ingredient_id", shortage.get("ingredient_id")) == ingredient_id
                for shortage in shortages
            ):
                return True
        return False

    def _expire_prepared_components(self) -> None:
        still_fresh: list[dict[str, Any]] = []
        for prepared in self._prepared_components:
            if self._state.step_count > int(prepared["expires_at"]):
                order = self._orders.get(prepared["order_id"])
                if order is not None and prepared["component_id"] in order["completed_components"]:
                    order["completed_components"].remove(prepared["component_id"])
                    recipe = RECIPES[order["dish_id"]]
                    self._refresh_order_status(order, recipe)
                self._waste_cost += float(prepared["component_cost"])
            else:
                still_fresh.append(prepared)
        self._prepared_components = still_fresh

    def _refresh_order_status(self, order: dict[str, Any], recipe: dict[str, Any]) -> None:
        if order["served_at"] is not None:
            order["status"] = "served"
            return
        if recipe.get("assemble_step") is not None and order["assembled"]:
            order["status"] = "ready_to_serve"
            return
        if recipe.get("assemble_step") is None and recipe.get("cook_step") is not None and order["cooked"]:
            order["status"] = "ready_to_serve"
            return
        if recipe.get("cook_step") is not None and order["cooked"]:
            order["status"] = "ready_to_assemble"
            return
        if self._next_component(order, recipe) is None:
            if recipe.get("cook_step") is None and recipe.get("assemble_step") is not None:
                order["status"] = "ready_to_assemble"
            else:
                order["status"] = "prep_complete"
            return
        if order["completed_components"]:
            order["status"] = "prepping"
            return
        order["status"] = "pending"

    def _action_payload(
        self,
        action_type: str,
        order_id: str = "",
        component_id: str = "",
        ingredient_id: str = "",
        quantity: float = 0.0,
        source_id: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "action_type": action_type,
            "order_id": order_id,
            "component_id": component_id,
            "ingredient_id": ingredient_id,
            "quantity": _round2(quantity),
            "source_id": source_id,
            "notes": notes,
        }

    def _action_sort_key(self, action: dict[str, Any]) -> tuple[bool, str]:
        signature = "|".join(
            (
                self._scenario_id,
                str(self._state.step_count),
                action.get("action_type", ""),
                action.get("order_id", ""),
                action.get("component_id", ""),
                action.get("ingredient_id", ""),
                str(action.get("quantity", 0.0)),
                action.get("source_id", ""),
            )
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
        return (action.get("action_type") == "CHECK_PROGRESS", digest)

    def _available_actions(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for order in sorted(self._orders.values(), key=lambda item: (item["due_by_step"], item["order_id"])):
            if order["served_at"] is not None:
                continue
            recipe = RECIPES[order["dish_id"]]
            self._refresh_order_status(order, recipe)

            if recipe.get("assemble_step") is not None and order["assembled"]:
                actions.append(self._action_payload("SERVE_ORDER", order_id=order["order_id"]))
                continue

            if recipe.get("assemble_step") is None and recipe.get("cook_step") is not None and order["cooked"]:
                actions.append(self._action_payload("SERVE_ORDER", order_id=order["order_id"]))
                continue

            if recipe.get("assemble_step") is not None:
                ready_to_assemble = self._next_component(order, recipe) is None and (
                    recipe.get("cook_step") is None or order["cooked"]
                )
                if ready_to_assemble and not order["assembled"]:
                    actions.append(self._action_payload("ASSEMBLE_DISH", order_id=order["order_id"]))
                    continue

            if recipe.get("cook_step") is not None and self._next_component(order, recipe) is None and not order["cooked"]:
                cook_shortages = self._cook_shortages(recipe)
                if cook_shortages:
                    shortage = cook_shortages[0]
                    actions.append(
                        self._action_payload(
                            "RESTOCK_INGREDIENT",
                            order_id=order["order_id"],
                            ingredient_id=shortage["ingredient_id"],
                            quantity=shortage["missing_quantity"],
                            source_id="rush_supplier",
                        )
                    )
                else:
                    actions.append(self._action_payload("COOK_DISH", order_id=order["order_id"]))
                continue

            next_component = self._next_component(order, recipe)
            if next_component is None:
                continue

            shortages = self._component_shortages(order, recipe, next_component)
            if shortages:
                shortage = shortages[0]
                for option in shortage["allowed_substitutes"]:
                    actions.append(
                        self._action_payload(
                            "SUBSTITUTE_INGREDIENT",
                            order_id=order["order_id"],
                            component_id=next_component["component_id"],
                            ingredient_id=shortage["ingredient_id"],
                            quantity=shortage["required_quantity"],
                            source_id=option["ingredient_id"],
                        )
                    )
                actions.append(
                    self._action_payload(
                        "RESTOCK_INGREDIENT",
                        order_id=order["order_id"],
                        component_id=next_component["component_id"],
                        ingredient_id=shortage["actual_ingredient_id"],
                        quantity=shortage["missing_quantity"],
                        source_id="rush_supplier",
                    )
                )
                continue

            actions.append(
                self._action_payload(
                    "PREP_COMPONENT",
                    order_id=order["order_id"],
                    component_id=next_component["component_id"],
                )
            )

        actions.append(self._action_payload("CHECK_PROGRESS"))
        actions.sort(key=self._action_sort_key)
        return actions

    def _service_board(self) -> list[dict[str, Any]]:
        board: list[dict[str, Any]] = []
        for order in sorted(self._orders.values(), key=lambda item: (item["due_by_step"], item["order_id"])):
            recipe = RECIPES[order["dish_id"]]
            next_component = self._next_component(order, recipe)
            shortages = (
                self._component_shortages(order, recipe, next_component)
                if next_component is not None
                else []
            )
            board.append(
                {
                    "order_id": order["order_id"],
                    "guest": order["guest"],
                    "dish_id": order["dish_id"],
                    "dish_name": recipe["display_name"],
                    "due_by_step": order["due_by_step"],
                    "slack_steps": int(order["due_by_step"]) - self._state.step_count,
                    "status": order["status"],
                    "completed_components": list(order["completed_components"]),
                    "total_components": len(recipe["components"]),
                    "cooked": order["cooked"],
                    "assembled": order["assembled"],
                    "served_at": order["served_at"],
                    "substitutions": deepcopy(order["substitutions"]),
                    "quality_penalty": _round2(order["quality_penalty"]),
                    "next_component_id": next_component["component_id"] if next_component else "",
                    "missing_ingredients": shortages,
                }
            )
        return board

    def _inventory_snapshot(self) -> list[dict[str, Any]]:
        snapshot = []
        for ingredient_id, item in sorted(self._inventory.items()):
            snapshot.append(
                {
                    "ingredient_id": ingredient_id,
                    "display_name": item["display_name"],
                    "category": item["category"],
                    "quantity": _round2(item["quantity"]),
                    "unit": item["unit"],
                    "low_stock": item["quantity"] <= item["restock_pack"] * 0.25,
                    "unit_cost": _round2(self._unit_cost(ingredient_id)),
                }
            )
        return snapshot

    def _prepared_snapshot(self) -> list[dict[str, Any]]:
        snapshot = []
        for prepared in sorted(
            self._prepared_components,
            key=lambda item: (item["expires_at"], item["order_id"], item["component_id"]),
        ):
            snapshot.append(
                {
                    "order_id": prepared["order_id"],
                    "dish_id": prepared["dish_id"],
                    "component_id": prepared["component_id"],
                    "display_name": prepared["display_name"],
                    "expires_in_steps": int(prepared["expires_at"]) - self._state.step_count,
                    "component_cost": _round2(prepared["component_cost"]),
                    "ingredients": deepcopy(prepared["ingredients"]),
                }
            )
        return snapshot

    def _kpis(self) -> dict[str, Any]:
        served_orders = sum(1 for order in self._orders.values() if order["served_at"] is not None)
        on_time_orders = sum(
            1
            for order in self._orders.values()
            if order["served_at"] is not None and order["served_at"] <= int(order["due_by_step"])
        )
        return {
            "orders_total": len(self._orders),
            "orders_served": served_orders,
            "orders_on_time": on_time_orders,
            "orders_late": sum(
                1
                for order in self._orders.values()
                if order["served_at"] is not None and order["served_at"] > int(order["due_by_step"])
            ),
            "food_cost": _round2(self._food_cost),
            "procurement_cost": _round2(self._procurement_cost),
            "total_cost": _round2(self._food_cost + self._procurement_cost),
            "waste_cost": _round2(self._waste_cost),
            "revenue": _round2(self._revenue),
            "invalid_actions": self._invalid_actions,
            "idle_actions": self._idle_actions,
            "productive_actions": self._productive_actions,
            "valid_actions": self._valid_actions,
        }

    def score_components(self) -> dict[str, float]:
        total_orders = max(1, len(self._orders))
        served_orders = [order for order in self._orders.values() if order["served_at"] is not None]
        completion_ratio = len(served_orders) / total_orders
        on_time_ratio = (
            sum(1 for order in served_orders if order["served_at"] <= int(order["due_by_step"])) / total_orders
        )
        steps_taken = max(1, self._state.step_count)
        target_total_cost = float(self._scenario.get("target_total_cost", 1.0))
        target_waste_cost = float(self._scenario.get("target_waste_cost", 0.1))
        total_cost = self._food_cost + self._procurement_cost
        raw_cost_efficiency = _clip01(
            1.0 - max(0.0, total_cost - target_total_cost) / max(target_total_cost, 0.01)
        )
        raw_waste_efficiency = _clip01(1.0 - (self._waste_cost / max(target_waste_cost, 0.01)))

        production_progress = 0.0
        for order in self._orders.values():
            recipe = RECIPES[order["dish_id"]]
            completed = len(order["completed_components"])
            total = len(recipe["components"]) + 1
            if recipe.get("cook_step") is not None:
                total += 1
                completed += int(order["cooked"])
            if recipe.get("assemble_step") is not None:
                total += 1
                completed += int(order["assembled"])
            completed += int(order["served_at"] is not None)
            production_progress += completed / total
        production_progress /= total_orders

        progress_gate = max(completion_ratio, production_progress)
        cost_efficiency = _clip01(raw_cost_efficiency * progress_gate)
        waste_efficiency = _clip01(raw_waste_efficiency * progress_gate)
        execution_accuracy = _clip01(self._productive_actions / steps_taken)

        lateness_steps = 0.0
        for order in self._orders.values():
            reference_step = (
                int(order["served_at"]) if order["served_at"] is not None else self._state.step_count
            )
            lateness_steps += max(0, reference_step - int(order["due_by_step"]))
        raw_lateness_efficiency = _clip01(1.0 - (lateness_steps / max(1.0, total_orders * 2.0)))
        lateness_efficiency = _clip01(raw_lateness_efficiency * max(progress_gate, execution_accuracy))

        substitution_quality = 1.0
        preferred = self._scenario.get("preferred_recovery")
        if preferred:
            order = self._orders[preferred["order_id"]]
            selected = order["substitutions"].get(preferred["ingredient_id"])
            if selected and selected["ingredient_id"] == preferred["preferred_substitute"]:
                substitution_quality = _clip01(1.0 - float(selected.get("quality_penalty", 0.0)))
            elif selected:
                substitution_quality = _clip01(0.8 - float(selected.get("quality_penalty", 0.0)))
            elif order["served_at"] is not None:
                substitution_quality = 0.6
            else:
                substitution_quality = 0.0

        return {
            "completion_ratio": _round2(completion_ratio),
            "on_time_ratio": _round2(on_time_ratio),
            "production_progress": _round2(production_progress),
            "cost_efficiency": _round2(cost_efficiency),
            "waste_efficiency": _round2(waste_efficiency),
            "execution_accuracy": _round2(execution_accuracy),
            "lateness_efficiency": _round2(lateness_efficiency),
            "substitution_quality": _round2(substitution_quality),
        }

    def grade_episode(self) -> float:
        components = self.score_components()
        if self._scenario_id == "breakfast_omelette":
            score = (
                0.45 * components["completion_ratio"]
                + 0.25 * components["on_time_ratio"]
                + 0.15 * components["execution_accuracy"]
                + 0.10 * components["waste_efficiency"]
                + 0.05 * components["lateness_efficiency"]
            )
        elif self._scenario_id == "lunch_combo":
            score = (
                0.30 * components["completion_ratio"]
                + 0.20 * components["on_time_ratio"]
                + 0.15 * components["production_progress"]
                + 0.10 * components["cost_efficiency"]
                + 0.10 * components["waste_efficiency"]
                + 0.10 * components["execution_accuracy"]
                + 0.05 * components["lateness_efficiency"]
            )
        else:
            score = (
                0.25 * components["completion_ratio"]
                + 0.15 * components["on_time_ratio"]
                + 0.15 * components["production_progress"]
                + 0.15 * components["substitution_quality"]
                + 0.10 * components["cost_efficiency"]
                + 0.10 * components["waste_efficiency"]
                + 0.05 * components["execution_accuracy"]
                + 0.05 * components["lateness_efficiency"]
            )
        return round(_clip01(score), 3)

    def _get_observation(
        self, reward: float, done: bool, error: str
    ) -> KitchenObservation:
        components = self.score_components()
        observation = KitchenObservation(
            scenario_id=self._scenario_id,
            scenario_description=self._scenario["description"],
            current_step=self._state.step_count,
            max_steps=self._scenario["max_steps"],
            service_board=self._service_board(),
            inventory=self._inventory_snapshot(),
            prepared_components=self._prepared_snapshot(),
            available_actions=self._available_actions(),
            kpis=self._kpis(),
            done=done,
            reward=_round2(reward),
            metadata={
                "scenario_title": self._scenario["title"],
                "difficulty": self._scenario["difficulty"],
                "last_error": error,
                "last_action": deepcopy(self._last_action),
                "score_preview": self.grade_episode(),
                "score_components": components,
            },
        )
        self._state.scenario_id = self._scenario_id
        self._state.scenario_description = self._scenario["description"]
        self._state.max_steps = self._scenario["max_steps"]
        self._state.inventory = deepcopy(self._inventory)
        self._state.orders = deepcopy(self._orders)
        self._state.prepared_components = deepcopy(self._prepared_components)
        self._state.kpis = deepcopy(self._kpis())
        self._state.score_components = deepcopy(components)
        return observation

    def _unit_cost(self, ingredient_id: str) -> float:
        item = self._inventory[ingredient_id]
        return float(item["restock_cost"]) / float(item["restock_pack"])

    def _check_done(self) -> bool:
        if self._state.step_count >= int(self._scenario["max_steps"]):
            return True
        return all(order["served_at"] is not None for order in self._orders.values())

    @property
    def state(self) -> KitchenState:
        self._state.scenario_id = self._scenario_id
        self._state.scenario_description = self._scenario["description"]
        self._state.max_steps = self._scenario["max_steps"]
        self._state.inventory = deepcopy(self._inventory)
        self._state.orders = deepcopy(self._orders)
        self._state.prepared_components = deepcopy(self._prepared_components)
        self._state.kpis = deepcopy(self._kpis())
        self._state.score_components = deepcopy(self.score_components())
        return self._state
