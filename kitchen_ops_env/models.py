"""Typed models for the kitchen operations environment."""

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class KitchenAction(Action):
    """Action model for restaurant-kitchen operations."""

    action_type: Literal[
        "PREP_COMPONENT",
        "COOK_DISH",
        "ASSEMBLE_DISH",
        "SERVE_ORDER",
        "RESTOCK_INGREDIENT",
        "SUBSTITUTE_INGREDIENT",
        "CHECK_PROGRESS",
    ] = Field(..., description="Type of action to perform in the kitchen")
    order_id: str = Field(default="", description="Target order identifier")
    component_id: str = Field(
        default="", description="Recipe component being prepared or assembled"
    )
    ingredient_id: str = Field(
        default="", description="Ingredient being restocked or substituted"
    )
    quantity: float = Field(
        default=0.0, ge=0.0, description="Ingredient quantity for restock actions"
    )
    source_id: str = Field(
        default="", description="Supplier or substitute ingredient identifier"
    )
    notes: str = Field(default="", description="Optional operator notes")


class KitchenState(State):
    """Internal environment state."""

    scenario_id: str = Field(default="", description="Current scenario/task id")
    scenario_description: str = Field(
        default="", description="Current scenario description"
    )
    max_steps: int = Field(default=0, description="Max steps for the episode")
    inventory: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Inventory snapshot keyed by ingredient id"
    )
    orders: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Order tracker keyed by order id"
    )
    prepared_components: list[dict[str, Any]] = Field(
        default_factory=list, description="Prepared recipe components waiting to be used"
    )
    kpis: dict[str, Any] = Field(
        default_factory=dict, description="Episode KPI summary"
    )
    score_components: dict[str, float] = Field(
        default_factory=dict, description="Latest grading components"
    )


class KitchenObservation(Observation):
    """Observation presented to the agent."""

    scenario_id: str = Field(..., description="Current scenario/task id")
    scenario_description: str = Field(..., description="Scenario description")
    current_step: int = Field(default=0, description="Current kitchen clock step")
    max_steps: int = Field(default=0, description="Episode horizon")
    service_board: list[dict[str, Any]] = Field(
        default_factory=list, description="Orders visible on the pass"
    )
    inventory: list[dict[str, Any]] = Field(
        default_factory=list, description="Inventory snapshot with quantities"
    )
    prepared_components: list[dict[str, Any]] = Field(
        default_factory=list, description="Prepared components and freshness"
    )
    available_actions: list[dict[str, Any]] = Field(
        default_factory=list, description="Concrete legal actions available right now"
    )
    kpis: dict[str, Any] = Field(
        default_factory=dict, description="Revenue, cost, waste, and service metrics"
    )

