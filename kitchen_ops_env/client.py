"""WebSocket client for the kitchen operations environment."""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import KitchenAction, KitchenObservation, KitchenState


class KitchenOpsEnv(EnvClient[KitchenAction, KitchenObservation, KitchenState]):
    """Client for a running kitchen-ops environment server."""

    def _step_payload(self, action: KitchenAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "order_id": action.order_id,
            "component_id": action.component_id,
            "ingredient_id": action.ingredient_id,
            "quantity": action.quantity,
            "source_id": action.source_id,
            "notes": action.notes,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[KitchenObservation]:
        obs_data = payload.get("observation", {})
        observation = KitchenObservation(
            scenario_id=obs_data.get("scenario_id", ""),
            scenario_description=obs_data.get("scenario_description", ""),
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 0),
            service_board=obs_data.get("service_board", []),
            inventory=obs_data.get("inventory", []),
            prepared_components=obs_data.get("prepared_components", []),
            available_actions=obs_data.get("available_actions", []),
            kpis=obs_data.get("kpis", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> KitchenState:
        return KitchenState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_id=payload.get("scenario_id", ""),
            scenario_description=payload.get("scenario_description", ""),
            max_steps=payload.get("max_steps", 0),
            inventory=payload.get("inventory", {}),
            orders=payload.get("orders", {}),
            prepared_components=payload.get("prepared_components", []),
            kpis=payload.get("kpis", {}),
            score_components=payload.get("score_components", {}),
        )

