"""Kitchen operations environment package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import KitchenAction, KitchenObservation, KitchenState

if TYPE_CHECKING:
    from .client import KitchenOpsEnv

__all__ = ["KitchenAction", "KitchenObservation", "KitchenOpsEnv", "KitchenState"]


def __getattr__(name: str) -> Any:
    if name == "KitchenOpsEnv":
        from .client import KitchenOpsEnv

        return KitchenOpsEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
