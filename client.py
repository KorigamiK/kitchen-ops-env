"""Root client export for OpenEnv tooling."""

try:
    from kitchen_ops_env.client import KitchenOpsEnv
except ImportError:  # pragma: no cover - used when importing from parent dir
    from .kitchen_ops_env.client import KitchenOpsEnv

__all__ = ["KitchenOpsEnv"]
