"""Compatibility wrapper for OpenEnv multi-mode validation."""

from kitchen_ops_env.server.app import app as app
from kitchen_ops_env.server.app import main as _main


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    _main()


if __name__ == "__main__":
    main()
