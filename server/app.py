"""Top-level OpenEnv server entrypoint."""

try:
    from kitchen_ops_env.server.app import app as app
except ImportError:  # pragma: no cover - used when importing from parent dir
    from kitchen_ops_env.kitchen_ops_env.server.app import app as app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
