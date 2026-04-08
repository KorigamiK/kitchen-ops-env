"""FastAPI app for the kitchen operations environment."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

from ..models import KitchenAction, KitchenObservation
from .kitchen_environment import KitchenOpsEnvironment, SCENARIOS, TASK_IDS

_env: KitchenOpsEnvironment | None = None


def get_env() -> KitchenOpsEnvironment:
    global _env
    if _env is None:
        _env = KitchenOpsEnvironment()
    return _env


def create_kitchen_app() -> FastAPI:
    app = create_app(
        get_env,
        KitchenAction,
        KitchenObservation,
        env_name="kitchen_ops_env",
    )

    @app.get("/", response_class=HTMLResponse)
    def landing_page() -> HTMLResponse:
        env = get_env()
        scenario = SCENARIOS[env.state.scenario_id]
        items = "".join(
            f"<li><code>{task_id}</code> ({SCENARIOS[task_id]['difficulty']})</li>"
            for task_id in TASK_IDS
        )
        return HTMLResponse(
            f"""
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Kitchen Ops Env</title>
                <style>
                  body {{ font-family: system-ui, sans-serif; margin: 32px; color: #10231c; background: #f7f4ea; }}
                  .card {{ max-width: 820px; padding: 28px; border-radius: 20px; background: white; box-shadow: 0 16px 40px rgba(0,0,0,0.08); }}
                  code {{ background: #f2efe6; padding: 2px 6px; border-radius: 6px; }}
                  a {{ color: #1d6f42; }}
                </style>
              </head>
              <body>
                <div class="card">
                  <h1>Kitchen Ops Env</h1>
                  <p>Restaurant kitchen simulation for OpenEnv with inventory quantities, substitutions, food cost, waste, and task-specific grading.</p>
                  <p><strong>Current task:</strong> <code>{env.state.scenario_id}</code> — {scenario['title']}</p>
                  <ul>{items}</ul>
                  <p>
                    <a href="/docs">Swagger UI</a> ·
                    <a href="/health">Health</a> ·
                    <a href="/tasks">Tasks</a> ·
                    <a href="/grade">Grade</a> ·
                    <a href="/openapi.json">OpenAPI</a>
                  </p>
                </div>
              </body>
            </html>
            """.strip()
        )

    @app.get("/tasks")
    def list_tasks() -> dict[str, list[str]]:
        return {"tasks": TASK_IDS}

    @app.get("/grade")
    def grade_episode() -> dict[str, object]:
        env = get_env()
        return {
            "task_id": env.state.scenario_id,
            "score": env.grade_episode(),
            "score_components": env.score_components(),
        }

    return app


app = create_kitchen_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

