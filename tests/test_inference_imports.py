from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_inference_import_does_not_require_optional_http_clients() -> None:
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"httpx", "openai"}:
                raise ImportError(f"blocked optional import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import

        import inference

        print("ok")
        """
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
