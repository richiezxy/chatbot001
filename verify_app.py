"""Local verification checks for TTRPG Studio.

Run with:
    python verify_app.py
"""

from __future__ import annotations

import platform
import py_compile
import sys


def main() -> int:
    print("== Environment ==")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    print("\n== Syntax check ==")
    py_compile.compile("streamlit_app.py", doraise=True)
    print("OK: streamlit_app.py compiles")

    print("\n== OpenAI SDK compatibility ==")
    try:
        import openai  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not import openai: {exc}")
        return 1

    version = getattr(openai, "__version__", "unknown")
    modern = hasattr(openai, "OpenAI")
    legacy = hasattr(openai, "ChatCompletion")

    print(f"openai version: {version}")
    print(f"modern client available (OpenAI): {modern}")
    print(f"legacy client available (ChatCompletion): {legacy}")

    if not modern and not legacy:
        print("FAIL: neither modern nor legacy SDK interface is available")
        return 1

    print("OK: at least one supported SDK interface is available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
