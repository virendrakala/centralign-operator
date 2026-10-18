"""Public entry points for the shared CentrAlign browser-agent runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent_loop import run_computer_use_agent
from documents import ensure_operator_assets, extract_document_text, list_workspace_files
from llm_clients import client_config


def planner_config(*, offline: bool = False, model_override: str | None = None) -> tuple[str, str, str | None]:
    """Return live-provider metadata, or FakeLLM metadata for explicit offline mode."""
    return client_config(offline=offline, model_override=model_override)


async def run_operator_agent(
    data_dir: Path,
    run_id: str,
    task: str,
    base_url: str,
    emit: Callable[[str, dict[str, Any]], None],
    approval_handler: Callable[[dict[str, Any]], Any],
    *,
    approval_mode: str = "policy_driven",
    offline: bool = False,
) -> tuple[str, str]:
    """Compatibility name for the shared browser-agent loop."""
    return await run_computer_use_agent(
        data_dir, run_id, task, base_url, emit, approval_handler,
        approval_mode=approval_mode, offline=offline,
    )


if __name__ == "__main__":
    from web_app import main

    main()
