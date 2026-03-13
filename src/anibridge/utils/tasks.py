"""Shared helpers for safely scheduling background tasks."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

__all__ = ["schedule_task"]

_background_tasks: set[asyncio.Task[Any]] = set()


async def _run_task(
    coro: Coroutine[Any, Any, Any],
    *,
    name: str,
    on_error: Callable[[str, Exception], None] | None = None,
) -> None:
    """Run a coroutine and report failures through the error callback."""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if on_error is not None:
            on_error(name, exc)


def schedule_task(
    coro: Coroutine[Any, Any, Any],
    *,
    name: str,
    on_error: Callable[[str, Exception], None] | None = None,
) -> asyncio.Task[Any]:
    """Schedule a coroutine in the background and track task lifecycle.

    Args:
        coro (Coroutine[Any, Any, Any]): The coroutine to execute.
        name (str): A name for the task, used in error reporting.
        on_error (Callable[[str, Exception], None] | None): Optional exception callback.

    Returns:
        asyncio.Task[Any]: The scheduled task instance.
    """
    task = asyncio.create_task(_run_task(coro, name=name, on_error=on_error))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
