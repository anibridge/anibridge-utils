"""Tests for shared background task helpers."""

import asyncio

import pytest

from anibridge.utils.tasks import schedule_task


@pytest.mark.asyncio
async def test_schedule_task_runs_and_tracks_lifecycle() -> None:
    """Scheduled tasks should execute and complete successfully."""
    event = asyncio.Event()

    async def worker() -> None:
        event.set()

    task = schedule_task(worker(), name="worker")

    await asyncio.wait_for(event.wait(), timeout=1)
    await task
    assert task.done()


@pytest.mark.asyncio
async def test_schedule_task_reports_errors() -> None:
    """Error callback should receive task name and exception."""
    seen: dict[str, object] = {}

    async def failing() -> None:
        raise RuntimeError("boom")

    def on_error(name: str, exc: Exception) -> None:
        seen["name"] = name
        seen["exc"] = exc

    task = schedule_task(failing(), name="failing-task", on_error=on_error)
    await task

    assert seen["name"] == "failing-task"
    assert isinstance(seen["exc"], RuntimeError)
