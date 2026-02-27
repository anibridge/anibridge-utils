"""Tests for rate limiter utilities."""

import asyncio

import pytest

from anibridge.utils.limiter import Limiter


def test_limiter_rejects_invalid_configuration() -> None:
    """Limiter should validate rate and capacity."""
    with pytest.raises(ValueError, match="rate must be > 0"):
        Limiter(rate=0, capacity=1)

    with pytest.raises(ValueError, match="capacity must be > 0"):
        Limiter(rate=1, capacity=0)


def test_sync_decorator_calls_consume_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synchronous wrapper should consume one token per invocation."""
    limiter = Limiter(rate=10, capacity=1)
    state = {"consumed": 0}

    def fake_consume() -> None:
        state["consumed"] += 1

    monkeypatch.setattr(limiter, "_consume_sync", fake_consume)

    @limiter
    def add_one(value: int) -> int:
        return value + 1

    assert add_one(4) == 5
    assert state["consumed"] == 1


def test_sync_decorator_factory_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limiter should support @limiter() decorator factory usage."""
    limiter = Limiter(rate=10, capacity=1)
    state = {"consumed": 0}

    def fake_consume() -> None:
        state["consumed"] += 1

    monkeypatch.setattr(limiter, "_consume_sync", fake_consume)

    @limiter()
    def double(value: int) -> int:
        return value * 2

    assert double(3) == 6
    assert state["consumed"] == 1


@pytest.mark.asyncio
async def test_async_decorator_calls_consume_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asynchronous wrapper should consume one token per invocation."""
    limiter = Limiter(rate=10, capacity=1)
    state = {"consumed": 0}

    async def fake_consume() -> None:
        state["consumed"] += 1

    monkeypatch.setattr(limiter, "_consume_async", fake_consume)

    @limiter
    async def add_one(value: int) -> int:
        await asyncio.sleep(0)
        return value + 1

    assert await add_one(4) == 5
    assert state["consumed"] == 1


@pytest.mark.asyncio
async def test_disabled_short_circuits_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DISABLED flag should bypass waiting paths."""
    limiter = Limiter(rate=1, capacity=1)
    limiter._tokens = 0

    def fail_sleep(_seconds: float) -> None:
        raise AssertionError("time.sleep should not be called when disabled")

    async def fail_async_sleep(_seconds: float) -> None:
        raise AssertionError("asyncio.sleep should not be called when disabled")

    monkeypatch.setattr("anibridge.utils.limiter.time.sleep", fail_sleep)
    monkeypatch.setattr("anibridge.utils.limiter.asyncio.sleep", fail_async_sleep)

    previous = Limiter.DISABLED
    Limiter.DISABLED = True
    try:
        limiter._consume_sync()
        await limiter._consume_async()
    finally:
        Limiter.DISABLED = previous
