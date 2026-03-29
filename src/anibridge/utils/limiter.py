"""Local rate limiter utilities."""

import asyncio
import functools
import inspect
import threading
import time
from collections.abc import Awaitable, Callable
from typing import ClassVar, ParamSpec, TypeVar, overload

__all__ = ["Limiter"]

P = ParamSpec("P")
R = TypeVar("R")


class Limiter:
    """Token-bucket limiter supporting sync and async call sites."""

    DISABLED: ClassVar[bool] = False  # Switch to disable all limiters during tests

    def __init__(self, rate: float, capacity: int) -> None:
        """Initialize the limiter with a rate and capacity.

        Args:
            rate (float): Tokens added per second.
            capacity (int): Maximum number of tokens in the bucket.
        """
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.rate = float(rate)
        self.capacity = int(capacity)
        self._tokens = float(capacity)
        self._last_check = time.monotonic()
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        """Refill tokens based on elapsed time since last check."""
        elapsed = now - self._last_check
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_check = now

    def _acquire_sync(self) -> None:
        """Block until one token is available and consume it."""
        if Limiter.DISABLED:
            return
        while True:
            with self._sync_lock:
                now = time.monotonic()
                self._refill(now)
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_time = (1 - self._tokens) / self.rate
            time.sleep(wait_time)

    async def _acquire_async(self) -> None:
        """Wait asynchronously until one token is available and consume it."""
        if Limiter.DISABLED:
            return
        while True:
            async with self._async_lock:
                now = time.monotonic()
                self._refill(now)
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_time = (1 - self._tokens) / self.rate
            await asyncio.sleep(wait_time)

    def acquire(self) -> None | Awaitable[None]:
        """Acquire one token, blocking or async depending on the running context.

        Returns an awaitable when called from within a running event loop,
        otherwise blocks synchronously.
        """
        try:
            asyncio.get_running_loop()
            return self._acquire_async()
        except RuntimeError:
            self._acquire_sync()
            return None

    @overload
    def __call__(self) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    @overload
    def __call__(self, func: Callable[P, R]) -> Callable[P, R]: ...

    @overload
    def __call__(
        self, func: Callable[P, Awaitable[R]]
    ) -> Callable[P, Awaitable[R]]: ...

    def __call__(self, func: Callable[P, R] | Callable[P, Awaitable[R]] | None = None):
        """Return a rate-limited wrapper for sync or async callables."""
        if func is None:
            return self
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                await self._acquire_async()
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs):
            self._acquire_sync()
            return func(*args, **kwargs)

        return sync_wrapper
