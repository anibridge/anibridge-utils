"""Caching decorators for LRU, TTL, and file-based caching. Supports async functions."""

import asyncio
import atexit
import contextlib
import functools
import hashlib
import inspect
import threading
import weakref
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, cast, overload

from cachetools import LRUCache as CachetoolsLRUCache
from cachetools import TTLCache as CachetoolsTTLCache
from diskcache import Cache as DiskCache

__all__ = [
    "CacheDict",
    "CacheInfo",
    "LRUDict",
    "TTLDict",
    "cache",
    "file_cache",
    "lru_cache",
    "ttl_cache",
]

P = ParamSpec("P")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

_UNBOUNDED_MAXSIZE = 2**31 - 1
_MISSING = object()
_DISK_CACHES: set[DiskCache] = set()
_DISK_CACHES_LOCK = threading.RLock()

_global_default_cache_dir: Path = Path(".cache")


def set_default_cache_dir(path: str | Path | None) -> None:
    """Set process-wide default cache dir for file_cache when cache_dir=None.

    Args:
        path (str | Path | None): Directory to use for caching.
    """
    global _global_default_cache_dir
    _global_default_cache_dir = Path(".cache") if path is None else Path(path)


def get_default_cache_dir() -> Path:
    """Resolve default cache dir at runtime (setter > env var > fallback)."""
    return _global_default_cache_dir


def _register_disk_cache(cache: DiskCache) -> None:
    with _DISK_CACHES_LOCK:
        _DISK_CACHES.add(cache)


def _close_disk_cache(cache: DiskCache) -> None:
    with contextlib.suppress(Exception):
        cache.close()


def _close_all_disk_caches() -> None:
    with _DISK_CACHES_LOCK:
        caches = tuple(_DISK_CACHES)
        _DISK_CACHES.clear()
    for cache in caches:
        _close_disk_cache(cache)


atexit.register(_close_all_disk_caches)


class CachedFunction(Protocol[P, T]):
    """Protocol for cached functions with cache management methods."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call the cached function."""
        ...

    def cache_clear(self) -> None:
        """Clear the cache."""
        ...

    def cache_info(self) -> CacheInfo:
        """Get cache information."""
        ...

    @overload
    def __get__(
        self: CachedFunction[Concatenate[object, P], T],
        instance: object,
        owner: type[Any] | None = None,
    ) -> BoundCachedFunction[T]: ...

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[Any] | None = None,
    ) -> CachedFunction[P, T]: ...


class CachedAsyncFunction(Protocol[P, T]):
    """Protocol for cached async functions with cache management methods."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
        """Call the cached async function."""
        ...

    def cache_clear(self) -> None:
        """Clear the cache."""
        ...

    def cache_info(self) -> CacheInfo:
        """Get cache information."""
        ...

    @overload
    def __get__(
        self: CachedAsyncFunction[Concatenate[object, P], T],
        instance: object,
        owner: type[Any] | None = None,
    ) -> BoundCachedAsyncFunction[T]: ...

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[Any] | None = None,
    ) -> CachedAsyncFunction[P, T]: ...


class BoundCachedFunction(Protocol[T]):
    """Protocol for bound cached sync methods."""

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...

    def cache_clear(self) -> None: ...

    def cache_info(self) -> CacheInfo: ...


class BoundCachedAsyncFunction(Protocol[T]):
    """Protocol for bound cached async methods."""

    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[T]: ...

    def cache_clear(self) -> None: ...

    def cache_info(self) -> CacheInfo: ...


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Cache statistics snapshot."""

    hits: int
    misses: int
    maxsize: int | None
    currsize: int
    ttl: float | None = None


class CacheDict[K, V](MutableMapping[K, V]):
    """Dictionary-like wrapper over a cache mapping."""

    def __init__(self, backing: Any) -> None:
        """Initialize with a backing cache-like mapping object."""
        self._cache = backing
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def __getitem__(self, key: K) -> V:
        """Return the value for `key` and record hit/miss stats."""
        with self._lock:
            try:
                value = self._cache[key]
            except KeyError:
                self._misses += 1
                raise
            self._hits += 1
            return cast(V, value)

    def __setitem__(self, key: K, value: V) -> None:
        """Store `value` under `key`."""
        with self._lock:
            self._cache[key] = value

    def __delitem__(self, key: K) -> None:
        """Remove `key` from the cache."""
        with self._lock:
            del self._cache[key]

    def __contains__(self, key: object) -> bool:
        """Return whether `key` exists in the cache."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Return number of currently cached entries."""
        with self._lock:
            return len(self._cache)

    def __iter__(self):
        """Iterate over a stable snapshot of cache keys."""
        with self._lock:
            return iter(tuple(self._cache.keys()))

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: T) -> V | T: ...

    def get(self, key: K, default: Any = None) -> V | Any:
        """Return `key` value if present, otherwise `default`."""
        with self._lock:
            value = self._cache.get(key, _MISSING)
            if value is _MISSING:
                self._misses += 1
                return default
            self._hits += 1
            return cast(V, value)

    @overload
    def pop(self, key: K) -> V: ...

    @overload
    def pop(self, key: K, default: T) -> V | T: ...

    def pop(self, key: K, default: Any = _MISSING) -> V | Any:
        """Remove and return `key` value, optionally using `default`."""
        with self._lock:
            if default is _MISSING:
                return cast(V, self._cache.pop(key))
            return cast(V | Any, self._cache.pop(key, default))

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._cache.clear()

    def keys(self):
        """Return a snapshot of cache keys."""
        with self._lock:
            return tuple(self._cache.keys())

    def values(self):
        """Return a snapshot of cache values."""
        with self._lock:
            return tuple(self._cache.values())

    def items(self):
        """Return a snapshot of cache items."""
        with self._lock:
            return tuple(self._cache.items())

    def cache_info(self) -> CacheInfo:
        """Return hit/miss stats and cache sizing information."""
        with self._lock:
            return CacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=getattr(self._cache, "maxsize", None),
                currsize=len(self._cache),
            )


class TTLDict[K, V](CacheDict[K, V]):
    """TTL-based dictionary using `cachetools.TTLCache`."""

    def __init__(self, ttl: float = 300, maxsize: int | None = None) -> None:
        """Initialize a TTL dictionary."""
        resolved_maxsize = _UNBOUNDED_MAXSIZE if maxsize is None else maxsize
        self.ttl = ttl
        super().__init__(CachetoolsTTLCache(maxsize=resolved_maxsize, ttl=ttl))

    def cache_info(self) -> CacheInfo:
        """Return cache stats including configured TTL value."""
        info = super().cache_info()
        return CacheInfo(
            hits=info.hits,
            misses=info.misses,
            maxsize=info.maxsize,
            currsize=info.currsize,
            ttl=self.ttl,
        )


class LRUDict[K, V](CacheDict[K, V]):
    """LRU-based dictionary using `cachetools.LRUCache`."""

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize an LRU dictionary."""
        super().__init__(CachetoolsLRUCache(maxsize=maxsize))


def _freeze_for_key(obj: Any, _visited_ids: set[int] | None = None) -> Any:
    if _visited_ids is None:
        _visited_ids = set()

    obj_id = id(obj)
    if obj_id in _visited_ids:
        raise TypeError

    try:
        hash(obj)
        return obj
    except TypeError:
        _visited_ids.add(obj_id)
        try:
            if isinstance(obj, tuple):
                return tuple(_freeze_for_key(item, _visited_ids) for item in obj)
            if isinstance(obj, list):
                return (
                    "__list__",
                    tuple(_freeze_for_key(item, _visited_ids) for item in obj),
                )
            if isinstance(obj, set):
                return (
                    "__set__",
                    frozenset(_freeze_for_key(item, _visited_ids) for item in obj),
                )
            if isinstance(obj, dict):
                return (
                    "__dict__",
                    tuple(
                        sorted(
                            (
                                _freeze_for_key(k, _visited_ids),
                                _freeze_for_key(v, _visited_ids),
                            )
                            for k, v in obj.items()
                        )
                    ),
                )
            raise
        finally:
            _visited_ids.discard(obj_id)


def _generic_hash(obj: Any, _visited_ids: set[int] | None = None) -> int:
    """Generate a hash for arbitrary objects, handling unhashable types."""
    return hash(_freeze_for_key(obj, _visited_ids))


def _make_key(
    args: tuple[Any, ...], kwargs: dict[str, Any], strict: bool = True
) -> int | tuple[Any, ...] | None:
    """Generate a cache key from args and kwargs. Supports unhashable types."""
    if strict:
        try:
            key = (args, tuple(sorted(kwargs.items())))
            hash(key)
            return key
        except TypeError:
            return None

    try:
        key = _freeze_for_key((args, kwargs))
        hash(key)
        return key
    except Exception:
        return None


class _InFlightCoalescer(dict[Any, asyncio.Future]):
    """Dict subclass for pending async computations.

    Avoids duplicating the coalescing pattern across every async wrapper.
    All access must be done while holding the caller's lock.
    """

    def start(self, cache_key: Any) -> tuple[bool, asyncio.Future]:
        """Register interest in *cache_key*.

        Returns `(should_compute, future)`.  When *should_compute* is True
        the caller is responsible for resolving the future.
        """
        pending = self.get(cache_key)
        if pending is not None:
            return False, pending
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_future_exception)
        self[cache_key] = future
        return True, future

    def resolve(self, cache_key: Any, value: Any) -> None:
        """Resolve a pending future with *value* and remove it."""
        active = self.pop(cache_key, None)
        if active is not None and not active.done():
            active.set_result(value)

    def reject(self, cache_key: Any, exc: BaseException) -> None:
        """Reject a pending future with *exc* and remove it."""
        active = self.pop(cache_key, None)
        if active is not None and not active.done():
            active.set_exception(exc)


def _consume_future_exception(future: asyncio.Future) -> None:
    """Mark internal coordination future exceptions as observed.

    The caller task does not await that future itself, so without consuming the stored
    exception here the event loop can later emit "Future exception was never retrieved"
    warnings even though the original caller already handled the raised exception.
    """
    with contextlib.suppress(asyncio.CancelledError, Exception):
        future.exception()


class _CacheDescriptor:
    """Descriptor that gives each class instance its own cache.

    When a cached decorator is applied to a method and `per_instance=True`,
    this descriptor is returned instead of a plain wrapper.  On first access
    via an instance, `__get__` builds a new wrapper with a fresh cache and
    stashes it in `instance.__dict__` so subsequent lookups bypass the
    descriptor entirely.

    **Limitations that are detected:**

    - `__slots__` classes without `__dict__` will raise `TypeError` during instantiation

    **Limitations to be aware of:**

    - `pickle` / `copy.deepcopy` will not preserve the per-instance cache.
    """

    def __init__(
        self,
        factory: Callable[..., Any],
        func: Callable[..., Any],
        wrapper: Any,
    ) -> None:
        """Initialize the descriptor with a cache factory and an unbound wrapper."""
        self._factory = factory
        self._func = func
        self._wrapper = wrapper
        self._attr_name: str | None = None
        functools.update_wrapper(self, func)

    def __set_name__(self, owner: type, name: str) -> None:
        """Store the attribute name this descriptor is assigned to for caching."""
        self._attr_name = name
        # Fail fast if the class won't support per-instance caching.
        if "__dict__" not in dir(owner) and "__dict__" not in getattr(
            owner, "__slots__", ()
        ):
            raise TypeError(
                f"Cannot use per-instance caching on {owner.__qualname__} "
                f"because it uses __slots__ without __dict__.  Either add "
                f"'__dict__' to __slots__ or use per_instance=False."
            )

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        """Return a cache wrapper bound to `instance`, creating it if needed."""
        if instance is None:
            return self  # class-level access returns the descriptor

        attr = self._attr_name or self._func.__name__  # ty:ignore[unresolved-attribute]

        # Already created for this instance.
        try:
            return instance.__dict__[attr]
        except KeyError:
            pass

        # Build a fresh cache wrapper bound to this instance.
        bound_func = self._func.__get__(instance, owner)  # ty:ignore[unresolved-attribute]
        per_instance_wrapper = self._factory(bound_func)
        functools.update_wrapper(per_instance_wrapper, self._func)

        instance.__dict__[attr] = per_instance_wrapper
        return per_instance_wrapper

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Support unbound calls on the descriptor itself."""
        # Called when someone does `MyClass.method(instance, ...)`.
        return self._wrapper(*args, **kwargs)

    def cache_clear(self) -> None:
        """Clear the class-level (shared) cache.

        To clear the per-instance cache, call `instance.method.cache_clear()`.
        """
        self._wrapper.cache_clear()

    def cache_info(self) -> CacheInfo:
        """Return cache info for the class-level (shared) cache.

        To get info for the per-instance cache, call `instance.method.cache_info()`.
        """
        return self._wrapper.cache_info()


def _make_sync_wrapper(
    func: Callable[..., Any],
    cache: Any,
    lock: threading.RLock,
    get_cache_key: Callable[..., Any],
    make_info: Callable[[int, int], CacheInfo],
) -> Any:
    """Build a sync cached wrapper.  Used by `lru_cache`, `ttl_cache`, `file_cache`."""
    hits = 0
    misses = 0

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal hits, misses
        cache_key = get_cache_key(args, kwargs)
        if cache_key is None:
            return func(*args, **kwargs)

        with lock:
            cached = cache.get(cache_key, _MISSING)
            if cached is not _MISSING:
                hits += 1
                return cached
            misses += 1

        result = func(*args, **kwargs)

        with lock:
            existing = cache.get(cache_key, _MISSING)
            if existing is not _MISSING:
                return existing
            cache[cache_key] = result
        return result

    def cache_clear() -> None:
        nonlocal hits, misses
        with lock:
            cache.clear()
            hits = 0
            misses = 0

    def cache_info() -> CacheInfo:
        with lock:
            return make_info(hits, misses)

    wrapper.cache_clear = cache_clear  # ty:ignore[unresolved-attribute]
    wrapper.cache_info = cache_info  # ty:ignore[unresolved-attribute]
    return wrapper


def _make_async_wrapper(
    func: Callable[..., Any],
    cache: Any,
    lock: threading.RLock,
    get_cache_key: Callable[..., Any],
    make_info: Callable[[int, int], CacheInfo],
) -> Any:
    """Build an async cached wrapper with in-flight coalescing."""
    hits = 0
    misses = 0
    in_flight = _InFlightCoalescer()

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal hits, misses
        cache_key = get_cache_key(args, kwargs)
        if cache_key is None:
            return await func(*args, **kwargs)

        with lock:
            cached = cache.get(cache_key, _MISSING)
            if cached is not _MISSING:
                hits += 1
                return cached

            should_compute, future = in_flight.start(cache_key)
            if should_compute:
                misses += 1

        if not should_compute:
            return await asyncio.shield(future)

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            with lock:
                in_flight.reject(cache_key, exc)
            raise

        with lock:
            existing = cache.get(cache_key, _MISSING)
            if existing is not _MISSING:
                final = existing
            else:
                cache[cache_key] = result
                final = result
            in_flight.resolve(cache_key, final)

        return final

    def cache_clear() -> None:
        nonlocal hits, misses
        with lock:
            cache.clear()
            hits = 0
            misses = 0

    def cache_info() -> CacheInfo:
        with lock:
            return make_info(hits, misses)

    wrapper.cache_clear = cache_clear  # ty:ignore[unresolved-attribute]
    wrapper.cache_info = cache_info  # ty:ignore[unresolved-attribute]
    return wrapper


@overload
def lru_cache(
    maxsize: int = 128,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = True,
) -> Callable[[Callable[P, T]], CachedFunction[P, T]]: ...


@overload
def lru_cache(
    maxsize: int = 128,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], CachedAsyncFunction[P, T]]: ...


def lru_cache(
    maxsize: int = 128,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = True,
) -> Callable[
    [Callable[P, T] | Callable[P, Awaitable[T]]],
    CachedFunction[P, T] | CachedAsyncFunction[P, T],
]:
    """LRU cache decorator for both sync and async functions.

    Args:
        maxsize (int): Maximum number of cached items.
        key (Callable | None): Optional function to generate cache key from args/kwargs.
            Should accept the same arguments as the decorated function and return a
            hashable key.
        per_instance (bool): When True (default), each class instance gets its own
            independent cache.

    Returns:
        Decorator: Decorated callable with LRU caching.

    Example:
        @lru_cache(maxsize=256)
        async def fetch_data(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

        # Custom key that only considers the first argument
        @lru_cache(maxsize=100, key=lambda user_id, include_details=False: user_id)
        async def get_user(user_id, include_details=False):
            return await fetch_user_data(user_id, include_details)

        # Works with sync functions too
        @lru_cache(maxsize=50, key=lambda x, y, z=None: (x, y))
        def compute(x, y, z=None):
            return x + y
    """

    def _get_cache_key(args: tuple, kwargs: dict) -> Any:
        if key is not None:
            try:
                return key(*args, **kwargs)
            except Exception:
                return None
        return _make_key(args, kwargs, strict=False)

    is_async_func: bool | None = None

    def _build_wrapper(func: Callable) -> Any:
        """Build a fresh wrapper with its own LRU cache around *func*."""
        cache = CachetoolsLRUCache(maxsize=maxsize)
        lock = threading.RLock()

        def make_info(hits: int, misses: int) -> CacheInfo:
            return CacheInfo(
                hits=hits,
                misses=misses,
                maxsize=maxsize,
                currsize=len(cache),
            )

        nonlocal is_async_func
        if is_async_func is None:
            is_async_func = inspect.iscoroutinefunction(func)

        if is_async_func:
            return _make_async_wrapper(func, cache, lock, _get_cache_key, make_info)
        return _make_sync_wrapper(func, cache, lock, _get_cache_key, make_info)

    def decorator(
        func: Callable[P, T] | Callable[P, Awaitable[T]],
    ) -> CachedFunction[P, T] | CachedAsyncFunction[P, T]:
        nonlocal is_async_func
        is_async_func = inspect.iscoroutinefunction(func)

        wrapper = _build_wrapper(func)

        if per_instance:
            return cast(
                CachedFunction[P, T] | CachedAsyncFunction[P, T],
                _CacheDescriptor(factory=_build_wrapper, func=func, wrapper=wrapper),
            )
        return cast(CachedFunction[P, T] | CachedAsyncFunction[P, T], wrapper)

    return decorator


@overload
def ttl_cache(
    ttl: float = 300,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = True,
) -> Callable[[Callable[P, T]], CachedFunction[P, T]]: ...


@overload
def ttl_cache(
    ttl: float = 300,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], CachedAsyncFunction[P, T]]: ...


def ttl_cache(
    ttl: float = 300,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = True,
) -> Callable[
    [Callable[P, T] | Callable[P, Awaitable[T]]],
    CachedFunction[P, T] | CachedAsyncFunction[P, T],
]:
    """Decorator that caches function results with a time-to-live.

    Args:
        ttl (float): Time in seconds before cache expires (default: 300).
        key (Callable | None): Optional function to generate cache key from args/kwargs.
            Should accept the same arguments as the decorated function and return a
            hashable key.
        per_instance (bool): When True (default), each class instance gets its own
            independent cache.

    Returns:
        Decorator: Decorated callable with TTL caching.

    Example:
        @ttl_cache(ttl=60)
        def expensive_function(x):
            return x ** 2

        @ttl_cache(ttl=120, key=lambda x, y: (x, y))
        async def async_expensive_function(x, y):
            await asyncio.sleep(1)
            return x + y
    """

    def _get_cache_key(args: tuple, kwargs: dict) -> Any:
        if key is not None:
            try:
                return key(*args, **kwargs)
            except Exception:
                return None
        return _make_key(args, kwargs, strict=False)

    is_async_func: bool | None = None  # resolved on first decoration

    def _build_wrapper(func: Callable) -> Any:
        """Build a fresh wrapper with its own cache around *func*."""
        cache = CachetoolsTTLCache(maxsize=_UNBOUNDED_MAXSIZE, ttl=ttl)
        lock = threading.RLock()

        def make_info(hits: int, misses: int) -> CacheInfo:
            return CacheInfo(
                hits=hits,
                misses=misses,
                maxsize=None,
                currsize=len(cache),
                ttl=ttl,
            )

        nonlocal is_async_func
        if is_async_func is None:
            is_async_func = inspect.iscoroutinefunction(func)

        if is_async_func:
            return _make_async_wrapper(func, cache, lock, _get_cache_key, make_info)
        return _make_sync_wrapper(func, cache, lock, _get_cache_key, make_info)

    def decorator(
        func: Callable[P, T] | Callable[P, Awaitable[T]],
    ) -> CachedFunction[P, T] | CachedAsyncFunction[P, T]:
        nonlocal is_async_func
        is_async_func = inspect.iscoroutinefunction(func)

        wrapper = _build_wrapper(func)

        if per_instance:
            return cast(
                CachedFunction[P, T] | CachedAsyncFunction[P, T],
                _CacheDescriptor(factory=_build_wrapper, func=func, wrapper=wrapper),
            )
        return cast(CachedFunction[P, T] | CachedAsyncFunction[P, T], wrapper)

    return decorator


@overload
def file_cache(
    cache_dir: str | Path | None = None,
    ttl: float | None = None,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = False,
) -> Callable[[Callable[P, T]], CachedFunction[P, T]]: ...


@overload
def file_cache(
    cache_dir: str | Path | None = None,
    ttl: float | None = None,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = False,
) -> Callable[[Callable[P, Awaitable[T]]], CachedAsyncFunction[P, T]]: ...


def file_cache(
    cache_dir: str | Path | None = None,
    ttl: float | None = None,
    *,
    key: Callable[..., Any] | None = None,
    per_instance: bool = False,
) -> Callable[
    [Callable[P, T] | Callable[P, Awaitable[T]]],
    CachedFunction[P, T] | CachedAsyncFunction[P, T],
]:
    """Decorator that caches function results to disk using pickle.

    Args:
        cache_dir (str | Path): Directory to store cache files (default: ".cache").
        ttl (float | None): Optional time-to-live in seconds (None = no expiration).
        key (Callable | None): Optional function to generate cache key from args/kwargs.
            Should accept the same arguments as the decorated function and return a
            hashable key.
        per_instance (bool): When True, each class instance gets its own independent
            cache.  Defaults to False because disk-backed caches are shared resources
            and each instance would open its own SQLite connection.

    Returns:
        Decorator: Decorated callable with file caching.

    Example:
        @file_cache(cache_dir="./my_cache", ttl=3600)
        def process_large_dataset(data_path):
            # Expensive computation
            return result

        @file_cache(ttl=600, key=lambda endpoint, **kwargs: endpoint)
        async def fetch_api_data(endpoint, **kwargs):
            # API call - cache only based on endpoint, ignore other params
            return data

        # Cache based on specific parameters only
        @file_cache(cache_dir="./cache", key=lambda x, y, z=None: (x, y))
        def compute(x, y, z=None):
            # z is not part of the cache key
            return x + y
    """
    resolved_cache_dir = (
        get_default_cache_dir() if cache_dir is None else Path(cache_dir)
    )

    def _get_cache_key(args: tuple, kwargs: dict) -> Any:
        if key is not None:
            try:
                return key(*args, **kwargs)
            except Exception:
                return None
        return _make_key(args, kwargs, strict=False)

    def _get_store_key(cache_key: Any) -> str:
        """Generate a stable string key for disk cache lookup."""
        try:
            key_str = str(cache_key)
        except Exception:
            key_str = repr(cache_key)
        return hashlib.md5(key_str.encode()).hexdigest()

    is_async_func: bool | None = None

    def _build_wrapper(func: Callable) -> Any:
        """Build a fresh wrapper with its own DiskCache around *func*."""
        func_name = str(getattr(func, "__name__", "unknown_function"))
        func_cache_dir = resolved_cache_dir / func_name
        func_cache_dir.mkdir(parents=True, exist_ok=True)
        disk_cache = DiskCache(str(func_cache_dir))
        _register_disk_cache(disk_cache)
        lock = threading.RLock()

        def make_info(hits: int, misses: int) -> CacheInfo:
            return CacheInfo(
                hits=hits,
                misses=misses,
                maxsize=None,
                currsize=len(disk_cache),
                ttl=ttl,
            )

        nonlocal is_async_func
        if is_async_func is None:
            is_async_func = inspect.iscoroutinefunction(func)

        if is_async_func:
            inner_hits = 0
            inner_misses = 0
            in_flight = _InFlightCoalescer()

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                nonlocal inner_hits, inner_misses
                cache_key = _get_cache_key(args, kwargs)
                if cache_key is None:
                    return await func(*args, **kwargs)

                store_key = _get_store_key(cache_key)

                with lock:
                    cached = disk_cache.get(store_key, default=_MISSING, retry=True)
                    if cached is not _MISSING:
                        inner_hits += 1
                        return cached

                    should_compute, future = in_flight.start(store_key)
                    if should_compute:
                        inner_misses += 1

                if not should_compute:
                    return await asyncio.shield(future)

                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    with lock:
                        in_flight.reject(store_key, exc)
                    raise

                with lock:
                    current = disk_cache.get(store_key, default=_MISSING, retry=True)
                    if current is not _MISSING:
                        final = current
                    else:
                        final = result
                        with contextlib.suppress(Exception):
                            disk_cache.set(
                                store_key,
                                result,
                                expire=ttl,
                                retry=True,
                            )
                    in_flight.resolve(store_key, final)

                return final

            def cache_clear() -> None:
                nonlocal inner_hits, inner_misses
                with lock:
                    disk_cache.clear()
                    inner_hits = 0
                    inner_misses = 0

            def cache_info() -> CacheInfo:
                with lock:
                    return make_info(inner_hits, inner_misses)

            def close_cache() -> None:
                with _DISK_CACHES_LOCK:
                    _DISK_CACHES.discard(disk_cache)
                _close_disk_cache(disk_cache)

            async_wrapper.cache_clear = cache_clear  # ty:ignore[unresolved-attribute]
            async_wrapper.cache_info = cache_info  # ty:ignore[unresolved-attribute]
            weakref.finalize(async_wrapper, close_cache)
            return async_wrapper

        # Sync path
        inner_hits = 0
        inner_misses = 0

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal inner_hits, inner_misses
            cache_key = _get_cache_key(args, kwargs)
            if cache_key is None:
                return func(*args, **kwargs)

            store_key = _get_store_key(cache_key)

            with lock:
                cached = disk_cache.get(store_key, default=_MISSING, retry=True)
            if cached is not _MISSING:
                with lock:
                    inner_hits += 1
                return cached

            with lock:
                inner_misses += 1
            result = func(*args, **kwargs)

            with lock, contextlib.suppress(Exception):
                disk_cache.set(store_key, result, expire=ttl, retry=True)
            return result

        def cache_clear() -> None:
            nonlocal inner_hits, inner_misses
            with lock:
                disk_cache.clear()
                inner_hits = 0
                inner_misses = 0

        def cache_info() -> CacheInfo:
            with lock:
                return make_info(inner_hits, inner_misses)

        def close_cache() -> None:
            with _DISK_CACHES_LOCK:
                _DISK_CACHES.discard(disk_cache)
            _close_disk_cache(disk_cache)

        sync_wrapper.cache_clear = cache_clear  # ty:ignore[unresolved-attribute]
        sync_wrapper.cache_info = cache_info  # ty:ignore[unresolved-attribute]
        weakref.finalize(sync_wrapper, close_cache)
        return sync_wrapper

    def decorator(
        func: Callable[P, T] | Callable[P, Awaitable[T]],
    ) -> CachedFunction[P, T] | CachedAsyncFunction[P, T]:
        nonlocal is_async_func
        is_async_func = inspect.iscoroutinefunction(func)

        wrapper = _build_wrapper(func)

        if per_instance:
            return cast(
                CachedFunction[P, T] | CachedAsyncFunction[P, T],
                _CacheDescriptor(factory=_build_wrapper, func=func, wrapper=wrapper),
            )
        return cast(CachedFunction[P, T] | CachedAsyncFunction[P, T], wrapper)

    return decorator


@overload
def cache[**P, T](
    func: Callable[P, T],
) -> CachedFunction[P, T]: ...


@overload
def cache[**P, T](
    func: Callable[P, Awaitable[T]],
) -> CachedAsyncFunction[P, T]: ...


def cache[**P, T](
    func: Callable[P, T] | Callable[P, Awaitable[T]],
) -> CachedFunction[P, T] | CachedAsyncFunction[P, T]:
    """Generic cache decorator that applies an LRU cache with cache size of 1.

    Args:
        func (Callable): Function to be cached.

    Returns:
        Decorator: Decorated function with LRU caching.

    Example:
        @cache
        def compute_square(x):
            return x * x

        @cache
        async def fetch_data(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
    """
    return cast(
        CachedFunction[P, T] | CachedAsyncFunction[P, T], lru_cache(maxsize=1)(func)
    )
