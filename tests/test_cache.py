"""Tests for caching utilities."""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from anibridge.utils.cache import (
    LRUDict,
    TTLDict,
    _close_all_disk_caches,
    _generic_hash,
    _make_key,
    _register_disk_cache,
    cache,
    file_cache,
    get_default_cache_dir,
    lru_cache,
    set_default_cache_dir,
    ttl_cache,
)


def test_generic_hash_order_insensitive_for_dicts():
    """Test that _generic_hash produces the same hash for dicts in different orders."""
    data_one = {"b": [1, 2], "a": {"x": 1}}
    data_two = {"a": {"x": 1}, "b": [1, 2]}

    assert _generic_hash(data_one) == _generic_hash(data_two)


def test_generic_hash_handles_cycles():
    """Cyclic data structures are rejected as cache keys."""
    cyclic = []
    cyclic.append(cyclic)

    with pytest.raises(TypeError):
        _generic_hash(cyclic)


def test_lru_cache_caches_unhashable_arguments():
    """Test that lru_cache caches results for unhashable arguments."""
    call_count = 0

    @lru_cache(maxsize=8)
    def compute(values):
        nonlocal call_count
        call_count += 1
        return sum(values)

    assert compute([1, 2, 3]) == 6
    assert compute([1, 2, 3]) == 6
    assert call_count == 1


def test_ttl_cache_caches_unhashable_arguments():
    """Test that ttl_cache caches results for unhashable arguments."""
    call_count = 0

    @ttl_cache(ttl=60)
    def compute(values):
        nonlocal call_count
        call_count += 1
        return sum(values)

    assert compute([4, 5]) == 9
    assert compute([4, 5]) == 9
    assert call_count == 1


def test_file_cache_sync_caches(tmp_path) -> None:
    """Synchronous file_cache should store and reuse results."""
    calls = {"count": 0}

    @file_cache(cache_dir=tmp_path)
    def add(x: int, y: int) -> int:
        calls["count"] += 1
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls["count"] == 1

    add.cache_clear()


def test_file_cache_sync_unpickleable_result(tmp_path) -> None:
    """Unpickleable results should not be cached."""
    calls = {"count": 0}

    @file_cache(cache_dir=tmp_path)
    def make_callable(x: int):
        calls["count"] += 1
        return lambda: x

    result = cast(Callable[[], int], make_callable(1))
    assert result() == 1
    result = cast(Callable[[], int], make_callable(1))
    assert result() == 1
    assert calls["count"] == 2


def test_file_cache_custom_key_error(tmp_path) -> None:
    """Key errors should skip caching."""
    calls = {"count": 0}

    def _bad_key(*_args, **_kwargs):
        raise ValueError("boom")

    @file_cache(cache_dir=tmp_path, key=_bad_key)
    def calc(x: int) -> int:
        calls["count"] += 1
        return x

    assert calc(5) == 5
    assert calc(5) == 5
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_file_cache_async_caches(tmp_path) -> None:
    """Async file_cache should store and reuse results."""
    calls = {"count": 0}

    @file_cache(cache_dir=tmp_path)
    async def fetch(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0)
        return x

    assert await fetch(1) == 1
    assert await fetch(1) == 1
    assert calls["count"] == 1

    fetch.cache_clear()


@pytest.mark.asyncio
async def test_lru_cache_async_single_flight() -> None:
    """Concurrent async calls with the same key should compute once."""
    calls = {"count": 0}

    @lru_cache(maxsize=16)
    async def compute(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0.01)
        return x * 2

    results = await asyncio.gather(
        compute(7),
        compute(7),
        compute(7),
    )
    assert results == [14, 14, 14]
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_ttl_cache_async_single_flight() -> None:
    """Concurrent async calls with the same key should compute once."""
    calls = {"count": 0}

    @ttl_cache(ttl=60)
    async def compute(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0.01)
        return x + 1

    results = await asyncio.gather(
        compute(9),
        compute(9),
        compute(9),
    )
    assert results == [10, 10, 10]
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_lru_cache_async_single_flight_exception_released() -> None:
    """Concurrent awaiters should all receive exceptions and allow retries."""
    calls = {"count": 0}
    should_fail = {"value": True}

    @lru_cache(maxsize=16)
    async def compute(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0.01)
        if should_fail["value"]:
            raise ValueError("boom")
        return x * 3

    first, second = await asyncio.gather(compute(5), compute(5), return_exceptions=True)
    assert isinstance(first, ValueError)
    assert isinstance(second, ValueError)
    assert calls["count"] == 1

    should_fail["value"] = False
    assert await compute(5) == 15
    assert calls["count"] == 2


def test_lru_cache_method_uses_per_instance_cache_by_default() -> None:
    """Method decorators should isolate caches per instance by default."""
    calls = {"count": 0}

    class Worker:
        @lru_cache(maxsize=16, key=lambda x: x)
        def compute(self, x: int) -> int:
            calls["count"] += 1
            return x + 1

    one = Worker()
    two = Worker()

    assert one.compute(7) == 8
    assert one.compute(7) == 8
    assert two.compute(7) == 8
    assert two.compute(7) == 8
    assert calls["count"] == 2


def test_lru_cache_method_can_share_cache_when_disabled() -> None:
    """Setting per_instance=False should share one method cache across instances."""
    calls = {"count": 0}

    class Worker:
        @lru_cache(maxsize=16, key=lambda self, x: x, per_instance=False)
        def compute(self, x: int) -> int:
            calls["count"] += 1
            return x + 1

    one = Worker()
    two = Worker()

    assert one.compute(4) == 5
    assert one.compute(4) == 5
    assert two.compute(4) == 5
    assert two.compute(4) == 5
    assert calls["count"] == 1


def test_ttl_cache_method_uses_per_instance_cache_by_default() -> None:
    """TTL method caches should also be isolated per instance by default."""
    calls = {"count": 0}

    class Worker:
        @ttl_cache(ttl=60, key=lambda x: x)
        def compute(self, x: int) -> int:
            calls["count"] += 1
            return x + 2

    one = Worker()
    two = Worker()

    assert one.compute(2) == 4
    assert one.compute(2) == 4
    assert two.compute(2) == 4
    assert two.compute(2) == 4
    assert calls["count"] == 2


def test_ttl_cache_per_instance_disabled_returns_plain_wrapper() -> None:
    """TTL decorator should return plain wrapper when per_instance is disabled."""

    @ttl_cache(ttl=60, per_instance=False)
    def compute(x: int) -> int:
        return x + 2

    assert compute(1) == 3
    assert compute(1) == 3

    info = compute.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_file_cache_method_is_shared_by_default(tmp_path) -> None:
    """file_cache defaults to shared method caching across instances."""
    calls = {"count": 0}

    class Worker:
        @file_cache(cache_dir=tmp_path, key=lambda self, x: x)
        def compute(self, x: int) -> int:
            calls["count"] += 1
            return x * 2

    one = Worker()
    two = Worker()

    assert one.compute(3) == 6
    assert two.compute(3) == 6
    assert calls["count"] == 1


def test_file_cache_method_per_instance_wrappers_share_disk_entries(tmp_path) -> None:
    """per_instance wrappers should still read shared disk entries for same key."""
    calls = {"count": 0}

    class Worker:
        @file_cache(cache_dir=tmp_path, key=lambda x: x, per_instance=True)
        def compute(self, x: int) -> int:
            calls["count"] += 1
            return x * 2

    one = Worker()
    two = Worker()

    assert one.compute(3) == 6
    assert one.compute(3) == 6
    assert two.compute(3) == 6
    assert two.compute(3) == 6
    assert calls["count"] == 1

    one_info = one.compute.cache_info()
    two_info = two.compute.cache_info()

    assert one_info.misses == 1
    assert one_info.hits == 1
    assert two_info.misses == 0
    assert two_info.hits == 2


def test_per_instance_cache_rejects_slots_without_dict() -> None:
    """per_instance decorators should fail for __slots__ classes without __dict__."""
    with pytest.raises(TypeError, match="Cannot use per-instance caching"):

        class Worker:
            __slots__ = ("value",)

            @lru_cache(maxsize=8)
            def compute(self, x: int) -> int:
                return x


def test_ttl_dict_expires_items() -> None:
    """TTLDict should expire entries after TTL."""
    cache = TTLDict[str, int](ttl=0.05)
    cache["answer"] = 42

    assert cache["answer"] == 42
    time.sleep(0.08)

    with pytest.raises(KeyError):
        _ = cache["answer"]


def test_lru_dict_evicts_oldest_item() -> None:
    """LRUDict should evict least-recently-used entries when full."""
    cache = LRUDict[str, int](maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    _ = cache["a"]
    cache["c"] = 3

    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache


def test_cache_dict_cache_info_tracks_hits_and_misses() -> None:
    """Cache-backed dicts should report hit/miss statistics."""
    cache = TTLDict[str, int](ttl=10)
    cache["x"] = 7

    assert cache.get("x") == 7
    assert cache.get("missing") is None

    info = cache.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.ttl == 10


def test_default_cache_dir_roundtrip() -> None:
    """Default cache dir setter/getter should roundtrip values."""
    original = get_default_cache_dir()
    try:
        set_default_cache_dir("custom-cache-dir")
        assert get_default_cache_dir() == Path("custom-cache-dir")

        set_default_cache_dir(None)
        assert get_default_cache_dir() == Path(".cache")
    finally:
        set_default_cache_dir(original)


def test_close_all_disk_caches_closes_registered_entries() -> None:
    """Registered disk-like caches should be closed and registry cleared."""

    class DummyCache:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    first = DummyCache()
    second = DummyCache()

    _register_disk_cache(cast(Any, first))
    _register_disk_cache(cast(Any, second))
    _close_all_disk_caches()

    assert first.closed == 1
    assert second.closed == 1

    # Calling again should be a no-op because registry was cleared.
    _close_all_disk_caches()
    assert first.closed == 1
    assert second.closed == 1


def test_cache_dict_mutation_helpers_are_covered() -> None:
    """CacheDict mapping helpers should behave like normal mutable mappings."""
    cache = LRUDict[str, int](maxsize=4)
    cache["a"] = 1
    cache["b"] = 2

    assert len(cache) == 2
    assert tuple(iter(cache)) == ("a", "b")
    assert cache.keys() == ("a", "b")
    assert cache.values() == (1, 2)
    assert cache.items() == (("a", 1), ("b", 2))

    assert cache.pop("a") == 1
    assert cache.pop("missing", 99) == 99

    del cache["b"]
    cache["x"] = 3
    cache.clear()
    assert len(cache) == 0


def test_generic_hash_supports_sets() -> None:
    """Structural hashing should include set payloads."""
    value = {"a": {1, 2, 3}, "b": [4, 5]}
    assert isinstance(_generic_hash(value), int)


def test_make_key_handles_strict_and_non_strict_failures() -> None:
    """_make_key should return None when inputs cannot be represented safely."""
    strict_key = _make_key(args=(1, "x"), kwargs={"a": 2}, strict=True)
    assert strict_key == ((1, "x"), (("a", 2),))

    assert _make_key(args=([1, 2],), kwargs={}, strict=True) is None

    class Unfreezable:
        __hash__ = None

    assert _make_key(args=(Unfreezable(),), kwargs={}, strict=False) is None


def test_lru_descriptor_class_access_and_class_level_controls() -> None:
    """Class-level access should expose descriptor control methods."""

    class Worker:
        @lru_cache(maxsize=8, key=lambda self, x: x, per_instance=True)
        def compute(self, x: int) -> int:
            return x + 10

    worker = Worker()

    assert Worker.compute(worker, 3) == 13
    assert worker.compute(3) == 13

    class_info = Worker.compute.cache_info()
    assert class_info.maxsize == 8

    Worker.compute.cache_clear()
    class_info_after = Worker.compute.cache_info()
    assert class_info_after.hits == 0
    assert class_info_after.misses == 0


def test_lru_cache_key_error_disables_caching() -> None:
    """If key generation fails, lru_cache should bypass cache."""
    calls = {"count": 0}

    @lru_cache(maxsize=8, key=lambda _x: (_ for _ in ()).throw(ValueError("boom")))
    def compute(x: int) -> int:
        calls["count"] += 1
        return x * 2

    assert compute(4) == 8
    assert compute(4) == 8
    assert calls["count"] == 2


def test_ttl_cache_key_error_disables_caching() -> None:
    """If key generation fails, ttl_cache should bypass cache."""
    calls = {"count": 0}

    @ttl_cache(ttl=30, key=lambda _x: (_ for _ in ()).throw(ValueError("boom")))
    def compute(x: int) -> int:
        calls["count"] += 1
        return x + 5

    assert compute(4) == 9
    assert compute(4) == 9
    assert calls["count"] == 2


def test_lru_sync_cache_info_and_clear_with_per_instance_disabled() -> None:
    """Sync wrapper cache_info/cache_clear should track/reset stats."""

    @lru_cache(maxsize=8, per_instance=False)
    def compute(x: int) -> int:
        return x * 2

    assert compute(2) == 4
    assert compute(2) == 4

    info = compute.cache_info()
    assert info.hits == 1
    assert info.misses == 1

    compute.cache_clear()
    cleared = compute.cache_info()
    assert cleared.hits == 0
    assert cleared.misses == 0


def test_lru_sync_race_returns_existing_cached_value() -> None:
    """Sync wrapper should return existing value if another caller stores first."""
    calls = {"count": 0}

    @lru_cache(maxsize=8, key=lambda x: x, per_instance=False)
    def compute(x: int) -> int:
        calls["count"] += 1
        # First caller is intentionally slower.
        if calls["count"] == 1:
            time.sleep(0.04)
            return 111
        time.sleep(0.005)
        return 222

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(compute, 9)
        second = pool.submit(compute, 9)
        result_one = first.result()
        result_two = second.result()

    # Both should observe the first stored value due second-stage existing check.
    assert result_one == 222
    assert result_two == 222
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_lru_async_cache_branches_for_none_hit_info_and_clear() -> None:
    """Async wrapper should cover key-none bypass, hit path, info and clear."""
    calls = {"count": 0}

    def _key(x: int) -> int:
        if x < 0:
            raise ValueError("skip")
        return x

    @lru_cache(maxsize=8, key=_key, per_instance=False)
    async def compute(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0)
        return x * 10

    assert await compute(-1) == -10
    assert await compute(-1) == -10
    assert await compute(3) == 30
    assert await compute(3) == 30

    info = compute.cache_info()
    assert info.hits == 1
    assert info.misses == 1

    compute.cache_clear()
    cleared = compute.cache_info()
    assert cleared.hits == 0
    assert cleared.misses == 0
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_file_cache_async_paths_for_waiters_exceptions_and_info(tmp_path) -> None:
    """Async file_cache should cover waiter, exception, and info branches."""
    calls = {"count": 0}
    fail_once = {"value": True}

    @file_cache(cache_dir=tmp_path, key=lambda x: x, per_instance=False)
    async def compute(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0.01)
        if x == 7 and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("fail")
        return x * 3

    first, second = await asyncio.gather(compute(7), compute(7), return_exceptions=True)
    assert isinstance(first, RuntimeError)
    assert isinstance(second, RuntimeError)

    assert await compute(7) == 21

    a, b, c = await asyncio.gather(compute(11), compute(11), compute(11))
    assert (a, b, c) == (33, 33, 33)

    info = compute.cache_info()
    assert info.hits == 0
    assert info.misses == 3
    assert info.currsize >= 2


@pytest.mark.asyncio
async def test_file_cache_async_key_error_bypasses_cache(tmp_path) -> None:
    """Async file_cache should bypass cache when key function raises."""
    calls = {"count": 0}

    def _bad_key(_x: int) -> int:
        raise ValueError("no key")

    @file_cache(cache_dir=tmp_path, key=_bad_key)
    async def compute(x: int) -> int:
        calls["count"] += 1
        await asyncio.sleep(0)
        return x

    assert await compute(1) == 1
    assert await compute(1) == 1
    assert calls["count"] == 2


def test_file_cache_store_key_falls_back_to_repr(tmp_path) -> None:
    """Store-key generation should fall back to repr when str fails."""
    calls = {"count": 0}

    class BadStr:
        def __str__(self) -> str:
            raise RuntimeError("bad str")

        def __repr__(self) -> str:
            return "bad-str-repr"

    @file_cache(cache_dir=tmp_path, key=lambda _x: BadStr(), per_instance=False)
    def compute(x: int) -> int:
        calls["count"] += 1
        return x + 1

    assert compute(2) == 3
    assert compute(2) == 3
    assert calls["count"] == 1


def test_cache_decorator_alias_behaves_like_single_entry_lru() -> None:
    """The generic cache decorator should cache only one recent entry."""
    calls = {"count": 0}

    @cache
    def compute(x: int) -> int:
        calls["count"] += 1
        return x * 4

    assert compute(1) == 4
    assert compute(1) == 4
    assert compute(2) == 8
    assert compute(1) == 4
    assert calls["count"] == 3
