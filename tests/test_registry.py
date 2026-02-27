"""Tests for provider registry utilities."""

import pytest

from anibridge.utils.registry import ProviderRegistry


class BaseProvider:
    """Simple base provider for registry typing in tests."""


class AlphaProvider(BaseProvider):
    """Provider with static namespace."""

    NAMESPACE = "alpha"

    def __init__(self, value: int, *, enabled: bool = False) -> None:
        self.value = value
        self.enabled = enabled


class BetaProvider(BaseProvider):
    """Provider with static namespace."""

    NAMESPACE = "beta"


def test_register_get_and_create_with_args() -> None:
    """Registry should register, retrieve, and instantiate providers."""
    registry = ProviderRegistry[BaseProvider]()

    registered = registry.register(AlphaProvider)
    assert registered is AlphaProvider

    assert registry.get("alpha") is AlphaProvider
    instance = registry.create("alpha", 7, enabled=True)
    assert isinstance(instance, AlphaProvider)
    assert instance.value == 7
    assert instance.enabled is True


def test_register_can_override_namespace() -> None:
    """Explicit namespace should override class NAMESPACE."""
    registry = ProviderRegistry[BaseProvider]()

    registry.register(AlphaProvider, namespace="custom")

    assert registry.get("custom") is AlphaProvider
    assert "alpha" not in registry


def test_register_rejects_missing_namespace() -> None:
    """Registration should fail if no namespace can be resolved."""
    registry = ProviderRegistry[BaseProvider]()

    class NoNamespaceProvider(BaseProvider):
        pass

    with pytest.raises(ValueError, match="must define a non-empty string `NAMESPACE`"):
        registry.register(NoNamespaceProvider)


def test_register_rejects_duplicate_namespace_for_different_class() -> None:
    """Different providers cannot share the same namespace."""
    registry = ProviderRegistry[BaseProvider]()

    class FirstProvider(BaseProvider):
        NAMESPACE = "dup"

    class SecondProvider(BaseProvider):
        NAMESPACE = "dup"

    registry.register(FirstProvider)

    with pytest.raises(ValueError, match="already registered for namespace 'dup'"):
        registry.register(SecondProvider)


def test_register_same_class_twice_is_allowed() -> None:
    """Registering the same class for the same namespace should be idempotent."""
    registry = ProviderRegistry[BaseProvider]()

    registry.register(BetaProvider)
    registry.register(BetaProvider)

    assert registry.get("beta") is BetaProvider


def test_unregister_clear_contains_and_namespaces() -> None:
    """Registry should expose membership and cleanup helpers."""
    registry = ProviderRegistry[BaseProvider]()
    registry.register(AlphaProvider)
    registry.register(BetaProvider)

    assert "alpha" in registry
    assert "beta" in registry
    assert registry.namespaces() == ("alpha", "beta")

    registry.unregister("alpha")
    assert "alpha" not in registry

    registry.clear()
    assert registry.namespaces() == ()


def test_get_raises_lookup_error_for_unknown_namespace() -> None:
    """Lookup should raise LookupError for missing namespace."""
    registry = ProviderRegistry[BaseProvider]()

    with pytest.raises(
        LookupError, match="No provider registered for namespace 'missing'"
    ):
        registry.get("missing")


def test_iter_returns_namespace_and_provider_pairs() -> None:
    """Iteration should yield the registered mapping items."""
    registry = ProviderRegistry[BaseProvider]()
    registry.register(AlphaProvider)
    registry.register(BetaProvider)

    assert list(registry) == [
        ("alpha", AlphaProvider),
        ("beta", BetaProvider),
    ]
