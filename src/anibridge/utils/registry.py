"""Registration utilities for provider implementations."""

from collections.abc import Callable, Iterator
from typing import Any, cast, overload

__all__ = ["ProviderRegistry", "provider"]


class ProviderRegistry[P: object]:
    """Registry for provider implementations."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: dict[str, type[P]] = {}

    def clear(self) -> None:
        """Remove all provider registrations."""
        self._providers.clear()

    def create(
        self,
        namespace: str,
        *args: object,
        **kwargs: object,
    ) -> P:
        """Instantiate the provider registered under `namespace`.

        Args:
            namespace (str): The namespace identifier to create.
            *args (object): Positional arguments forwarded to the provider
                constructor.
            **kwargs (object): Keyword arguments forwarded to the provider
                constructor.

        Returns:
            P: An instance of the registered provider.
        """
        provider_cls = self.get(namespace)
        return provider_cls(*args, **kwargs)

    def get(self, namespace: str) -> type[P]:
        """Return the provider class registered under `namespace`.

        Args:
            namespace (str): The namespace identifier to look up.

        Returns:
            type[P]: The registered provider class.
        """
        try:
            return self._providers[namespace]
        except KeyError as exc:
            raise LookupError(
                f"No provider registered for namespace '{namespace}'."
            ) from exc

    def namespaces(self) -> tuple[str, ...]:
        """Return a tuple of registered namespace identifiers.

        Returns:
            tuple[str, ...]: The registered namespace identifiers.
        """
        return tuple(self._providers)

    @overload
    def register(
        self,
        provider_cls: type[P],
        *,
        namespace: str | None = None,
    ) -> type[P]: ...

    @overload
    def register(
        self,
        provider_cls: None = None,
        *,
        namespace: str | None = None,
    ) -> Callable[[type[P]], type[P]]: ...

    def register(
        self,
        provider_cls: type[P] | None = None,
        *,
        namespace: str | None = None,
    ) -> type[P] | Callable[[type[P]], type[P]]:
        """Register a provider class, optionally used as a decorator.

        Args:
            provider_cls (type[P] | None): The provider class to register. If `None`,
                the method acts as a decorator factory.
            namespace (str | None): Explicit namespace override. Defaults to the class'
                `NAMESPACE` attribute.

        Returns:
            type[P] | Callable[[type[P]], type[P]]: The registered provider class, or
                a decorator that registers the class.
        """

        def decorator(cls: type[P]) -> type[P]:
            """Register `cls` as a provider."""
            resolved_namespace = namespace or getattr(cls, "NAMESPACE", None)
            if not isinstance(resolved_namespace, str) or not resolved_namespace:
                raise ValueError(
                    "Providers must define a non-empty string `NAMESPACE` "
                    "attribute or pass `namespace=` when registering."
                )
            namespace_key = resolved_namespace

            existing = self._providers.get(namespace_key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"A provider is already registered for namespace '{namespace_key}'."
                )

            self._providers[namespace_key] = cls
            return cls

        if provider_cls is None:
            return decorator
        return decorator(provider_cls)

    def unregister(self, namespace: str) -> None:
        """Remove a provider registration if it exists.

        Args:
            namespace (str): The namespace identifier to unregister.
        """
        self._providers.pop(namespace, None)

    def __contains__(self, namespace: object) -> bool:
        """Check if a provider is registered under `namespace`."""
        return isinstance(namespace, str) and namespace in self._providers

    def __iter__(self) -> Iterator[tuple[str, type[P]]]:
        """Iterate over `(namespace, provider_class)` pairs."""
        return iter(self._providers.items())


@overload
def provider[P: object](
    cls: type[P],
    *,
    namespace: str | None = None,
    registry: ProviderRegistry[Any],
) -> type[P]: ...


@overload
def provider[P: object](
    cls: None = None,
    *,
    namespace: str | None = None,
    registry: ProviderRegistry[Any],
) -> Callable[[type[P]], type[P]]: ...


def provider[P: object](
    cls: type[P] | None = None,
    *,
    namespace: str | None = None,
    registry: ProviderRegistry[Any],
) -> type[P] | Callable[[type[P]], type[P]]:
    """Class decorator that registers provider implementations.

    This helper lets providers register themselves declaratively:

        ```
        from anibridge.utils.registry import provider


        @provider(namespace="some_provider")
        class SomeProvider:
            NAMESPACE = "some_provider"
            ...
        ```

    Args:
        cls (type[P] | None): The provider class to register. If `None`, the function
            acts as a decorator factory.
        namespace (str | None): Explicit namespace override. Defaults to the class'
            `NAMESPACE` attribute.
        registry (ProviderRegistry[Any] | None): Alternate registry to insert into.
            Defaults to the module-level one.

    Returns:
        type[P] | Callable[[type[P]], type[P]]: The registered provider class, or
            a decorator that registers the class.
    """
    return cast(
        type[P] | Callable[[type[P]], type[P]],
        registry.register(cls, namespace=namespace),
    )
