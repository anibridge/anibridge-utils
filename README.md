# anibridge-utils

Shared utilities used across the [AniBridge](https://github.com/anibridge/anibridge) package ecosystem.

> [!IMPORTANT]
> This package is primarily an internal dependency for AniBridge packages, but it can also be useful when building related extensions.

- `anibridge.utils.cache`: Cache helpers shared by provider implementations.
- `anibridge.utils.limiter`: Async rate-limiting utilities.
- `anibridge.utils.registry`: Generic `ProviderRegistry` used to register provider classes by namespace.
- `anibridge.utils.types`: Shared type aliases such as `MappingDescriptor` and `ProviderLogger`.
