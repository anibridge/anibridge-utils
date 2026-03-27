# anibridge-utils

Shared utilities used across the [AniBridge](https://github.com/anibridge/anibridge) package ecosystem.

> [!IMPORTANT]
> This package is primarily an internal dependency for AniBridge packages, but it can also be useful when building related extensions.

- `anibridge.utils.cache`: Cache helpers shared by provider implementations.
- `anibridge.utils.datetime`: Timezone and datetime normalization helpers.
- `anibridge.utils.image`: Helpers for provider image fetching and data URL encoding.
- `anibridge.utils.limiter`: Async rate-limiting utilities.
- `anibridge.utils.mappings`: Parsers and helpers for the AniBridge mappings schema.
- `anibridge.utils.registry`: Generic `ProviderRegistry` used to register provider classes by namespace.
- `anibridge.utils.tasks`: Safe background task scheduling helpers.
- `anibridge.utils.types`: Shared type aliases such as `MappingDescriptor` and `ProviderLogger`.
