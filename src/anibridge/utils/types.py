"""Shared types for AniBridge."""

from logging import Logger

__all__ = ["MappingDescriptor", "ProviderLogger"]

type MappingDescriptor = tuple[str, str, str | None]
type ProviderLogger = Logger  # TODO: `success()` typing
