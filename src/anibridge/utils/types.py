"""Shared types for AniBridge."""

from typing import Any, Protocol

__all__ = ["Comparable", "MappingDescriptor", "ProviderLogger"]

type MappingDescriptor = tuple[str, str, str | None]


class ProviderLogger(Protocol):
    """Protocol for loggers injected into provider and app clients."""

    def debug(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def info(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def success(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def warning(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def error(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def exception(self, msg: object, *args: object, **kwargs: object) -> None: ...



class Comparable(Protocol):
    """Protocol for objects that can be compared using <, >, <=, >= operators."""

    def __lt__(self, other: Any, /) -> bool:
        """Return True if this object is less than other."""
        ...

    def __gt__(self, other: Any, /) -> bool:
        """Return True if this object is greater than other."""
        ...

    def __le__(self, other: Any, /) -> bool:
        """Return True if this object is less than or equal to other."""
        ...

    def __ge__(self, other: Any, /) -> bool:
        """Return True if this object is greater than or equal to other."""
        ...
