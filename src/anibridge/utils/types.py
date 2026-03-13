"""Shared types for AniBridge."""

from typing import Any, Protocol

__all__ = ["Comparable", "MappingDescriptor", "ProviderLogger"]

type MappingDescriptor = tuple[str, str, str | None]


class ProviderLogger(Protocol):
    """Protocol for loggers injected into provider and app clients."""

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        """Debug log."""
        ...

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        """Info log."""
        ...

    def success(self, msg: object, *args: object, **kwargs: object) -> None:
        """Success log."""
        ...

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        """Warning log."""
        ...

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        """Error log."""
        ...

    def exception(self, msg: object, *args: object, **kwargs: object) -> None:
        """Exception log."""
        ...
    
    def getChild(self, name: str) -> ProviderLogger:
        """Get a child logger with the given name."""
        ...


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
