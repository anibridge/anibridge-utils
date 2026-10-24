"""Shared types for AniBridge."""

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Protocol

__all__ = ["Comparable", "MappingDescriptor", "ProviderLogger"]

type MappingDescriptor = tuple[str, str, str | None]


type _ExcInfo = (
    bool
    | BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)


class ProviderLogger(Protocol):
    """Protocol for loggers injected into provider and app clients."""

    def debug(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Debug log."""
        ...

    def info(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Info log."""
        ...

    def warning(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Warning log."""
        ...

    def error(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Error log."""
        ...

    def exception(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfo = True,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Exception log."""
        ...

    def getChild(self, suffix: str) -> ProviderLogger:
        """Get a child logger with the given suffix."""
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
