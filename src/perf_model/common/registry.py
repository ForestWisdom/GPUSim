"""Simple name-to-object registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(value: T) -> T:
            if name in self._items:
                raise KeyError(f"{name} already registered")
            self._items[name] = value
            return value

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown registry key: {name}") from exc

    def keys(self) -> list[str]:
        return sorted(self._items.keys())
