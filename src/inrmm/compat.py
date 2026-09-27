"""Small internal compatibility helpers used by the public package."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeAlias

PathLike: TypeAlias = str | Path
Number: TypeAlias = int | float


class LoggerMixin:
    """Provide a class-named standard-library logger."""

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__module__)


def init_fancy_logging() -> None:
    """Initialize concise logging without requiring an external helper package."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def convert(name: str, converter: Callable[[Any], Any]) -> Callable:
    """Convert one named argument before calling the decorated function."""

    def decorate(function: Callable) -> Callable:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            value = bound.arguments.get(name)
            if value is not None:
                bound.arguments[name] = converter(value)
            return function(*bound.args, **bound.kwargs)

        return wrapped

    return decorate


def timing() -> Callable:
    """Preserve the historical decorator API without forcing timing output."""

    def decorate(function: Callable) -> Callable:
        return function

    return decorate


def concat_dicts(dicts: Iterable[dict], extend_lists: bool = False) -> dict:
    """Collect equally named dictionary values into lists."""
    output: dict = {}
    for item in dicts:
        for key, value in item.items():
            if key not in output:
                output[key] = []
            if extend_lists and isinstance(value, list):
                output[key].extend(value)
            else:
                output[key].append(value)
    return output
