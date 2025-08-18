"""Type definitions for plsno429."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Union

HTTPResponse = Any
SyncFunction = Callable[..., Any]
AsyncFunction = Callable[..., Awaitable[Any]]
ThrottledFunction = Union[SyncFunction, AsyncFunction]
ConfigDict = dict[str, Any]
