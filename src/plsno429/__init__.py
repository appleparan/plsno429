"""plsno429 package.

A tiny Python library that politely says pls no 429 by auto-handling OpenAI rate limits.
"""

from importlib.metadata import PackageNotFoundError, version

from plsno429.decorators import (
    throttle_httpx,
    throttle_httpx_async,
    throttle_openai,
    throttle_openai_async,
    throttle_requests,
)
from plsno429.exceptions import (
    CircuitBreakerOpen,
    ConfigurationError,
    RateLimitExceeded,
    ThrottleError,
)

__all__ = [
    'CircuitBreakerOpen',
    'ConfigurationError',
    'RateLimitExceeded',
    'ThrottleError',
    'throttle_httpx',
    'throttle_httpx_async',
    'throttle_openai',
    'throttle_openai_async',
    'throttle_requests',
]

__author__ = """Jongsu Liam Kim"""
__email__ = 'jongsukim8@gmail.com'

try:
    __version__ = version('plsno429')
except PackageNotFoundError:
    # fallback for development
    __version__ = 'unknown'
