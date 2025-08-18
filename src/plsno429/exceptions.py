"""Exception classes for plsno429."""


class ThrottleError(Exception):
    """Base exception for throttling errors."""


class RateLimitExceeded(ThrottleError):
    """Raised when rate limit is exceeded and retries are exhausted."""


class ConfigurationError(ThrottleError):
    """Raised when throttling configuration is invalid."""


class CircuitBreakerOpen(ThrottleError):
    """Raised when circuit breaker is open and blocking requests."""
