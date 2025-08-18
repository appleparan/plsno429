"""Base throttling algorithm interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from .exceptions import RateLimitExceeded
from .utils import calculate_wait_until_next_minute


class BaseThrottleAlgorithm(ABC):
    """Abstract base class for throttling algorithms."""

    def __init__(
        self,
        tpm_limit: int = 90000,
        safety_margin: float = 0.9,
        max_wait_minutes: float = 5.0,
        jitter: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize base throttling algorithm.
        
        Args:
            tpm_limit: Tokens per minute limit
            safety_margin: Safety margin (0.0-1.0) to stop before limit
            max_wait_minutes: Maximum time to wait in minutes
            jitter: Whether to add jitter to delays
            **kwargs: Algorithm-specific parameters
        """
        self.tpm_limit = tpm_limit
        self.safety_margin = safety_margin
        self.max_wait_minutes = max_wait_minutes
        self.jitter = jitter

        # TPM tracking
        self._token_usage: dict[int, int] = {}  # minute -> token count
        self._last_cleanup = time.time()

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate algorithm configuration."""
        if self.tpm_limit <= 0:
            from .exceptions import ConfigurationError
            raise ConfigurationError('tpm_limit must be positive')

        if not 0 <= self.safety_margin <= 1:
            from .exceptions import ConfigurationError
            raise ConfigurationError('safety_margin must be between 0 and 1')

        if self.max_wait_minutes <= 0:
            from .exceptions import ConfigurationError
            raise ConfigurationError('max_wait_minutes must be positive')

    def _cleanup_old_token_usage(self) -> None:
        """Remove token usage data older than 2 minutes."""
        current_time = time.time()
        if current_time - self._last_cleanup < 30:  # Cleanup every 30 seconds
            return

        current_minute = int(current_time // 60)
        cutoff_minute = current_minute - 2

        keys_to_remove = [minute for minute in self._token_usage if minute < cutoff_minute]
        for key in keys_to_remove:
            del self._token_usage[key]

        self._last_cleanup = current_time

    def _get_current_tpm_usage(self) -> int:
        """Get current tokens per minute usage."""
        self._cleanup_old_token_usage()
        current_minute = int(time.time() // 60)
        return self._token_usage.get(current_minute, 0)

    def _add_token_usage(self, tokens: int) -> None:
        """Add token usage to current minute."""
        current_minute = int(time.time() // 60)
        self._token_usage[current_minute] = self._token_usage.get(current_minute, 0) + tokens

    def _check_tpm_limit(self, estimated_tokens: int = 0) -> float | None:
        """Check if adding tokens would exceed TPM limit.
        
        Args:
            estimated_tokens: Estimated tokens for upcoming request
            
        Returns:
            Seconds to wait until next minute if limit would be exceeded, None otherwise
        """
        effective_limit = int(self.tpm_limit * self.safety_margin)
        current_usage = self._get_current_tpm_usage()

        if current_usage + estimated_tokens > effective_limit:
            return calculate_wait_until_next_minute()

        return None

    @abstractmethod
    def should_throttle(self, **kwargs: Any) -> float | None:
        """Check if request should be throttled.
        
        Args:
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Delay in seconds if throttling needed, None otherwise
        """

    @abstractmethod
    def on_request_success(self, **kwargs: Any) -> None:
        """Handle successful request.
        
        Args:
            **kwargs: Algorithm-specific parameters
        """

    @abstractmethod
    def on_request_failure(self, exception: Exception, **kwargs: Any) -> float | None:
        """Handle failed request.
        
        Args:
            exception: Exception that occurred
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Delay in seconds before retry, None if no retry
        """

    def _enforce_max_wait(self, delay: float) -> float:
        """Enforce maximum wait time.
        
        Args:
            delay: Requested delay in seconds
            
        Returns:
            Capped delay
            
        Raises:
            RateLimitExceeded: If delay exceeds maximum wait time
        """
        max_wait_seconds = self.max_wait_minutes * 60

        if delay > max_wait_seconds:
            raise RateLimitExceeded(
                f'Rate limit delay ({delay:.1f}s) exceeds maximum wait time '
                f'({max_wait_seconds:.1f}s)'
            )

        return delay
