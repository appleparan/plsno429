"""Throttling algorithm implementations."""

from __future__ import annotations

from typing import Any

from .base import BaseThrottleAlgorithm
from .exceptions import ConfigurationError
from .utils import add_jitter, is_rate_limit_error, parse_retry_after


class RetryAlgorithm(BaseThrottleAlgorithm):
    """Basic retry algorithm with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        **kwargs: Any
    ) -> None:
        """Initialize retry algorithm.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            backoff_multiplier: Exponential backoff multiplier
            **kwargs: Base class parameters
        """
        super().__init__(**kwargs)

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier

        # Track retry state
        self._retry_count = 0

        self._validate_retry_config()

    def _validate_retry_config(self) -> None:
        """Validate retry-specific configuration."""
        if self.max_retries < 0:
            raise ConfigurationError('max_retries must be non-negative')

        if self.base_delay <= 0:
            raise ConfigurationError('base_delay must be positive')

        if self.max_delay <= 0:
            raise ConfigurationError('max_delay must be positive')

        if self.backoff_multiplier <= 0:
            raise ConfigurationError('backoff_multiplier must be positive')

    def should_throttle(self, estimated_tokens: int = 0, **kwargs: Any) -> float | None:
        """Check if request should be throttled due to TPM limits.
        
        Args:
            estimated_tokens: Estimated tokens for upcoming request
            **kwargs: Additional parameters
            
        Returns:
            Delay in seconds if throttling needed, None otherwise
        """
        # Check TPM limit
        tpm_delay = self._check_tpm_limit(estimated_tokens)
        if tpm_delay is not None:
            return self._enforce_max_wait(tpm_delay)

        return None

    def on_request_success(self, tokens_used: int = 0, **kwargs: Any) -> None:
        """Handle successful request.
        
        Args:
            tokens_used: Number of tokens used in request
            **kwargs: Additional parameters
        """
        # Reset retry counter on success
        self._retry_count = 0

        # Track token usage
        if tokens_used > 0:
            self._add_token_usage(tokens_used)

    def on_request_failure(
        self,
        exception: Exception,
        estimated_tokens: int = 0,
        **kwargs: Any
    ) -> float | None:
        """Handle failed request and determine retry delay.
        
        Args:
            exception: Exception that occurred
            estimated_tokens: Estimated tokens for the request
            **kwargs: Additional parameters
            
        Returns:
            Delay in seconds before retry, None if no retry
        """
        # Only retry on rate limit errors
        if not is_rate_limit_error(exception):
            return None

        # Check if we've exceeded max retries
        if self._retry_count >= self.max_retries:
            return None

        self._retry_count += 1

        # Try to get retry delay from Retry-After header
        retry_after_delay = None
        if hasattr(exception, 'response'):
            retry_after_delay = parse_retry_after(exception.response)

        if retry_after_delay is not None:
            # Use server-provided delay
            delay = retry_after_delay
        else:
            # Use exponential backoff
            delay = min(
                self.base_delay * (self.backoff_multiplier ** (self._retry_count - 1)),
                self.max_delay
            )

        # Add jitter to distribute requests
        delay = add_jitter(delay, self.jitter)

        # Enforce maximum wait time
        return self._enforce_max_wait(delay)

    def reset_retry_count(self) -> None:
        """Reset retry counter."""
        self._retry_count = 0
