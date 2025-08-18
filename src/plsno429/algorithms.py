"""Throttling algorithm implementations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .base import BaseThrottleAlgorithm
from .exceptions import ConfigurationError
from .utils import add_jitter, estimate_tokens, is_rate_limit_error, parse_retry_after


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


class TokenBucketAlgorithm(BaseThrottleAlgorithm):
    """Token bucket algorithm for rate limiting."""

    def __init__(
        self,
        burst_size: int = 1000,
        refill_rate: float = 1500.0,
        token_estimate_func: Callable[..., int] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize token bucket algorithm.

        Args:
            burst_size: Maximum number of tokens in bucket
            refill_rate: Tokens per second refill rate
            token_estimate_func: Function to estimate token usage
            **kwargs: Base class parameters
        """
        super().__init__(**kwargs)

        self.burst_size = burst_size
        self.refill_rate = refill_rate
        self.token_estimate_func = token_estimate_func or estimate_tokens

        # Token bucket state
        self._tokens = float(burst_size)
        self._last_refill = time.time()

        self._validate_token_bucket_config()

    def _validate_token_bucket_config(self) -> None:
        """Validate token bucket specific configuration."""
        if self.burst_size <= 0:
            msg = 'burst_size must be positive'
            raise ConfigurationError(msg)

        if self.refill_rate <= 0:
            msg = 'refill_rate must be positive'
            raise ConfigurationError(msg)

    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed."""
        current_time = time.time()
        time_elapsed = current_time - self._last_refill

        # Add tokens based on refill rate
        tokens_to_add = time_elapsed * self.refill_rate
        self._tokens = min(self.burst_size, self._tokens + tokens_to_add)
        self._last_refill = current_time

    def _consume_tokens(self, tokens: int) -> bool:
        """Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill_tokens()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        return False

    def _calculate_wait_time(self, tokens: int) -> float:
        """Calculate time to wait for tokens to be available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Time to wait in seconds
        """
        self._refill_tokens()

        if self._tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self._tokens
        wait_time = tokens_needed / self.refill_rate

        return wait_time

    def should_throttle(self, *args: Any, **kwargs: Any) -> float | None:
        """Check if request should be throttled due to token bucket or TPM limits.

        Args:
            *args: Arguments passed to token estimation function
            **kwargs: Keyword arguments including estimated_tokens

        Returns:
            Delay in seconds if throttling needed, None otherwise
        """
        # Get estimated tokens from kwargs or estimate them
        estimated_tokens = kwargs.get('estimated_tokens', 0)
        if estimated_tokens == 0 and self.token_estimate_func:
            try:
                estimated_tokens = self.token_estimate_func(*args, **kwargs)
            except Exception:
                estimated_tokens = 100  # Default fallback

        # Check TPM limit first
        tpm_delay = self._check_tpm_limit(estimated_tokens)
        if tpm_delay is not None:
            return self._enforce_max_wait(tpm_delay)

        # Check token bucket
        if not self._consume_tokens(estimated_tokens):
            wait_time = self._calculate_wait_time(estimated_tokens)
            jittered_wait = add_jitter(wait_time, self.jitter)
            return self._enforce_max_wait(jittered_wait)

        return None

    def on_request_success(self, tokens_used: int = 0, **kwargs: Any) -> None:
        """Handle successful request.

        Args:
            tokens_used: Actual number of tokens used in request
            **kwargs: Additional parameters
        """
        # Track token usage for TPM
        if tokens_used > 0:
            self._add_token_usage(tokens_used)

        # Refill tokens (no additional action needed for success)
        self._refill_tokens()

    def on_request_failure(
        self,
        exception: Exception,
        estimated_tokens: int = 0,
        **kwargs: Any
    ) -> float | None:
        """Handle failed request.

        Args:
            exception: Exception that occurred
            estimated_tokens: Estimated tokens for the request
            **kwargs: Additional parameters

        Returns:
            Delay in seconds before retry, None if no retry
        """
        # Only handle rate limit errors
        if not is_rate_limit_error(exception):
            return None

        # Try to get retry delay from Retry-After header
        retry_after_delay = None
        if hasattr(exception, 'response'):
            retry_after_delay = parse_retry_after(exception.response)

        if retry_after_delay is not None:
            # Use server-provided delay
            delay = retry_after_delay
        else:
            # Calculate wait time based on token bucket state
            delay = self._calculate_wait_time(estimated_tokens)
            if delay == 0:
                # If no token bucket delay, use small default
                delay = 1.0

        # Add jitter to distribute requests
        delay = add_jitter(delay, self.jitter)

        # Enforce maximum wait time
        return self._enforce_max_wait(delay)

    def get_tokens_available(self) -> float:
        """Get current number of tokens available in bucket.

        Returns:
            Number of tokens currently available
        """
        self._refill_tokens()
        return self._tokens

    def reset_bucket(self) -> None:
        """Reset token bucket to full capacity."""
        self._tokens = float(self.burst_size)
        self._last_refill = time.time()
