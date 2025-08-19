"""Throttling algorithm implementations."""

from __future__ import annotations

import statistics
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from plsno429.base import BaseThrottleAlgorithm
from plsno429.exceptions import CircuitBreakerOpen, ConfigurationError
from plsno429.utils import add_jitter, estimate_tokens, is_rate_limit_error, parse_retry_after

if TYPE_CHECKING:
    from collections.abc import Callable


class RetryAlgorithm(BaseThrottleAlgorithm):
    """Basic retry algorithm with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        **kwargs: Any,
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
            msg = 'max_retries must be non-negative'
            raise ConfigurationError(msg)

        if self.base_delay <= 0:
            msg = 'base_delay must be positive'
            raise ConfigurationError(msg)

        if self.max_delay <= 0:
            msg = 'max_delay must be positive'
            raise ConfigurationError(msg)

        if self.backoff_multiplier <= 0:
            msg = 'backoff_multiplier must be positive'
            raise ConfigurationError(msg)

    def should_throttle(
        self, estimated_tokens: int = 0, model: str | None = None, **kwargs: Any
    ) -> float | None:
        """Check if request should be throttled due to TPM limits.

        Args:
            estimated_tokens: Estimated tokens for upcoming request
            model: Model name for model-specific limits
            **kwargs: Additional parameters

        Returns:
            Delay in seconds if throttling needed, None otherwise
        """
        # Check TPM limit
        tpm_delay = self._check_tpm_limit(estimated_tokens, model)
        if tpm_delay is not None:
            return self._enforce_max_wait(tpm_delay)

        return None

    def on_request_success(
        self, tokens_used: int = 0, model: str | None = None, **kwargs: Any
    ) -> None:
        """Handle successful request.

        Args:
            tokens_used: Number of tokens used in request
            model: Model name for model-specific tracking
            **kwargs: Additional parameters
        """
        # Reset retry counter on success
        self._retry_count = 0

        # Track token usage
        if tokens_used > 0:
            self._add_token_usage(tokens_used, model)

    def on_request_failure(
        self,
        exception: Exception,
        estimated_tokens: int = 0,
        model: str | None = None,
        **kwargs: Any,
    ) -> float | None:
        """Handle failed request and determine retry delay.

        Args:
            exception: Exception that occurred
            estimated_tokens: Estimated tokens for the request
            model: Model name for model-specific tracking
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
                self.max_delay,
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
        **kwargs: Any,
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
        self, exception: Exception, estimated_tokens: int = 0, **kwargs: Any
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


class AdaptiveAlgorithm(BaseThrottleAlgorithm):
    """Adaptive algorithm that learns from 429 patterns and adjusts automatically."""

    def __init__(
        self,
        learning_window: int = 100,
        adaptation_rate: float = 0.1,
        min_delay: float = 0.1,
        max_delay: float = 300.0,
        **kwargs: Any,
    ) -> None:
        """Initialize adaptive algorithm.

        Args:
            learning_window: Number of requests to analyze for learning
            adaptation_rate: How quickly to adapt (0.0-1.0)
            min_delay: Minimum delay between requests
            max_delay: Maximum adaptive delay
            **kwargs: Base class parameters
        """
        super().__init__(**kwargs)

        self.learning_window = learning_window
        self.adaptation_rate = adaptation_rate
        self.min_delay = min_delay
        self.max_delay = max_delay

        # Learning state
        self._request_history: deque[dict[str, Any]] = deque(maxlen=learning_window)
        self._success_rate = 1.0
        self._current_delay = min_delay
        self._consecutive_429s = 0
        self._last_request_time = 0.0

        self._validate_adaptive_config()

    def _validate_adaptive_config(self) -> None:
        """Validate adaptive-specific configuration."""
        if self.learning_window <= 0:
            msg = 'learning_window must be positive'
            raise ConfigurationError(msg)

        if not 0 <= self.adaptation_rate <= 1:
            msg = 'adaptation_rate must be between 0 and 1'
            raise ConfigurationError(msg)

        if self.min_delay < 0:
            msg = 'min_delay must be non-negative'
            raise ConfigurationError(msg)

        if self.max_delay <= self.min_delay:
            msg = 'max_delay must be greater than min_delay'
            raise ConfigurationError(msg)

    def _record_request(self, success: bool, delay_used: float = 0.0, tokens: int = 0) -> None:
        """Record request outcome for learning.

        Args:
            success: Whether request was successful
            delay_used: Delay used before request
            tokens: Number of tokens used
        """
        current_time = time.time()
        self._request_history.append(
            {
                'timestamp': current_time,
                'success': success,
                'delay_used': delay_used,
                'tokens': tokens,
            }
        )

        # Update success rate
        if len(self._request_history) >= 10:  # Need minimum data
            recent_success = sum(1 for req in list(self._request_history)[-20:] if req['success'])
            recent_total = min(20, len(self._request_history))
            self._success_rate = recent_success / recent_total

    def _analyze_patterns(self) -> float:
        """Analyze request patterns and suggest optimal delay.

        Returns:
            Suggested delay in seconds
        """
        if len(self._request_history) < 10:
            return self._current_delay

        recent_requests = list(self._request_history)[-50:]

        # Calculate time-based patterns
        current_time = time.time()
        current_hour = int(current_time // 3600) % 24

        # Find average delay for current hour in history
        hour_delays = [
            req['delay_used']
            for req in recent_requests
            if req['success'] and int(req['timestamp'] // 3600) % 24 == current_hour
        ]

        suggested_delay = statistics.median(hour_delays) if hour_delays else self._current_delay

        # Adjust based on recent success rate
        if self._success_rate < 0.8:
            suggested_delay *= 1.5  # Increase delay if low success rate
        elif self._success_rate > 0.95:
            suggested_delay *= 0.8  # Decrease delay if high success rate

        # Consider consecutive 429s
        if self._consecutive_429s > 3:
            suggested_delay *= 1.2**self._consecutive_429s

        return max(self.min_delay, min(self.max_delay, suggested_delay))

    def _update_delay(self, target_delay: float) -> None:
        """Update current delay using adaptive rate.

        Args:
            target_delay: Target delay to adapt towards
        """
        # Exponential moving average
        self._current_delay = (
            1 - self.adaptation_rate
        ) * self._current_delay + self.adaptation_rate * target_delay
        self._current_delay = max(self.min_delay, min(self.max_delay, self._current_delay))

    def should_throttle(self, estimated_tokens: int = 0, **kwargs: Any) -> float | None:
        """Check if request should be throttled based on learned patterns.

        Args:
            estimated_tokens: Estimated tokens for upcoming request
            **kwargs: Additional parameters

        Returns:
            Delay in seconds if throttling needed, None otherwise
        """
        current_time = time.time()

        # Check TPM limit first
        tpm_delay = self._check_tpm_limit(estimated_tokens)
        if tpm_delay is not None:
            return self._enforce_max_wait(tpm_delay)

        # Analyze patterns and update delay
        suggested_delay = self._analyze_patterns()
        self._update_delay(suggested_delay)

        # Calculate time since last request
        time_since_last = current_time - self._last_request_time

        # If enough time has passed, no throttling needed
        if time_since_last >= self._current_delay:
            self._last_request_time = current_time
            return None

        # Calculate required wait time
        required_wait = self._current_delay - time_since_last
        jittered_wait = add_jitter(required_wait, self.jitter)

        return self._enforce_max_wait(jittered_wait)

    def on_request_success(self, tokens_used: int = 0, **kwargs: Any) -> None:
        """Handle successful request.

        Args:
            tokens_used: Number of tokens used in request
            **kwargs: Additional parameters
        """
        # Reset consecutive 429 counter
        self._consecutive_429s = 0

        # Record successful request
        self._record_request(success=True, delay_used=self._current_delay, tokens=tokens_used)

        # Track token usage for TPM
        if tokens_used > 0:
            self._add_token_usage(tokens_used)

    def on_request_failure(
        self, exception: Exception, estimated_tokens: int = 0, **kwargs: Any
    ) -> float | None:
        """Handle failed request and learn from failure.

        Args:
            exception: Exception that occurred
            estimated_tokens: Estimated tokens for the request
            **kwargs: Additional parameters

        Returns:
            Delay in seconds before retry, None if no retry
        """
        # Record failed request
        self._record_request(success=False, delay_used=self._current_delay, tokens=estimated_tokens)

        # Only handle rate limit errors
        if not is_rate_limit_error(exception):
            return None

        # Increment consecutive 429 counter
        self._consecutive_429s += 1

        # Try to get retry delay from Retry-After header
        retry_after_delay = None
        if hasattr(exception, 'response'):
            retry_after_delay = parse_retry_after(exception.response)

        if retry_after_delay is not None:
            # Learn from server-provided delay
            self._update_delay(retry_after_delay)
            delay = retry_after_delay
        else:
            # Use adaptive delay and increase it due to 429
            adaptive_delay = self._analyze_patterns()
            # Increase delay aggressively on 429
            delay = adaptive_delay * (1.5**self._consecutive_429s)
            self._update_delay(delay)

        # Add jitter to distribute requests
        delay = add_jitter(delay, self.jitter)

        # Enforce maximum wait time
        return self._enforce_max_wait(delay)

    def get_learning_stats(self) -> dict[str, Any]:
        """Get current learning statistics.

        Returns:
            Dictionary with learning stats
        """
        return {
            'success_rate': self._success_rate,
            'current_delay': self._current_delay,
            'consecutive_429s': self._consecutive_429s,
            'requests_analyzed': len(self._request_history),
            'learning_window': self.learning_window,
        }

    def reset_learning(self) -> None:
        """Reset learning state."""
        self._request_history.clear()
        self._success_rate = 1.0
        self._current_delay = self.min_delay
        self._consecutive_429s = 0
        self._last_request_time = 0.0


class SlidingWindowAlgorithm(BaseThrottleAlgorithm):
    """Sliding window algorithm for precise rate limiting."""

    def __init__(
        self,
        window_size: int = 60,
        max_requests: int = 1500,
        cleanup_interval: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize sliding window algorithm.

        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests per window
            cleanup_interval: Cleanup old entries interval in seconds
            **kwargs: Base class parameters
        """
        super().__init__(**kwargs)

        self.window_size = window_size
        self.max_requests = max_requests
        self.cleanup_interval = cleanup_interval

        # Request tracking
        self._request_times: deque[float] = deque()
        self._last_cleanup = time.time()

        self._validate_sliding_window_config()

    def _validate_sliding_window_config(self) -> None:
        """Validate sliding window specific configuration."""
        if self.window_size <= 0:
            msg = 'window_size must be positive'
            raise ConfigurationError(msg)

        if self.max_requests <= 0:
            msg = 'max_requests must be positive'
            raise ConfigurationError(msg)

        if self.cleanup_interval <= 0:
            msg = 'cleanup_interval must be positive'
            raise ConfigurationError(msg)

    def _cleanup_old_requests(self) -> None:
        """Remove request times outside the current window."""
        current_time = time.time()

        # Only cleanup periodically to avoid overhead
        if current_time - self._last_cleanup < self.cleanup_interval:
            return

        cutoff_time = current_time - self.window_size

        # Remove old requests
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()

        self._last_cleanup = current_time

    def _get_current_request_count(self) -> int:
        """Get number of requests in current window."""
        self._cleanup_old_requests()
        return len(self._request_times)

    def _calculate_wait_time(self) -> float:
        """Calculate time to wait until request can be made.

        Returns:
            Time to wait in seconds
        """
        self._cleanup_old_requests()

        if len(self._request_times) < self.max_requests:
            return 0.0

        # Find the oldest request that needs to expire
        oldest_request_time = self._request_times[0]
        current_time = time.time()
        window_expiry = oldest_request_time + self.window_size

        return max(0.0, window_expiry - current_time)

    def should_throttle(self, estimated_tokens: int = 0, **kwargs: Any) -> float | None:
        """Check if request should be throttled due to sliding window or TPM limits.

        Args:
            estimated_tokens: Estimated tokens for upcoming request
            **kwargs: Additional parameters

        Returns:
            Delay in seconds if throttling needed, None otherwise
        """
        # Check TPM limit first
        tpm_delay = self._check_tpm_limit(estimated_tokens)
        if tpm_delay is not None:
            return self._enforce_max_wait(tpm_delay)

        # Check sliding window limit
        if self._get_current_request_count() >= self.max_requests:
            wait_time = self._calculate_wait_time()
            if wait_time > 0:
                jittered_wait = add_jitter(wait_time, self.jitter)
                return self._enforce_max_wait(jittered_wait)

        # Record this request time
        self._request_times.append(time.time())
        return None

    def on_request_success(self, tokens_used: int = 0, **kwargs: Any) -> None:
        """Handle successful request.

        Args:
            tokens_used: Number of tokens used in request
            **kwargs: Additional parameters
        """
        # Track token usage for TPM
        if tokens_used > 0:
            self._add_token_usage(tokens_used)

    def on_request_failure(
        self, exception: Exception, estimated_tokens: int = 0, **kwargs: Any
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
            # Use sliding window wait time
            delay = self._calculate_wait_time()
            if delay == 0:
                delay = 1.0  # Default delay

        # Add jitter to distribute requests
        delay = add_jitter(delay, self.jitter)

        # Enforce maximum wait time
        return self._enforce_max_wait(delay)

    def get_window_stats(self) -> dict[str, Any]:
        """Get current window statistics.

        Returns:
            Dictionary with window stats
        """
        current_count = self._get_current_request_count()
        return {
            'current_requests': current_count,
            'max_requests': self.max_requests,
            'window_size': self.window_size,
            'utilization': current_count / self.max_requests,
            'next_available': self._calculate_wait_time(),
        }

    def reset_window(self) -> None:
        """Reset sliding window state."""
        self._request_times.clear()
        self._last_cleanup = time.time()


class CircuitBreakerAlgorithm(BaseThrottleAlgorithm):
    """Circuit breaker algorithm to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
        half_open_max_calls: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize circuit breaker algorithm.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls to test in half-open state
            **kwargs: Base class parameters
        """
        super().__init__(**kwargs)

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        # Circuit breaker state
        self._state = 'closed'  # closed, open, half_open
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._half_open_successes = 0

        self._validate_circuit_breaker_config()

    def _validate_circuit_breaker_config(self) -> None:
        """Validate circuit breaker specific configuration."""
        if self.failure_threshold <= 0:
            msg = 'failure_threshold must be positive'
            raise ConfigurationError(msg)

        if self.recovery_timeout <= 0:
            msg = 'recovery_timeout must be positive'
            raise ConfigurationError(msg)

        if self.half_open_max_calls <= 0:
            msg = 'half_open_max_calls must be positive'
            raise ConfigurationError(msg)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset.

        Returns:
            True if should attempt reset
        """
        return (
            self._state == 'open' and time.time() - self._last_failure_time >= self.recovery_timeout
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        self._state = 'half_open'
        self._half_open_calls = 0
        self._half_open_successes = 0

    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        self._state = 'closed'
        self._failure_count = 0
        self._half_open_calls = 0
        self._half_open_successes = 0

    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        self._state = 'open'
        self._last_failure_time = time.time()

    def should_throttle(self, estimated_tokens: int = 0, **kwargs: Any) -> float | None:
        """Check if request should be throttled due to circuit breaker or TPM limits.

        Args:
            estimated_tokens: Estimated tokens for upcoming request
            **kwargs: Additional parameters

        Returns:
            Delay in seconds if throttling needed, None otherwise

        Raises:
            CircuitBreakerOpen: If circuit is open and blocking requests
        """
        # Check TPM limit first
        tpm_delay = self._check_tpm_limit(estimated_tokens)
        if tpm_delay is not None:
            return self._enforce_max_wait(tpm_delay)

        # Check circuit breaker state
        if self._state == 'open':
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                msg = (
                    f'Circuit breaker is open. Will retry after '
                    f'{self.recovery_timeout - (time.time() - self._last_failure_time):.1f}s'
                )
                raise CircuitBreakerOpen(
                    msg
                )

        elif self._state == 'half_open':
            if self._half_open_calls >= self.half_open_max_calls:
                # Exceeded half-open calls, go back to open
                self._transition_to_open()
                msg = 'Circuit breaker half-open calls exceeded'
                raise CircuitBreakerOpen(msg)

        # Allow request to proceed
        return None

    def on_request_success(self, tokens_used: int = 0, **kwargs: Any) -> None:
        """Handle successful request.

        Args:
            tokens_used: Number of tokens used in request
            **kwargs: Additional parameters
        """
        if self._state == 'half_open':
            self._half_open_calls += 1
            self._half_open_successes += 1

            # Check if we should close the circuit
            if self._half_open_successes >= self.half_open_max_calls // 2:
                self._transition_to_closed()

        elif self._state == 'closed':
            # Reset failure count on success
            self._failure_count = 0

        # Track token usage for TPM
        if tokens_used > 0:
            self._add_token_usage(tokens_used)

    def on_request_failure(
        self, exception: Exception, estimated_tokens: int = 0, **kwargs: Any
    ) -> float | None:
        """Handle failed request.

        Args:
            exception: Exception that occurred
            estimated_tokens: Estimated tokens for the request
            **kwargs: Additional parameters

        Returns:
            Delay in seconds before retry, None if no retry
        """
        # Check if this is a failure that should affect circuit breaker
        is_rate_limit = is_rate_limit_error(exception)
        is_circuit_breaker_failure = is_rate_limit or isinstance(
            exception, ConnectionError | TimeoutError
        )

        if is_circuit_breaker_failure:
            if self._state == 'closed':
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to_open()

            elif self._state == 'half_open':
                # Failure in half-open state goes back to open
                self._transition_to_open()

        # Only handle rate limit errors for retry logic
        if not is_rate_limit:
            return None

        # Try to get retry delay from Retry-After header
        retry_after_delay = None
        if hasattr(exception, 'response'):
            retry_after_delay = parse_retry_after(exception.response)

        if retry_after_delay is not None:
            delay = retry_after_delay
        else:
            # Use exponential backoff based on failure count
            delay = min(2.0**self._failure_count, 60.0)

        # Add jitter to distribute requests
        delay = add_jitter(delay, self.jitter)

        # Enforce maximum wait time
        return self._enforce_max_wait(delay)

    def get_circuit_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics.

        Returns:
            Dictionary with circuit stats
        """
        stats = {
            'state': self._state,
            'failure_count': self._failure_count,
            'failure_threshold': self.failure_threshold,
        }

        if self._state == 'open':
            time_until_retry = self.recovery_timeout - (time.time() - self._last_failure_time)
            stats['time_until_retry'] = max(0, time_until_retry)

        elif self._state == 'half_open':
            stats.update(
                {
                    'half_open_calls': self._half_open_calls,
                    'half_open_successes': self._half_open_successes,
                    'max_half_open_calls': self.half_open_max_calls,
                }
            )

        return stats

    def reset_circuit(self) -> None:
        """Reset circuit breaker to closed state."""
        self._transition_to_closed()
