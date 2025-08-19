"""Base throttling algorithm interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from plsno429.exceptions import RateLimitExceeded
from plsno429.utils import calculate_wait_until_next_minute


class BaseThrottleAlgorithm(ABC):
    """Abstract base class for throttling algorithms."""

    def __init__(
        self,
        tpm_limit: int = 90000,
        safety_margin: float = 0.9,
        max_wait_minutes: float = 5.0,
        jitter: bool = True,
        model_limits: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize base throttling algorithm.

        Args:
            tpm_limit: Default tokens per minute limit
            safety_margin: Safety margin (0.0-1.0) to stop before limit
            max_wait_minutes: Maximum time to wait in minutes
            jitter: Whether to add jitter to delays
            model_limits: Model-specific TPM limits (e.g., {'gpt-4': 90000, 'gpt-3.5-turbo': 90000})
            **kwargs: Algorithm-specific parameters
        """
        self.tpm_limit = tpm_limit
        self.safety_margin = safety_margin
        self.max_wait_minutes = max_wait_minutes
        self.jitter = jitter
        self.model_limits = model_limits or {}

        # Enhanced TPM tracking - separate tracking per model
        self._token_usage: dict[str, dict[int, int]] = {}  # model -> minute -> token count
        self._model_token_usage: dict[int, int] = {}  # minute -> total tokens across all models
        self._last_cleanup = time.time()

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate algorithm configuration."""
        if self.tpm_limit <= 0:
            from plsno429.exceptions import ConfigurationError

            msg = 'tpm_limit must be positive'
            raise ConfigurationError(msg)

        if not 0 <= self.safety_margin <= 1:
            from plsno429.exceptions import ConfigurationError

            msg = 'safety_margin must be between 0 and 1'
            raise ConfigurationError(msg)

        if self.max_wait_minutes <= 0:
            from plsno429.exceptions import ConfigurationError

            msg = 'max_wait_minutes must be positive'
            raise ConfigurationError(msg)

    def _cleanup_old_token_usage(self) -> None:
        """Remove token usage data older than 2 minutes."""
        current_time = time.time()
        if current_time - self._last_cleanup < 30:  # Cleanup every 30 seconds
            return

        current_minute = int(current_time // 60)
        cutoff_minute = current_minute - 2

        # Clean up model-specific usage
        for model_name in self._token_usage:
            keys_to_remove = [
                minute for minute in self._token_usage[model_name] if minute < cutoff_minute
            ]
            for key in keys_to_remove:
                del self._token_usage[model_name][key]

        # Clean up total usage
        keys_to_remove = [minute for minute in self._model_token_usage if minute < cutoff_minute]
        for key in keys_to_remove:
            del self._model_token_usage[key]

        self._last_cleanup = current_time

    def _get_current_tpm_usage(self, model: str | None = None) -> int:
        """Get current tokens per minute usage for a specific model or total.

        Args:
            model: Model name to check, or None for total usage

        Returns:
            Current token usage for the model or total
        """
        self._cleanup_old_token_usage()
        current_minute = int(time.time() // 60)

        if model is not None:
            # Return usage for specific model
            if model not in self._token_usage:
                return 0
            return self._token_usage[model].get(current_minute, 0)
        else:
            # Return total usage across all models
            return self._model_token_usage.get(current_minute, 0)

    def _add_token_usage(self, tokens: int, model: str | None = None) -> None:
        """Add token usage to current minute for a specific model.

        Args:
            tokens: Number of tokens used
            model: Model name, or None for default tracking
        """
        current_minute = int(time.time() // 60)

        # Track model-specific usage
        if model is not None:
            if model not in self._token_usage:
                self._token_usage[model] = {}
            self._token_usage[model][current_minute] = (
                self._token_usage[model].get(current_minute, 0) + tokens
            )

        # Track total usage
        self._model_token_usage[current_minute] = (
            self._model_token_usage.get(current_minute, 0) + tokens
        )

    def _get_effective_tpm_limit(self, model: str | None = None) -> int:
        """Get the effective TPM limit for a model.

        Args:
            model: Model name to get limit for

        Returns:
            Effective TPM limit for the model
        """
        if model and model in self.model_limits:
            return int(self.model_limits[model] * self.safety_margin)
        return int(self.tpm_limit * self.safety_margin)

    def _check_tpm_limit(self, estimated_tokens: int = 0, model: str | None = None) -> float | None:
        """Check if adding tokens would exceed TPM limit.

        Args:
            estimated_tokens: Estimated tokens for upcoming request
            model: Model name for model-specific limits

        Returns:
            Seconds to wait until next minute if limit would be exceeded, None otherwise
        """
        effective_limit = self._get_effective_tpm_limit(model)

        # Check both model-specific and total limits
        model_usage = self._get_current_tpm_usage(model) if model else 0
        total_usage = self._get_current_tpm_usage(None)

        # Check model-specific limit if applicable
        if model and model_usage + estimated_tokens > effective_limit:
            return calculate_wait_until_next_minute()

        # Check total limit
        total_effective_limit = int(self.tpm_limit * self.safety_margin)
        if total_usage + estimated_tokens > total_effective_limit:
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
            msg = (
                f'Rate limit delay ({delay:.1f}s) exceeds maximum wait time '
                f'({max_wait_seconds:.1f}s)'
            )
            raise RateLimitExceeded(
                msg
            )

        return delay

    def get_tpm_stats(self, model: str | None = None) -> dict[str, Any]:
        """Get current TPM statistics.

        Args:
            model: Model name to get stats for, or None for total stats

        Returns:
            Dictionary with TPM statistics
        """
        current_usage = self._get_current_tpm_usage(model)
        effective_limit = self._get_effective_tpm_limit(model)

        stats = {
            'current_usage': current_usage,
            'effective_limit': effective_limit,
            'raw_limit': self.model_limits.get(model, self.tpm_limit) if model else self.tpm_limit,
            'safety_margin': self.safety_margin,
            'utilization': current_usage / effective_limit if effective_limit > 0 else 0.0,
            'tokens_remaining': max(0, effective_limit - current_usage),
            'next_reset_seconds': calculate_wait_until_next_minute(),
        }

        if model:
            stats['model'] = model
            stats['total_usage'] = self._get_current_tpm_usage(None)

        return stats

    def get_all_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get TPM statistics for all tracked models.

        Returns:
            Dictionary mapping model names to their TPM stats
        """
        stats = {'total': self.get_tpm_stats(None)}

        # Get stats for each model that has been used
        for model_name in self._token_usage:
            if any(self._token_usage[model_name].values()):  # Has usage data
                stats[model_name] = self.get_tpm_stats(model_name)

        # Add stats for configured models even if not used yet
        for model_name in self.model_limits:
            if model_name not in stats:
                stats[model_name] = self.get_tpm_stats(model_name)

        return stats

    def reset_tpm_tracking(self, model: str | None = None) -> None:
        """Reset TPM tracking for a specific model or all models.

        Args:
            model: Model name to reset, or None to reset all tracking
        """
        if model is not None:
            # Reset specific model
            if model in self._token_usage:
                del self._token_usage[model]
        else:
            # Reset all tracking
            self._token_usage.clear()
            self._model_token_usage.clear()
            self._last_cleanup = time.time()
