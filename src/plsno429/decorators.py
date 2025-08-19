"""Throttling decorators for different HTTP libraries."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from typing import Any

from plsno429.algorithms import AdaptiveAlgorithm, CircuitBreakerAlgorithm, RetryAlgorithm, SlidingWindowAlgorithm, TokenBucketAlgorithm
from plsno429.base import BaseThrottleAlgorithm
from plsno429.types import AsyncFunction, SyncFunction, ThrottledFunction


def _get_algorithm_class(algorithm: str) -> type[BaseThrottleAlgorithm]:
    """Get algorithm class by name.
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        Algorithm class
    """
    algorithms = {
        'retry': RetryAlgorithm,
        'token_bucket': TokenBucketAlgorithm,
        'adaptive': AdaptiveAlgorithm,
        'sliding_window': SlidingWindowAlgorithm,
        'circuit_breaker': CircuitBreakerAlgorithm,
    }

    if algorithm not in algorithms:
        from plsno429.exceptions import ConfigurationError
        raise ConfigurationError(f'Unknown algorithm: {algorithm}')

    return algorithms[algorithm]


def _create_throttled_sync_wrapper(
    func: SyncFunction,
    algorithm: BaseThrottleAlgorithm,
    token_estimate_func: Callable[..., int] | None = None,
    model_func: Callable[..., str] | None = None
) -> SyncFunction:
    """Create synchronous throttled wrapper.
    
    Args:
        func: Function to wrap
        algorithm: Throttling algorithm
        token_estimate_func: Function to estimate token usage
        model_func: Function to extract model name from request
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract model name for model-specific limits
        model = None
        if model_func:
            try:
                model = model_func(*args, **kwargs)
            except Exception:
                pass  # Fall back to default model handling

        # Estimate tokens for the request
        estimated_tokens = 0
        if token_estimate_func:
            estimated_tokens = token_estimate_func(*args, **kwargs)

        # Check if we should throttle before making request
        pre_delay = algorithm.should_throttle(estimated_tokens=estimated_tokens, model=model)
        if pre_delay is not None:
            time.sleep(pre_delay)

        # Execute request with retry logic
        last_exception = None

        while True:
            try:
                result = func(*args, **kwargs)

                # Track successful request
                actual_tokens = estimated_tokens
                if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
                    actual_tokens = result.usage.total_tokens
                elif hasattr(result, 'model') and hasattr(result, 'usage'):
                    # OpenAI response object
                    if hasattr(result.usage, 'total_tokens'):
                        actual_tokens = result.usage.total_tokens
                    if not model and hasattr(result, 'model'):
                        model = result.model

                algorithm.on_request_success(tokens_used=actual_tokens, model=model)
                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                retry_delay = algorithm.on_request_failure(
                    e,
                    estimated_tokens=estimated_tokens,
                    model=model
                )

                if retry_delay is None:
                    # No retry, re-raise the exception
                    raise

                # Wait before retry
                time.sleep(retry_delay)

    return wrapper


def _create_throttled_async_wrapper(
    func: AsyncFunction,
    algorithm: BaseThrottleAlgorithm,
    token_estimate_func: Callable[..., int] | None = None,
    model_func: Callable[..., str] | None = None
) -> AsyncFunction:
    """Create asynchronous throttled wrapper.
    
    Args:
        func: Async function to wrap
        algorithm: Throttling algorithm
        token_estimate_func: Function to estimate token usage
        model_func: Function to extract model name from request
        
    Returns:
        Wrapped async function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract model name for model-specific limits
        model = None
        if model_func:
            try:
                model = model_func(*args, **kwargs)
            except Exception:
                pass  # Fall back to default model handling

        # Estimate tokens for the request
        estimated_tokens = 0
        if token_estimate_func:
            estimated_tokens = token_estimate_func(*args, **kwargs)

        # Check if we should throttle before making request
        pre_delay = algorithm.should_throttle(estimated_tokens=estimated_tokens, model=model)
        if pre_delay is not None:
            await asyncio.sleep(pre_delay)

        # Execute request with retry logic
        last_exception = None

        while True:
            try:
                result = await func(*args, **kwargs)

                # Track successful request
                actual_tokens = estimated_tokens
                if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
                    actual_tokens = result.usage.total_tokens
                elif hasattr(result, 'model') and hasattr(result, 'usage'):
                    # OpenAI response object
                    if hasattr(result.usage, 'total_tokens'):
                        actual_tokens = result.usage.total_tokens
                    if not model and hasattr(result, 'model'):
                        model = result.model

                algorithm.on_request_success(tokens_used=actual_tokens, model=model)
                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                retry_delay = algorithm.on_request_failure(
                    e,
                    estimated_tokens=estimated_tokens,
                    model=model
                )

                if retry_delay is None:
                    # No retry, re-raise the exception
                    raise

                # Wait before retry
                await asyncio.sleep(retry_delay)

    return wrapper


def throttle_requests(
    algorithm: str = 'retry',
    token_estimate_func: Callable[..., int] | None = None,
    model_func: Callable[..., str] | None = None,
    **algorithm_kwargs: Any
) -> Callable[[ThrottledFunction], ThrottledFunction]:
    """Throttle requests library calls.
    
    Args:
        algorithm: Throttling algorithm name
        token_estimate_func: Function to estimate token usage from request
        model_func: Function to extract model name from request arguments
        **algorithm_kwargs: Algorithm-specific configuration
        
    Returns:
        Decorator function
    """
    def decorator(func: ThrottledFunction) -> ThrottledFunction:
        # Create algorithm instance
        algorithm_class = _get_algorithm_class(algorithm)
        algo_instance = algorithm_class(**algorithm_kwargs)

        # Wrap function based on whether it's async
        if asyncio.iscoroutinefunction(func):
            return _create_throttled_async_wrapper(func, algo_instance, token_estimate_func, model_func)
        else:
            return _create_throttled_sync_wrapper(func, algo_instance, token_estimate_func, model_func)

    return decorator


# Aliases for different HTTP libraries
throttle_httpx = throttle_requests
throttle_httpx_async = throttle_requests
throttle_openai = throttle_requests
throttle_openai_async = throttle_requests
