"""Utility functions for plsno429."""

import random
import time

from .types import HTTPResponse


def parse_retry_after(response: HTTPResponse) -> float | None:
    """Parse Retry-After header from HTTP response.
    
    Args:
        response: HTTP response object from any library
        
    Returns:
        Retry delay in seconds, or None if not found
    """
    retry_after = None

    # Try different ways to get headers based on response type
    if hasattr(response, 'headers'):
        headers = response.headers
    elif hasattr(response, 'response') and hasattr(response.response, 'headers'):
        headers = response.response.headers
    else:
        return None

    # Ensure headers is dict-like and has items() method
    if not hasattr(headers, 'items'):
        return None

    # Get Retry-After header (case insensitive)
    for key, value in headers.items():
        if key.lower() == 'retry-after':
            retry_after = value
            break

    if not retry_after:
        return None

    # Parse integer seconds
    try:
        return float(retry_after)
    except ValueError:
        pass

    # Parse HTTP date format (RFC 2822)
    try:
        import email.utils
        from datetime import datetime
        target_time = email.utils.parsedate_to_datetime(retry_after)
        current_time = datetime.now(target_time.tzinfo)
        delta = (target_time - current_time).total_seconds()
        return max(0.0, float(delta))
    except (ValueError, TypeError):
        return None


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception indicates a rate limit (429) error.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if this is a 429 rate limit error
    """
    # Check status code in various exception types
    if hasattr(exception, 'status_code'):
        return bool(exception.status_code == 429)

    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        return bool(exception.response.status_code == 429)

    if hasattr(exception, 'code'):
        return bool(exception.code == 429)

    # Check error message for common rate limit indicators
    error_msg = str(exception).lower()
    rate_limit_indicators = [
        '429',
        'rate limit',
        'rate_limit_exceeded',
        'too many requests',
        'quota exceeded',
        'request rate exceeded'
    ]

    return any(indicator in error_msg for indicator in rate_limit_indicators)


def add_jitter(delay: float, jitter: bool = True) -> float:
    """Add random jitter to delay to distribute requests.
    
    Args:
        delay: Base delay in seconds
        jitter: Whether to add jitter
        
    Returns:
        Delay with jitter applied
    """
    if not jitter or delay <= 0:
        return delay

    # Add up to 25% random jitter
    jitter_amount = delay * 0.25 * random.random()
    return delay + jitter_amount


def estimate_tokens(text: str, model: str = 'gpt-4') -> int:
    """Estimate token count for text.
    
    Args:
        text: Text to estimate tokens for
        model: Model name for context
        
    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 characters per token for most models
    # This is a rough approximation
    return len(text) // 4 + 1


def get_current_minute_boundary() -> float:
    """Get timestamp of next minute boundary.
    
    Returns:
        Unix timestamp of next minute boundary
    """
    current_time = time.time()
    current_minute = int(current_time // 60) * 60
    next_minute = current_minute + 60
    return next_minute


def calculate_wait_until_next_minute() -> float:
    """Calculate seconds to wait until next minute boundary.
    
    Returns:
        Seconds to wait
    """
    current_time = time.time()
    seconds_into_minute = current_time % 60
    
    # If we're at an exact minute boundary, no wait needed
    if seconds_into_minute == 0.0:
        return 0.0
    
    # Otherwise, wait until next minute
    return 60.0 - seconds_into_minute
