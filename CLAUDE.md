# OpenAI Throttling Library (plsno429)

A simple Python library that automatically handles 429 rate limit errors for OpenAI API calls with multiple throttling algorithms.

## Features

- 🚀 **Simple Usage**: Just add one decorator
- 🔄 **Retry-After Header Support**: Uses OpenAI's exact retry timing
- 📚 **Multiple HTTP Libraries**: Works with requests, httpx, and OpenAI SDK
- ⚡ **Sync/Async Support**: Both synchronous and asynchronous functions
- 🎲 **Request Distribution**: Jitter prevents concurrent request collisions
- 🧠 **Multiple Throttling Algorithms**: Choose the best strategy for your use case
- 📊 **TPM Limit Awareness**: Minute-level throttling to avoid TPM limits
- 🔧 **Adaptive Learning**: Learns from 429 patterns and adjusts automatically
- 📦 **Minimal Dependencies**: Only uses standard library

## Installation

```bash
uv add plsno429
```

## Quick Start

### Basic Usage (All Libraries)

```python
from plsno429 import throttle_requests, throttle_httpx, throttle_httpx_async, throttle_openai, throttle_openai_async

# Simple retry-based throttling
@throttle_requests()
def simple_call():
    # Your API call

# Adaptive throttling with TPM awareness
@throttle_requests(algorithm="adaptive", tpm_limit=90000)
def smart_call():
    # Your API call with intelligent throttling
```

## Throttling Algorithms

### 1. Simple Retry (`algorithm="retry"`) - Default
Basic exponential backoff with Retry-After header support.

```python
@throttle_requests(
    algorithm="retry",
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0
)
```

### 2. Token Bucket (`algorithm="token_bucket"`)
Rate limiting using token bucket algorithm.

```python
@throttle_requests(
    algorithm="token_bucket",
    tpm_limit=90000,
    burst_size=1000,
    refill_rate=1500  # tokens per second
)
```

### 3. Adaptive Learning (`algorithm="adaptive"`)
Learns from 429 patterns and adjusts throttling automatically.

```python
@throttle_requests(
    algorithm="adaptive",
    tpm_limit=90000,
    learning_window=100,
    adaptation_rate=0.1
)
```

### 4. Sliding Window (`algorithm="sliding_window"`)
Tracks requests in a sliding time window.

```python
@throttle_requests(
    algorithm="sliding_window",
    window_size=60,  # seconds
    max_requests=1500,  # per window
    tpm_limit=90000
)
```

### 5. Circuit Breaker (`algorithm="circuit_breaker"`)
Prevents cascading failures with circuit breaker pattern.

```python
@throttle_requests(
    algorithm="circuit_breaker",
    failure_threshold=5,
    recovery_timeout=300,  # seconds
    half_open_max_calls=3
)
```

## Configuration Options

### Common Options (All Algorithms)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | "retry" | Throttling algorithm to use |
| `jitter` | True | Add random delay to distribute requests |
| `tpm_limit` | 90000 | Tokens per minute limit |
| `safety_margin` | 0.9 | Stop at 90% of TPM limit |
| `max_wait_minutes` | 5 | Maximum wait time in minutes |

### Algorithm-Specific Options

#### Retry Algorithm
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | 3 | Maximum number of retry attempts |
| `base_delay` | 1.0 | Base delay in seconds |
| `max_delay` | 60.0 | Maximum delay in seconds |
| `backoff_multiplier` | 2.0 | Exponential backoff multiplier |

#### Token Bucket Algorithm
| Parameter | Default | Description |
|-----------|---------|-------------|
| `burst_size` | 1000 | Maximum burst tokens |
| `refill_rate` | 1500 | Tokens per second refill rate |
| `token_estimate_func` | None | Custom token estimation function |

#### Adaptive Algorithm
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_window` | 100 | Number of requests to analyze |
| `adaptation_rate` | 0.1 | How quickly to adapt (0.0-1.0) |
| `min_delay` | 0.1 | Minimum delay between requests |
| `max_delay` | 300.0 | Maximum adaptive delay |

#### Sliding Window Algorithm
| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 60 | Time window in seconds |
| `max_requests` | 1500 | Max requests per window |
| `cleanup_interval` | 10 | Cleanup old entries interval |

#### Circuit Breaker Algorithm
| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Failures before opening circuit |
| `recovery_timeout` | 300 | Seconds before attempting recovery |
| `half_open_max_calls` | 3 | Max calls in half-open state |

## Advanced Features

### TPM-Aware Throttling
Automatically tracks tokens per minute and waits until next minute boundary when approaching limits.

### Pattern Learning
Adaptive algorithm learns from:
- Time-of-day patterns
- Consecutive 429 error patterns  
- Success/failure ratios
- Response time trends

### Minute-Level Recovery
When TPM limits are hit, automatically waits until the next minute boundary instead of arbitrary delays.

### Multi-Model Support
Different TPM limits for different OpenAI models:

```python
@throttle_requests(
    algorithm="adaptive",
    model_limits={
        "gpt-4": 90000,
        "gpt-3.5-turbo": 90000,
        "text-embedding-ada-002": 1000000
    }
)
```

## How Different Algorithms Work

### Retry Algorithm Flow
1. Execute function → 429 error → Check Retry-After → Wait → Retry

### Token Bucket Flow  
1. Check available tokens → Consume tokens → Execute → Refill tokens over time

### Adaptive Flow
1. Track request patterns → Learn optimal delays → Predict and prevent 429s

### Sliding Window Flow
1. Track requests in time window → Block if limit exceeded → Allow when window slides

### Circuit Breaker Flow
1. Monitor failure rate → Open circuit on threshold → Gradually test recovery

## Use Cases by Algorithm

### Retry Algorithm
- **Best for**: Simple applications, low request volume
- **Pros**: Simple, reliable, works with any API
- **Cons**: Reactive only, may waste time on retries

### Token Bucket Algorithm  
- **Best for**: Steady request rates, predictable workloads
- **Pros**: Smooth rate limiting, allows bursts
- **Cons**: Requires accurate token estimation

### Adaptive Algorithm
- **Best for**: Production applications, varying workloads
- **Pros**: Self-optimizing, learns patterns, prevents 429s
- **Cons**: More complex, requires warm-up period

### Sliding Window Algorithm
- **Best for**: High-volume applications with strict rate limits
- **Pros**: Precise rate limiting, prevents TPM violations
- **Cons**: Memory overhead for tracking requests

### Circuit Breaker Algorithm
- **Best for**: Critical applications, cascading failure prevention
- **Pros**: Prevents system overload, fast failure recovery
- **Cons**: May be overly conservative

## Performance Comparison

| Algorithm | Memory Usage | CPU Overhead | Learning Ability | Prevention vs Reaction |
|-----------|--------------|--------------|------------------|----------------------|
| Retry | Low | Very Low | None | Reactive |
| Token Bucket | Low | Low | None | Preventive |
| Adaptive | Medium | Medium | High | Predictive |
| Sliding Window | High | Medium | None | Preventive |
| Circuit Breaker | Low | Low | Basic | Protective |

## Requirements

- Python 3.12+
- No external dependencies (uses only standard library)
- Compatible with requests, httpx, and OpenAI Python SDK

## License

Apache 2.0 License

## Contributing

Issues and pull requests are welcome!

## Notes

- Choose algorithm based on your specific use case and requirements
- Adaptive algorithm provides best long-term performance but needs time to learn
- All algorithms respect OpenAI's Retry-After headers when provided
- Consider your application's latency requirements when choosing algorithms
- Circuit breaker is recommended for mission-critical applications
