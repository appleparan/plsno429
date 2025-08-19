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

**How it works**: A reactive approach that waits for 429 errors, then retries with exponential backoff. 

The algorithm doubles the delay after each failed attempt: 1s → 2s → 4s → 8s, etc. If the API provides a `Retry-After` header, it uses that exact timing instead of exponential backoff. This is the simplest and most reliable approach for basic use cases.

**When to use**: 
- Simple applications with occasional API calls
- When you want predictable, well-tested behavior
- Low to medium request volumes
- Getting started with rate limiting

```python
@throttle_requests(
    algorithm="retry",
    max_retries=3,        # Stop after 3 failed attempts
    base_delay=1.0,       # Start with 1 second delay
    max_delay=60.0,       # Never wait more than 60 seconds
    backoff_multiplier=2.0  # Double delay each time
)
```

**Pros**: Simple, reliable, works with any API, respects server hints  
**Cons**: Reactive only (waits for errors), may waste time on repeated failures

### 2. Token Bucket (`algorithm="token_bucket"`)

**How it works**: Implements a classic token bucket for smooth rate limiting. Imagine a bucket that holds tokens - each API request consumes tokens, and the bucket refills at a steady rate.

The algorithm prevents sudden bursts while allowing natural request patterns. If you need 100 tokens but only have 50, it calculates exactly how long to wait for 50 more tokens to be added to the bucket.

**When to use**:
- Steady, predictable request rates
- Applications that need to handle bursts gracefully  
- When you know your token consumption patterns
- Production apps with consistent load

```python
@throttle_requests(
    algorithm="token_bucket",
    tpm_limit=90000,      # Total TPM limit
    burst_size=1000,      # Max tokens in bucket (allows bursts)
    refill_rate=1500,     # Add 1500 tokens per second
    token_estimate_func=custom_estimator  # Custom token counting
)
```

**Pros**: Smooth rate limiting, allows bursts, predictable behavior  
**Cons**: Requires accurate token estimation, more complex than retry

### 3. Adaptive Learning (`algorithm="adaptive"`)

**How it works**: The smartest algorithm that learns from your actual usage patterns and API behavior. It tracks success rates, response times, and 429 error patterns to predict optimal delays.

The algorithm adjusts its behavior based on:
- **Time-of-day patterns**: Different delays for peak vs. off-peak hours
- **Success rate feedback**: Increases delays if seeing many failures
- **429 error clustering**: Detects when errors come in waves
- **Server response analysis**: Learns from `Retry-After` headers

**When to use**:
- Production applications with varying loads
- Long-running applications that can learn over time
- Complex usage patterns (batch processing + real-time)
- When you want "set it and forget it" behavior

```python
@throttle_requests(
    algorithm="adaptive",
    tpm_limit=90000,
    learning_window=100,    # Analyze last 100 requests
    adaptation_rate=0.1,    # How quickly to adapt (0.0-1.0) 
    min_delay=0.1,         # Never go below 100ms
    max_delay=300.0        # Never wait more than 5 minutes
)
```

**Pros**: Self-optimizing, learns patterns, prevents 429s proactively, handles varying loads  
**Cons**: Complex behavior, needs warm-up period, harder to debug

### 4. Sliding Window (`algorithm="sliding_window"`)

**How it works**: Maintains a precise sliding time window of all requests. Unlike simple counters that reset every minute, this tracks the exact timestamp of each request and continuously slides the window forward.

For example, with a 60-second window allowing 1500 requests: at 2:30:45, it counts all requests from 2:29:45 to 2:30:45. This provides the most accurate rate limiting possible.

**When to use**:
- High-volume applications with strict rate limits
- When you need precise control over request timing
- Applications that can't afford to exceed rate limits
- Compliance-critical environments

```python
@throttle_requests(
    algorithm="sliding_window",
    window_size=60,       # 60-second sliding window
    max_requests=1500,    # Max 1500 requests per window
    cleanup_interval=10,  # Clean up old entries every 10s
    tpm_limit=90000      # Also respect TPM limits
)
```

**Pros**: Most precise rate limiting, prevents violations, handles bursts well  
**Cons**: Higher memory usage, more CPU overhead for large volumes

### 5. Circuit Breaker (`algorithm="circuit_breaker"`)

**How it works**: Implements the circuit breaker pattern to prevent cascading failures. Like an electrical circuit breaker, it has three states:

- **Closed**: Normal operation, requests pass through
- **Open**: Too many failures detected, blocks all requests temporarily  
- **Half-Open**: Testing if the service has recovered

This prevents your application from overwhelming a failing service and allows graceful degradation.

**When to use**:
- Mission-critical applications
- Services that must handle downstream failures gracefully
- When you need to prevent cascading failures
- Applications with multiple API dependencies

```python
@throttle_requests(
    algorithm="circuit_breaker",
    failure_threshold=5,    # Open after 5 consecutive failures
    recovery_timeout=300,   # Wait 5 minutes before testing recovery
    half_open_max_calls=3   # Test with max 3 calls in half-open state
)
```

**Pros**: Prevents cascade failures, fast failure detection, graceful degradation  
**Cons**: May be overly conservative, can block valid requests, complex state management

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
```
Request comes in
├── Execute function immediately
├── Success? → Return result ✅
└── 429 error?
    ├── Check retry count < max_retries?
    │   ├── Yes: Parse Retry-After header or use exponential backoff
    │   ├── Wait calculated delay (with jitter)
    │   └── Retry request
    └── No: Raise exception ❌
```

### Token Bucket Flow
```
Request comes in
├── Refill bucket based on time elapsed
├── Estimate tokens needed for request
├── Enough tokens in bucket?
│   ├── Yes: Consume tokens → Execute request → Return result ✅
│   └── No: Calculate wait time for token availability
└── Wait for tokens → Retry token check
```

### Adaptive Flow
```
Request comes in
├── Analyze historical patterns (success rate, timing, 429s)
├── Calculate optimal delay based on learning
├── Time since last request > optimal delay?
│   ├── Yes: Execute immediately
│   └── No: Wait remaining time
├── Execute request
├── Record outcome (success/failure, tokens, delay)
└── Update learning model for future requests
```

### Sliding Window Flow
```
Request comes in
├── Clean up old requests outside window
├── Count current requests in sliding window
├── Current count < max_requests?
│   ├── Yes: Add timestamp to window → Execute → Return ✅
│   └── No: Calculate wait time until oldest request expires
└── Wait for window to slide → Retry count check
```

### Circuit Breaker Flow
```
Request comes in
├── Check circuit state
├── CLOSED (normal): Execute request
│   ├── Success: Reset failure counter
│   └── Failure: Increment counter → Threshold reached? → Open circuit
├── OPEN (blocking): Check recovery timeout
│   ├── Timeout passed: Transition to HALF_OPEN
│   └── Still cooling down: Block request ❌
└── HALF_OPEN (testing): Allow limited test requests
    ├── Success: Close circuit (recovery successful)
    └── Failure: Back to OPEN (still failing)
```

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

## Real-World Examples

### Example 1: Data Processing Pipeline (Adaptive)
```python
from plsno429 import throttle_openai
import openai

@throttle_openai(
    algorithm="adaptive",
    tpm_limit=90000,
    learning_window=50,
    adaptation_rate=0.15,  # Learn relatively quickly
    min_delay=0.05,        # Very responsive
    max_delay=120.0        # Don't wait too long
)
def process_document_batch(documents):
    """Process a batch of documents with adaptive learning."""
    results = []
    for doc in documents:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Summarize: {doc}"}]
        )
        results.append(response.choices[0].message.content)
    return results
```

### Example 2: High-Volume API Service (Sliding Window)
```python
from plsno429 import throttle_requests
import httpx

@throttle_requests(
    algorithm="sliding_window",
    window_size=60,        # 1-minute window
    max_requests=1500,     # OpenAI's RPM limit
    tpm_limit=90000,       # Also respect TPM
    cleanup_interval=5     # Clean up frequently
)
async def translation_service(texts):
    """High-volume translation service with precise rate limiting."""
    async with httpx.AsyncClient() as client:
        tasks = []
        for text in texts:
            task = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": f"Translate: {text}"}]
                }
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [r.json() for r in results]
```

### Example 3: Mission-Critical Application (Circuit Breaker)
```python
from plsno429 import throttle_openai
from plsno429.exceptions import CircuitBreakerOpen
import logging

@throttle_openai(
    algorithm="circuit_breaker",
    failure_threshold=3,     # Open after 3 failures
    recovery_timeout=180,    # 3-minute recovery
    half_open_max_calls=2    # Test with 2 calls
)
def critical_ai_service(prompt):
    """Mission-critical service with graceful degradation."""
    try:
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
    except CircuitBreakerOpen as e:
        logging.warning(f"AI service unavailable: {e}")
        # Fallback to cached responses or simpler processing
        return {"fallback": "Service temporarily unavailable"}
```

### Example 4: Multi-Model Application (Model-Specific Limits)
```python
from plsno429 import throttle_openai

def extract_model_from_request(**kwargs):
    """Extract model name from request parameters."""
    return kwargs.get('model', 'gpt-3.5-turbo')

@throttle_openai(
    algorithm="adaptive",
    model_func=extract_model_from_request,
    model_limits={
        "gpt-4": 40000,                    # Lower limit for expensive model
        "gpt-3.5-turbo": 90000,           # Standard limit
        "text-embedding-ada-002": 1000000  # Higher limit for embeddings
    },
    tpm_limit=90000  # Global fallback limit
)
def multi_model_ai_service(prompt, model="gpt-3.5-turbo", task_type="chat"):
    """Service supporting multiple models with different limits."""
    if task_type == "embedding":
        return openai.Embedding.create(
            model="text-embedding-ada-002",
            input=prompt
        )
    else:
        return openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
```

## Requirements

- Python 3.12+
- No external dependencies (uses only standard library)
- Compatible with requests, httpx, and OpenAI Python SDK

## Development

### Install dependencies

```bash
uv sync --group dev --group docs
```

### Run tests

```bash
uv run pytest
```

### Formatting and linting

```bash
uv run ruff format
uv run ruff check --fix .
```

### Build package

```bash
uv build
```

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