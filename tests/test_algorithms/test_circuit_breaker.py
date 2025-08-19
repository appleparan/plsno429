"""Tests for circuit breaker algorithm."""

import time
from unittest.mock import Mock, patch

import pytest

from plsno429.algorithms import CircuitBreakerAlgorithm
from plsno429.exceptions import CircuitBreakerOpen, ConfigurationError, RateLimitExceeded


class TestCircuitBreakerAlgorithm:
    def test_initialization_with_defaults(self):
        algorithm = CircuitBreakerAlgorithm()

        assert algorithm.failure_threshold == 5
        assert algorithm.recovery_timeout == 300.0
        assert algorithm.half_open_max_calls == 3
        assert algorithm._state == 'closed'
        assert algorithm._failure_count == 0

    def test_initialization_with_custom_values(self):
        algorithm = CircuitBreakerAlgorithm(
            failure_threshold=3, recovery_timeout=120.0, half_open_max_calls=2
        )

        assert algorithm.failure_threshold == 3
        assert algorithm.recovery_timeout == 120.0
        assert algorithm.half_open_max_calls == 2

    def test_invalid_configuration(self):
        with pytest.raises(ConfigurationError):
            CircuitBreakerAlgorithm(failure_threshold=0)

        with pytest.raises(ConfigurationError):
            CircuitBreakerAlgorithm(recovery_timeout=0)

        with pytest.raises(ConfigurationError):
            CircuitBreakerAlgorithm(half_open_max_calls=0)

        with pytest.raises(ConfigurationError):
            CircuitBreakerAlgorithm(failure_threshold=-1)

    def test_should_throttle_closed_state_no_throttling(self):
        algorithm = CircuitBreakerAlgorithm()

        result = algorithm.should_throttle()
        assert result is None
        assert algorithm._state == 'closed'

    def test_should_throttle_with_tpm_limit(self):
        algorithm = CircuitBreakerAlgorithm(tpm_limit=1000, safety_margin=0.9)

        # Add tokens to exceed TPM limit
        algorithm._add_token_usage(950)

        result = algorithm.should_throttle(estimated_tokens=100)
        assert result is not None
        assert result > 0

    def test_should_throttle_open_state_blocks_requests(self):
        algorithm = CircuitBreakerAlgorithm(recovery_timeout=60.0)
        algorithm._state = 'open'
        algorithm._last_failure_time = 100.0

        # Mock both time modules to avoid TPM checking issues
        with (
            patch('time.time', return_value=130.0),  # Only 30 seconds passed
            pytest.raises(CircuitBreakerOpen),
        ):
            algorithm.should_throttle()

    def test_should_throttle_open_state_transitions_to_half_open(self):
        algorithm = CircuitBreakerAlgorithm(recovery_timeout=60.0)
        algorithm._state = 'open'
        algorithm._last_failure_time = 100.0

        # Current time is after recovery timeout
        with patch('time.time', return_value=170.0):  # 70 seconds passed
            result = algorithm.should_throttle()
            assert result is None
            assert algorithm._state == 'half_open'

    def test_should_throttle_half_open_allows_limited_calls(self):
        algorithm = CircuitBreakerAlgorithm(half_open_max_calls=3)
        algorithm._state = 'half_open'
        algorithm._half_open_calls = 2

        result = algorithm.should_throttle()
        assert result is None

    def test_should_throttle_half_open_exceeds_max_calls(self):
        algorithm = CircuitBreakerAlgorithm(half_open_max_calls=3)
        algorithm._state = 'half_open'
        algorithm._half_open_calls = 3

        with pytest.raises(CircuitBreakerOpen):
            algorithm.should_throttle()

        assert algorithm._state == 'open'

    def test_on_request_success_closed_state(self):
        algorithm = CircuitBreakerAlgorithm()
        algorithm._failure_count = 3

        algorithm.on_request_success(tokens_used=150)

        # Should reset failure count
        assert algorithm._failure_count == 0
        assert algorithm._state == 'closed'

    def test_on_request_success_half_open_state(self):
        algorithm = CircuitBreakerAlgorithm(half_open_max_calls=4)
        algorithm._state = 'half_open'
        algorithm._half_open_calls = 0
        algorithm._half_open_successes = 0

        algorithm.on_request_success()

        assert algorithm._half_open_calls == 1
        assert algorithm._half_open_successes == 1
        assert algorithm._state == 'half_open'  # Not enough successes yet

    def test_on_request_success_half_open_transitions_to_closed(self):
        algorithm = CircuitBreakerAlgorithm(half_open_max_calls=4)
        algorithm._state = 'half_open'
        algorithm._half_open_calls = 1
        algorithm._half_open_successes = 1

        algorithm.on_request_success()

        # Should transition to closed after half_open_max_calls // 2 successes
        assert algorithm._state == 'closed'
        assert algorithm._failure_count == 0

    def test_on_request_failure_non_circuit_breaker_error(self):
        algorithm = CircuitBreakerAlgorithm()
        exception = ValueError('Some other error')

        result = algorithm.on_request_failure(exception)
        assert result is None
        assert algorithm._failure_count == 0

    def test_on_request_failure_rate_limit_error_closed_state(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=3, jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = None

        # First failure
        result = algorithm.on_request_failure(exception)
        assert result == 2.0  # 2^1
        assert algorithm._failure_count == 1
        assert algorithm._state == 'closed'

    def test_on_request_failure_transitions_to_open(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=2, jitter=False)
        algorithm._failure_count = 1

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        result = algorithm.on_request_failure(exception)
        assert result == 4.0  # 2^2
        assert algorithm._failure_count == 2
        assert algorithm._state == 'open'

    def test_on_request_failure_connection_error_affects_circuit(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=2)
        algorithm._failure_count = 1

        exception = ConnectionError('Connection failed')

        result = algorithm.on_request_failure(exception)
        assert result is None  # No retry for non-rate-limit errors
        assert algorithm._failure_count == 2
        assert algorithm._state == 'open'

    def test_on_request_failure_half_open_transitions_to_open(self):
        algorithm = CircuitBreakerAlgorithm()
        algorithm._state = 'half_open'

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        algorithm.on_request_failure(exception)
        assert algorithm._state == 'open'

    def test_on_request_failure_with_retry_after_header(self):
        algorithm = CircuitBreakerAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '30'}

        result = algorithm.on_request_failure(exception)
        assert result == 30.0

    def test_exponential_backoff_capped_at_max(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=10, jitter=False)
        algorithm._failure_count = 10  # Very high failure count

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        result = algorithm.on_request_failure(exception)
        assert result == 60.0  # Capped at max delay

    def test_get_circuit_stats_closed_state(self):
        algorithm = CircuitBreakerAlgorithm()
        algorithm._failure_count = 2

        stats = algorithm.get_circuit_stats()

        assert stats['state'] == 'closed'
        assert stats['failure_count'] == 2
        assert stats['failure_threshold'] == 5

    @patch('plsno429.algorithms.time.time')
    def test_get_circuit_stats_open_state(self, mock_time):
        algorithm = CircuitBreakerAlgorithm(recovery_timeout=300.0)
        algorithm._state = 'open'
        algorithm._last_failure_time = 100.0

        mock_time.return_value = 200.0  # 100 seconds passed

        stats = algorithm.get_circuit_stats()

        assert stats['state'] == 'open'
        assert stats['time_until_retry'] == 200.0  # 300 - 100

    def test_get_circuit_stats_half_open_state(self):
        algorithm = CircuitBreakerAlgorithm()
        algorithm._state = 'half_open'
        algorithm._half_open_calls = 2
        algorithm._half_open_successes = 1

        stats = algorithm.get_circuit_stats()

        assert stats['state'] == 'half_open'
        assert stats['half_open_calls'] == 2
        assert stats['half_open_successes'] == 1
        assert stats['max_half_open_calls'] == 3

    def test_reset_circuit(self):
        algorithm = CircuitBreakerAlgorithm()

        # Set some state
        algorithm._state = 'open'
        algorithm._failure_count = 5
        algorithm._half_open_calls = 2
        algorithm._half_open_successes = 1
        algorithm._last_failure_time = 100.0

        algorithm.reset_circuit()

        # Should reset to closed state
        assert algorithm._state == 'closed'
        assert algorithm._failure_count == 0
        assert algorithm._half_open_calls == 0
        assert algorithm._half_open_successes == 0

    def test_max_wait_time_exceeded(self):
        algorithm = CircuitBreakerAlgorithm(
            max_wait_minutes=0.01  # 0.6 seconds
        )

        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '120'}  # 2 minutes

        with pytest.raises(RateLimitExceeded):
            algorithm.on_request_failure(exception)

    def test_circuit_breaker_state_machine_flow(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=2, recovery_timeout=60.0)

        # Start in closed state
        assert algorithm._state == 'closed'

        # First failure
        exception = Mock()
        exception.status_code = 429
        exception.response = None

        algorithm.on_request_failure(exception)
        assert algorithm._state == 'closed'
        assert algorithm._failure_count == 1

        # Second failure -> open
        algorithm.on_request_failure(exception)
        assert algorithm._state == 'open'
        assert algorithm._failure_count == 2

        # Try to make request in open state (too early)
        early_time = algorithm._last_failure_time + 30.0
        with patch('time.time', return_value=early_time), pytest.raises(CircuitBreakerOpen):
            algorithm.should_throttle()

        # Wait for recovery timeout -> half-open
        late_time = algorithm._last_failure_time + 70.0
        with patch('time.time', return_value=late_time):
            result = algorithm.should_throttle()
            assert result is None
            assert algorithm._state == 'half_open'

        # Success in half-open -> closed
        algorithm.on_request_success()
        algorithm.on_request_success()  # Need multiple successes
        assert algorithm._state == 'closed'

    def test_circuit_breaker_open_exception_message(self):
        algorithm = CircuitBreakerAlgorithm(recovery_timeout=60.0)
        algorithm._state = 'open'
        algorithm._last_failure_time = time.time() - 30.0  # 30 seconds ago

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            algorithm.should_throttle()

        # Should include time until retry in message
        assert 'Circuit breaker is open' in str(exc_info.value)
        assert 'Will retry after' in str(exc_info.value)

    def test_should_attempt_reset_logic(self):
        algorithm = CircuitBreakerAlgorithm(recovery_timeout=60.0)

        # Test closed state
        algorithm._state = 'closed'
        assert not algorithm._should_attempt_reset()

        # Test open state, not enough time passed
        algorithm._state = 'open'
        algorithm._last_failure_time = time.time() - 30.0
        assert not algorithm._should_attempt_reset()

        # Test open state, enough time passed
        algorithm._last_failure_time = time.time() - 70.0
        assert algorithm._should_attempt_reset()

    def test_timeout_error_affects_circuit_breaker(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=1)

        exception = TimeoutError('Request timeout')

        result = algorithm.on_request_failure(exception)
        assert result is None  # No retry for timeout
        assert algorithm._failure_count == 1
        assert algorithm._state == 'open'  # Should transition to open

    def test_multiple_consecutive_failures_in_closed_state(self):
        algorithm = CircuitBreakerAlgorithm(failure_threshold=3, jitter=False)

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        # First two failures stay closed
        algorithm.on_request_failure(exception)
        assert algorithm._state == 'closed'
        algorithm.on_request_failure(exception)
        assert algorithm._state == 'closed'

        # Third failure opens circuit
        algorithm.on_request_failure(exception)
        assert algorithm._state == 'open'

    def test_half_open_failure_immediately_opens_circuit(self):
        algorithm = CircuitBreakerAlgorithm()
        algorithm._state = 'half_open'
        algorithm._half_open_calls = 1
        algorithm._half_open_successes = 1

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        algorithm.on_request_failure(exception)

        # Should immediately transition back to open
        assert algorithm._state == 'open'
