"""Tests for retry algorithm."""

from unittest.mock import Mock, patch

import pytest

from plsno429.algorithms import RetryAlgorithm
from plsno429.exceptions import ConfigurationError, RateLimitExceeded


class TestRetryAlgorithm:
    def test_initialization_with_defaults(self):
        algorithm = RetryAlgorithm()

        assert algorithm.max_retries == 3
        assert algorithm.base_delay == 1.0
        assert algorithm.max_delay == 60.0
        assert algorithm.backoff_multiplier == 2.0
        assert algorithm.tpm_limit == 90000

    def test_initialization_with_custom_values(self):
        algorithm = RetryAlgorithm(
            max_retries=5, base_delay=2.0, max_delay=120.0, backoff_multiplier=1.5, tpm_limit=50000
        )

        assert algorithm.max_retries == 5
        assert algorithm.base_delay == 2.0
        assert algorithm.max_delay == 120.0
        assert algorithm.backoff_multiplier == 1.5
        assert algorithm.tpm_limit == 50000

    def test_invalid_configuration(self):
        with pytest.raises(ConfigurationError):
            RetryAlgorithm(max_retries=-1)

        with pytest.raises(ConfigurationError):
            RetryAlgorithm(base_delay=0)

        with pytest.raises(ConfigurationError):
            RetryAlgorithm(max_delay=0)

        with pytest.raises(ConfigurationError):
            RetryAlgorithm(backoff_multiplier=0)

    def test_should_throttle_under_tpm_limit(self):
        algorithm = RetryAlgorithm(tpm_limit=1000)

        result = algorithm.should_throttle(estimated_tokens=100)
        assert result is None

    @patch('plsno429.base.time.time')
    def test_should_throttle_over_tpm_limit(self, mock_time):
        mock_time.return_value = 60.0  # Start of minute

        algorithm = RetryAlgorithm(tpm_limit=1000, safety_margin=0.9)

        # Add tokens to exceed 90% of limit (900 tokens)
        algorithm._add_token_usage(950)

        result = algorithm.should_throttle(estimated_tokens=1)
        assert result is not None
        assert result > 0

    def test_on_request_success_resets_retry_count(self):
        algorithm = RetryAlgorithm()
        algorithm._retry_count = 2

        algorithm.on_request_success(tokens_used=100)

        assert algorithm._retry_count == 0

    def test_on_request_success_tracks_tokens(self):
        algorithm = RetryAlgorithm()

        with patch.object(algorithm, '_add_token_usage') as mock_add:
            algorithm.on_request_success(tokens_used=100)
            mock_add.assert_called_once_with(100)

    def test_on_request_failure_non_rate_limit_error(self):
        algorithm = RetryAlgorithm()
        exception = Exception('Connection error')

        result = algorithm.on_request_failure(exception)

        assert result is None

    def test_on_request_failure_rate_limit_error_first_retry(self):
        algorithm = RetryAlgorithm(base_delay=2.0, jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = None  # No response object

        result = algorithm.on_request_failure(exception)

        assert result == 2.0  # base_delay
        assert algorithm._retry_count == 1

    def test_on_request_failure_exponential_backoff(self):
        algorithm = RetryAlgorithm(base_delay=1.0, backoff_multiplier=2.0, jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = None  # No response object

        # First retry
        result1 = algorithm.on_request_failure(exception)
        assert result1 == 1.0

        # Second retry
        result2 = algorithm.on_request_failure(exception)
        assert result2 == 2.0

        # Third retry
        result3 = algorithm.on_request_failure(exception)
        assert result3 == 4.0

    def test_on_request_failure_max_delay_cap(self):
        algorithm = RetryAlgorithm(
            base_delay=10.0, backoff_multiplier=10.0, max_delay=30.0, jitter=False
        )
        exception = Mock()
        exception.status_code = 429
        exception.response = None  # No response object

        # First retry: 10.0
        algorithm.on_request_failure(exception)

        # Second retry: would be 100.0, capped to 30.0
        result = algorithm.on_request_failure(exception)
        assert result == 30.0

    def test_on_request_failure_max_retries_exceeded(self):
        algorithm = RetryAlgorithm(max_retries=2)
        exception = Mock()
        exception.status_code = 429
        exception.response = None  # No response object

        # First retry
        result1 = algorithm.on_request_failure(exception)
        assert result1 is not None

        # Second retry
        result2 = algorithm.on_request_failure(exception)
        assert result2 is not None

        # Third attempt - should return None (no more retries)
        result3 = algorithm.on_request_failure(exception)
        assert result3 is None

    def test_on_request_failure_with_retry_after_header(self):
        algorithm = RetryAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '15'}

        result = algorithm.on_request_failure(exception)

        assert result == 15.0  # Should use Retry-After value

    def test_reset_retry_count(self):
        algorithm = RetryAlgorithm()
        algorithm._retry_count = 5

        algorithm.reset_retry_count()

        assert algorithm._retry_count == 0

    def test_max_wait_time_exceeded(self):
        algorithm = RetryAlgorithm(max_wait_minutes=0.01)  # 0.6 seconds
        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '120'}  # 2 minutes

        with pytest.raises(RateLimitExceeded):
            algorithm.on_request_failure(exception)
