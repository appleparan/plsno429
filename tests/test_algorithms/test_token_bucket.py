"""Tests for token bucket algorithm."""

import time
from unittest.mock import Mock, patch

import pytest

from plsno429.algorithms import TokenBucketAlgorithm
from plsno429.exceptions import ConfigurationError, RateLimitExceeded


class TestTokenBucketAlgorithm:
    def test_initialization_with_defaults(self):
        algorithm = TokenBucketAlgorithm()
        
        assert algorithm.burst_size == 1000
        assert algorithm.refill_rate == 1500.0
        assert algorithm.tpm_limit == 90000
        assert algorithm._tokens == 1000.0

    def test_initialization_with_custom_values(self):
        algorithm = TokenBucketAlgorithm(
            burst_size=2000,
            refill_rate=500.0,
            tpm_limit=50000
        )
        
        assert algorithm.burst_size == 2000
        assert algorithm.refill_rate == 500.0
        assert algorithm.tpm_limit == 50000
        assert algorithm._tokens == 2000.0

    def test_invalid_configuration(self):
        with pytest.raises(ConfigurationError):
            TokenBucketAlgorithm(burst_size=0)
        
        with pytest.raises(ConfigurationError):
            TokenBucketAlgorithm(refill_rate=0)
        
        with pytest.raises(ConfigurationError):
            TokenBucketAlgorithm(burst_size=-1)
        
        with pytest.raises(ConfigurationError):
            TokenBucketAlgorithm(refill_rate=-1)

    @patch('plsno429.algorithms.time.time')
    def test_refill_tokens(self, mock_time):
        # Start at time 0
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000, refill_rate=100.0)
        
        # Consume some tokens
        algorithm._consume_tokens(500)
        assert algorithm._tokens == 500.0
        
        # Advance time by 2 seconds
        mock_time.return_value = 2.0
        algorithm._refill_tokens()
        
        # Should have added 200 tokens (100 per second * 2 seconds)
        assert algorithm._tokens == 700.0

    @patch('plsno429.algorithms.time.time')
    def test_refill_tokens_capped_at_burst_size(self, mock_time):
        # Start at time 0
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000, refill_rate=100.0)
        
        # Advance time by 20 seconds (would add 2000 tokens)
        mock_time.return_value = 20.0
        algorithm._refill_tokens()
        
        # Should be capped at burst_size
        assert algorithm._tokens == 1000.0

    def test_consume_tokens_success(self):
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        # Should be able to consume tokens
        result = algorithm._consume_tokens(500)
        assert result is True
        assert algorithm._tokens == 500.0

    def test_consume_tokens_insufficient(self):
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        # Try to consume more tokens than available
        result = algorithm._consume_tokens(1500)
        assert result is False
        assert algorithm._tokens == 1000.0  # Should remain unchanged

    @patch('plsno429.algorithms.time.time')
    def test_calculate_wait_time(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000, refill_rate=100.0)
        
        # Consume all tokens
        algorithm._consume_tokens(1000)
        
        # Calculate wait time for 200 tokens
        wait_time = algorithm._calculate_wait_time(200)
        assert wait_time == 2.0  # 200 tokens / 100 tokens per second

    def test_calculate_wait_time_no_wait_needed(self):
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        # Calculate wait time when enough tokens are available
        wait_time = algorithm._calculate_wait_time(500)
        assert wait_time == 0.0

    def test_should_throttle_with_available_tokens(self):
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        result = algorithm.should_throttle(estimated_tokens=500)
        assert result is None
        assert algorithm._tokens == 500.0  # Tokens should be consumed

    @patch('plsno429.algorithms.time.time')
    def test_should_throttle_insufficient_tokens(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000, refill_rate=100.0, jitter=False)
        
        # Try to use more tokens than available
        result = algorithm.should_throttle(estimated_tokens=1500)
        assert result is not None
        assert result == 5.0  # (1500-1000)/100 = 5 seconds

    def test_should_throttle_with_token_estimation(self):
        def custom_estimate(*args, **kwargs):
            return 200
        
        algorithm = TokenBucketAlgorithm(
            burst_size=1000,
            token_estimate_func=custom_estimate
        )
        
        result = algorithm.should_throttle()
        assert result is None
        assert algorithm._tokens == 800.0  # 1000 - 200

    def test_should_throttle_with_tpm_limit(self):
        algorithm = TokenBucketAlgorithm(tpm_limit=1000, safety_margin=0.9)
        
        # Add tokens to exceed TPM limit
        algorithm._add_token_usage(950)
        
        result = algorithm.should_throttle(estimated_tokens=100)
        assert result is not None
        assert result > 0

    def test_on_request_success(self):
        algorithm = TokenBucketAlgorithm()
        
        with patch.object(algorithm, '_add_token_usage') as mock_add:
            algorithm.on_request_success(tokens_used=150)
            mock_add.assert_called_once_with(150)

    def test_on_request_failure_non_rate_limit_error(self):
        algorithm = TokenBucketAlgorithm()
        exception = Exception('Connection error')
        
        result = algorithm.on_request_failure(exception)
        assert result is None

    def test_on_request_failure_with_retry_after_header(self):
        algorithm = TokenBucketAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '10'}
        
        result = algorithm.on_request_failure(exception)
        assert result == 10.0

    @patch('plsno429.algorithms.time.time')
    def test_on_request_failure_with_token_bucket_delay(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(
            burst_size=1000,
            refill_rate=100.0,
            jitter=False
        )
        
        # Consume all tokens
        algorithm._consume_tokens(1000)
        
        exception = Mock()
        exception.status_code = 429
        exception.response = None
        
        result = algorithm.on_request_failure(exception, estimated_tokens=200)
        assert result == 2.0  # 200 tokens / 100 tokens per second

    def test_on_request_failure_default_delay(self):
        algorithm = TokenBucketAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = None
        
        # When bucket has enough tokens, should use default delay
        result = algorithm.on_request_failure(exception, estimated_tokens=100)
        assert result == 1.0  # Default delay

    @patch('plsno429.algorithms.time.time')
    def test_get_tokens_available(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        # Initially full
        assert algorithm.get_tokens_available() == 1000.0
        
        # After consuming some
        algorithm._consume_tokens(300)
        assert algorithm.get_tokens_available() == 700.0

    def test_reset_bucket(self):
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        # Consume some tokens
        algorithm._consume_tokens(500)
        assert algorithm._tokens == 500.0
        
        # Reset bucket
        algorithm.reset_bucket()
        assert algorithm._tokens == 1000.0

    def test_max_wait_time_exceeded(self):
        algorithm = TokenBucketAlgorithm(
            burst_size=10,
            refill_rate=1.0,
            max_wait_minutes=0.01  # 0.6 seconds
        )
        
        # Consume all tokens
        algorithm._consume_tokens(10)
        
        # Try to get many tokens (would require long wait)
        with pytest.raises(RateLimitExceeded):
            algorithm.should_throttle(estimated_tokens=1000)

    @patch('plsno429.algorithms.time.time')
    def test_token_refill_over_time(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000, refill_rate=100.0)
        
        # Consume half the tokens
        algorithm._consume_tokens(500)
        assert algorithm._tokens == 500.0
        
        # Advance time by 1 second
        mock_time.return_value = 1.0
        
        # Should be able to use more tokens now
        result = algorithm.should_throttle(estimated_tokens=200)
        assert result is None
        # Should have: 500 + 100 (refilled) - 200 (consumed) = 400
        assert algorithm._tokens == 400.0

    @patch('plsno429.algorithms.time.time')
    def test_burst_capability(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = TokenBucketAlgorithm(burst_size=1000)
        
        # Should be able to handle burst of requests up to burst_size
        for _ in range(10):
            result = algorithm.should_throttle(estimated_tokens=100)
            assert result is None
        
        # Final token count should be 0
        assert algorithm._tokens == 0.0
        
        # Next request should be throttled
        result = algorithm.should_throttle(estimated_tokens=100)
        assert result is not None