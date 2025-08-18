"""Tests for algorithm switching in decorators."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from plsno429 import throttle_requests
from plsno429.exceptions import ConfigurationError


class TestAlgorithmSwitching:
    def test_retry_algorithm_selection(self):
        @throttle_requests(algorithm='retry', max_retries=2, base_delay=1.0)
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'

    def test_token_bucket_algorithm_selection(self):
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=500,
            refill_rate=100.0
        )
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'

    def test_invalid_algorithm_selection(self):
        with pytest.raises(ConfigurationError, match='Unknown algorithm: invalid'):
            @throttle_requests(algorithm='invalid')
            def mock_request():
                return 'success'

    def test_retry_algorithm_with_429_error(self):
        call_count = 0
        
        @throttle_requests(
            algorithm='retry',
            max_retries=2,
            base_delay=0.01,
            jitter=False
        )
        def mock_request():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exception = Exception('Rate limit exceeded')
                exception.status_code = 429
                raise exception
            return 'success'
        
        result = mock_request()
        assert result == 'success'
        assert call_count == 2

    @patch('plsno429.algorithms.time.time')
    def test_token_bucket_algorithm_throttling(self, mock_time):
        mock_time.return_value = 0.0
        
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=100,
            refill_rate=10.0,
            jitter=False
        )
        def mock_request():
            return 'request_made'
        
        # First request should succeed (consume 100 tokens)
        with patch('plsno429.decorators.time.sleep') as mock_sleep:
            result = mock_request()
            assert result == 'request_made'
            mock_sleep.assert_not_called()
        
        # Second request should be throttled (no tokens left)
        with patch('plsno429.decorators.time.sleep') as mock_sleep:
            result = mock_request()
            assert result == 'request_made'
            # Should wait for tokens to refill (100 tokens / 10 tokens per second = 10 seconds)
            mock_sleep.assert_called_once_with(10.0)

    def test_algorithm_parameter_passing(self):
        # Test that algorithm-specific parameters are correctly passed
        @throttle_requests(
            algorithm='retry',
            max_retries=5,
            base_delay=2.0,
            backoff_multiplier=3.0
        )
        def retry_request():
            return 'retry_success'
        
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=2000,
            refill_rate=500.0
        )
        def bucket_request():
            return 'bucket_success'
        
        # Both should work without errors
        assert retry_request() == 'retry_success'
        assert bucket_request() == 'bucket_success'

    def test_mixed_base_and_algorithm_parameters(self):
        @throttle_requests(
            algorithm='token_bucket',
            # Base parameters
            tpm_limit=50000,
            safety_margin=0.8,
            max_wait_minutes=2.0,
            jitter=False,
            # Token bucket specific parameters
            burst_size=1500,
            refill_rate=200.0
        )
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'

    @pytest.mark.asyncio
    async def test_async_retry_algorithm(self):
        call_count = 0
        
        @throttle_requests(
            algorithm='retry',
            max_retries=1,
            base_delay=0.01,
            jitter=False
        )
        async def async_mock_request():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exception = Exception('Rate limit exceeded')
                exception.status_code = 429
                raise exception
            return 'async_success'
        
        result = await async_mock_request()
        assert result == 'async_success'
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_token_bucket_algorithm(self):
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=200,
            refill_rate=50.0
        )
        async def async_mock_request():
            return 'async_bucket_success'
        
        result = await async_mock_request()
        assert result == 'async_bucket_success'

    def test_custom_token_estimate_function_with_token_bucket(self):
        def custom_estimator(*args, **kwargs):
            return 50  # Always estimate 50 tokens
        
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=100,
            token_estimate_func=custom_estimator
        )
        def mock_request():
            return 'success'
        
        # Should be able to make 2 requests (100 tokens / 50 per request)
        assert mock_request() == 'success'
        assert mock_request() == 'success'
        
        # Third request should be throttled
        with patch('plsno429.decorators.time.sleep'):
            assert mock_request() == 'success'

    def test_token_bucket_with_429_error_and_retry_after(self):
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=1000,
            jitter=False
        )
        def mock_request():
            exception = Exception('Rate limit exceeded')
            exception.status_code = 429
            exception.response = Mock()
            exception.response.headers = {'Retry-After': '5'}
            raise exception
        
        with patch('plsno429.decorators.time.sleep') as mock_sleep:
            with pytest.raises(Exception):
                mock_request()
            
            # Should use Retry-After header value
            mock_sleep.assert_called_with(5.0)

    def test_algorithm_specific_configuration_validation(self):
        # Test that algorithm-specific validation works
        with pytest.raises(ConfigurationError):
            @throttle_requests(algorithm='retry', max_retries=-1)
            def mock_request():
                return 'success'
        
        with pytest.raises(ConfigurationError):
            @throttle_requests(algorithm='token_bucket', burst_size=0)
            def mock_request():
                return 'success'

    def test_default_algorithm_is_retry(self):
        # When no algorithm is specified, should default to retry
        @throttle_requests(max_retries=3)
        def mock_request():
            return 'default_success'
        
        result = mock_request()
        assert result == 'default_success'

    def test_algorithm_state_isolation(self):
        # Test that different algorithm instances maintain separate state
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=100,
            refill_rate=10.0
        )
        def request_a():
            return 'a'
        
        @throttle_requests(
            algorithm='token_bucket',
            burst_size=200,
            refill_rate=20.0
        )
        def request_b():
            return 'b'
        
        # Both should work independently
        assert request_a() == 'a'
        assert request_b() == 'b'