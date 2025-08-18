"""Tests for requests decorator."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from plsno429 import throttle_requests
from plsno429.exceptions import ConfigurationError, RateLimitExceeded


class TestThrottleRequests:
    def test_successful_request(self):
        @throttle_requests()
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'

    def test_rate_limit_error_with_retry(self):
        call_count = 0
        
        @throttle_requests(base_delay=0.01, jitter=False)
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

    def test_rate_limit_error_max_retries_exceeded(self):
        @throttle_requests(max_retries=1, base_delay=0.01, jitter=False)
        def mock_request():
            exception = Exception('Rate limit exceeded')
            exception.status_code = 429
            raise exception
        
        with pytest.raises(Exception):
            mock_request()

    def test_non_rate_limit_error_not_retried(self):
        call_count = 0
        
        @throttle_requests()
        def mock_request():
            nonlocal call_count
            call_count += 1
            raise ValueError('Some other error')
        
        with pytest.raises(ValueError):
            mock_request()
        
        assert call_count == 1

    def test_token_tracking_with_openai_response(self):
        @throttle_requests()
        def mock_request():
            response = Mock()
            response.usage = Mock()
            response.usage.total_tokens = 150
            return response
        
        result = mock_request()
        assert result.usage.total_tokens == 150

    def test_custom_token_estimate_function(self):
        def estimate_tokens(*args, **kwargs):
            return 100
        
        @throttle_requests(token_estimate_func=estimate_tokens)
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'

    def test_tpm_limit_throttling(self):
        # This test is simplified - just check that the decorator works with TPM settings
        @throttle_requests(tpm_limit=100, safety_margin=0.5)
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'

    def test_invalid_algorithm(self):
        with pytest.raises(ConfigurationError):
            @throttle_requests(algorithm='invalid')
            def mock_request():
                return 'success'

    @pytest.mark.asyncio
    async def test_async_function(self):
        @throttle_requests()
        async def mock_async_request():
            return 'async_success'
        
        result = await mock_async_request()
        assert result == 'async_success'

    @pytest.mark.asyncio
    async def test_async_rate_limit_error_with_retry(self):
        call_count = 0
        
        @throttle_requests(base_delay=0.01, jitter=False)
        async def mock_async_request():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exception = Exception('Rate limit exceeded')
                exception.status_code = 429
                raise exception
            return 'async_success'
        
        result = await mock_async_request()
        assert result == 'async_success'
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_max_retries_exceeded(self):
        @throttle_requests(max_retries=1, base_delay=0.01, jitter=False)
        async def mock_async_request():
            exception = Exception('Rate limit exceeded')
            exception.status_code = 429
            raise exception
        
        with pytest.raises(Exception):
            await mock_async_request()

    def test_retry_after_header_respected(self):
        with patch('plsno429.decorators.time.sleep') as mock_sleep:
            @throttle_requests(jitter=False)
            def mock_request():
                exception = Exception('Rate limit exceeded')
                exception.status_code = 429
                exception.response = Mock()
                exception.response.headers = {'Retry-After': '5'}
                raise exception
            
            with pytest.raises(Exception):
                mock_request()
            
            # Should have slept for 5 seconds as specified in Retry-After
            mock_sleep.assert_called_with(5.0)

    def test_algorithm_configuration_passed_through(self):
        @throttle_requests(
            algorithm='retry',
            max_retries=5,
            base_delay=2.0,
            tpm_limit=50000
        )
        def mock_request():
            return 'success'
        
        result = mock_request()
        assert result == 'success'