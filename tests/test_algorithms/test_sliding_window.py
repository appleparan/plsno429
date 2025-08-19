"""Tests for sliding window algorithm."""

import time
from unittest.mock import Mock, patch

import pytest

from plsno429.algorithms import SlidingWindowAlgorithm
from plsno429.exceptions import ConfigurationError, RateLimitExceeded


class TestSlidingWindowAlgorithm:
    def test_initialization_with_defaults(self):
        algorithm = SlidingWindowAlgorithm()

        assert algorithm.window_size == 60
        assert algorithm.max_requests == 1500
        assert algorithm.cleanup_interval == 10
        assert len(algorithm._request_times) == 0

    def test_initialization_with_custom_values(self):
        algorithm = SlidingWindowAlgorithm(window_size=120, max_requests=1000, cleanup_interval=5)

        assert algorithm.window_size == 120
        assert algorithm.max_requests == 1000
        assert algorithm.cleanup_interval == 5

    def test_invalid_configuration(self):
        with pytest.raises(ConfigurationError):
            SlidingWindowAlgorithm(window_size=0)

        with pytest.raises(ConfigurationError):
            SlidingWindowAlgorithm(max_requests=0)

        with pytest.raises(ConfigurationError):
            SlidingWindowAlgorithm(cleanup_interval=0)

        with pytest.raises(ConfigurationError):
            SlidingWindowAlgorithm(window_size=-1)

    @patch('plsno429.algorithms.time.time')
    def test_cleanup_old_requests(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=60, cleanup_interval=5)

        # Add some request times
        mock_time.return_value = 100.0
        algorithm._request_times.extend([50.0, 60.0, 70.0, 80.0, 90.0])
        algorithm._last_cleanup = 95.0  # Force cleanup

        # Current time is 100, window is 60s, so cutoff is 40
        mock_time.return_value = 105.0
        algorithm._cleanup_old_requests()

        # Should remove requests before time 45 (105 - 60)
        remaining_times = list(algorithm._request_times)
        assert all(t >= 45.0 for t in remaining_times)
        assert 50.0 not in remaining_times  # Should be removed

    @patch('plsno429.algorithms.time.time')
    def test_cleanup_respects_interval(self, mock_time):
        algorithm = SlidingWindowAlgorithm(cleanup_interval=10)

        mock_time.return_value = 100.0
        algorithm._last_cleanup = 95.0  # Less than interval ago

        # Add old request
        algorithm._request_times.append(10.0)  # Very old

        algorithm._cleanup_old_requests()

        # Should not cleanup yet
        assert 10.0 in algorithm._request_times
        assert algorithm._last_cleanup == 95.0  # Unchanged

    @patch('plsno429.algorithms.time.time')
    def test_get_current_request_count(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=60)

        mock_time.return_value = 100.0
        # Add requests: some old, some current
        algorithm._request_times.extend([30.0, 50.0, 70.0, 90.0, 95.0])
        algorithm._last_cleanup = 0.0  # Force cleanup

        count = algorithm._get_current_request_count()

        # Current time 100, window 60, so cutoff is 40
        # Requests at 50, 70, 90, 95 should remain
        assert count == 4

    @patch('plsno429.algorithms.time.time')
    def test_calculate_wait_time_no_wait(self, mock_time):
        algorithm = SlidingWindowAlgorithm(max_requests=10)

        mock_time.return_value = 100.0
        # Add fewer than max requests
        algorithm._request_times.extend([90.0, 95.0])

        wait_time = algorithm._calculate_wait_time()
        assert wait_time == 0.0

    @patch('plsno429.algorithms.time.time')
    def test_calculate_wait_time_with_wait(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=60, max_requests=3)

        mock_time.return_value = 100.0
        # Add max requests
        algorithm._request_times.extend([50.0, 70.0, 90.0])

        wait_time = algorithm._calculate_wait_time()

        # Oldest request at 50, window 60, so it expires at 110
        # Current time 100, so wait = 110 - 100 = 10
        assert wait_time == 10.0

    @patch('plsno429.algorithms.time.time')
    def test_should_throttle_under_limit(self, mock_time):
        algorithm = SlidingWindowAlgorithm(max_requests=10)

        mock_time.return_value = 100.0
        # Add fewer than max requests
        algorithm._request_times.extend([90.0, 95.0])

        result = algorithm.should_throttle()
        assert result is None

        # Should record this request
        assert len(algorithm._request_times) == 3
        assert algorithm._request_times[-1] == 100.0

    @patch('plsno429.algorithms.time.time')
    def test_should_throttle_over_limit(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=60, max_requests=2, jitter=False)

        mock_time.return_value = 100.0
        # Add max requests
        algorithm._request_times.extend([50.0, 90.0])

        result = algorithm.should_throttle()

        # Should throttle and return wait time
        assert result is not None
        assert result > 0

    def test_should_throttle_with_tpm_limit(self):
        algorithm = SlidingWindowAlgorithm(tpm_limit=1000, safety_margin=0.9)

        # Add tokens to exceed TPM limit
        algorithm._add_token_usage(950)

        result = algorithm.should_throttle(estimated_tokens=100)
        assert result is not None
        assert result > 0

    def test_on_request_success(self):
        algorithm = SlidingWindowAlgorithm()

        with patch.object(algorithm, '_add_token_usage') as mock_add:
            algorithm.on_request_success(tokens_used=200)
            mock_add.assert_called_once_with(200)

    def test_on_request_failure_non_rate_limit_error(self):
        algorithm = SlidingWindowAlgorithm()
        exception = Exception('Connection error')

        result = algorithm.on_request_failure(exception)
        assert result is None

    def test_on_request_failure_with_retry_after_header(self):
        algorithm = SlidingWindowAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '15'}

        result = algorithm.on_request_failure(exception)
        assert result == 15.0

    @patch('plsno429.algorithms.time.time')
    def test_on_request_failure_with_sliding_window_delay(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=60, max_requests=2, jitter=False)

        mock_time.return_value = 100.0
        # Add requests to fill window
        algorithm._request_times.extend([50.0, 90.0])

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        result = algorithm.on_request_failure(exception)

        # Should calculate wait time based on sliding window
        assert result == 10.0  # (50 + 60) - 100 = 10

    def test_on_request_failure_default_delay(self):
        algorithm = SlidingWindowAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = None

        # Window has space, so calculate_wait_time returns 0
        result = algorithm.on_request_failure(exception)
        assert result == 1.0  # Default delay

    def test_get_window_stats(self):
        algorithm = SlidingWindowAlgorithm(max_requests=10)

        # Add some requests
        algorithm._request_times.extend([90.0, 95.0, 98.0])

        with patch.object(algorithm, '_get_current_request_count', return_value=3):
            with patch.object(algorithm, '_calculate_wait_time', return_value=5.0):
                stats = algorithm.get_window_stats()

        assert stats['current_requests'] == 3
        assert stats['max_requests'] == 10
        assert stats['window_size'] == 60  # Default
        assert stats['utilization'] == 0.3  # 3/10
        assert stats['next_available'] == 5.0

    def test_reset_window(self):
        algorithm = SlidingWindowAlgorithm()

        # Add some state
        algorithm._request_times.extend([90.0, 95.0, 98.0])
        algorithm._last_cleanup = 50.0

        with patch('plsno429.algorithms.time.time', return_value=100.0):
            algorithm.reset_window()

        # Should reset state
        assert len(algorithm._request_times) == 0
        assert algorithm._last_cleanup == 100.0

    def test_max_wait_time_exceeded(self):
        algorithm = SlidingWindowAlgorithm(
            window_size=10,
            max_requests=1,
            max_wait_minutes=0.01,  # 0.6 seconds
        )

        # Fill the window
        algorithm._request_times.append(time.time())

        # Should raise exception when wait time exceeds maximum
        with pytest.raises(RateLimitExceeded):
            algorithm.should_throttle()

    @patch('plsno429.algorithms.time.time')
    def test_sliding_window_behavior_over_time(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=10, max_requests=3)

        # Time 0: Add 3 requests (fill window)
        mock_time.return_value = 0.0
        for _ in range(3):
            result = algorithm.should_throttle()
            assert result is None

        # Time 5: Should be throttled (window still full)
        mock_time.return_value = 5.0
        result = algorithm.should_throttle()
        assert result is not None

        # Time 11: First request should have expired, space available
        mock_time.return_value = 11.0
        result = algorithm.should_throttle()
        assert result is None

    @patch('plsno429.algorithms.time.time')
    def test_precise_rate_limiting(self, mock_time):
        algorithm = SlidingWindowAlgorithm(window_size=60, max_requests=100)

        # Fill exactly to the limit
        mock_time.return_value = 100.0
        start_time = 50.0

        # Add 100 requests between time 50-100
        for i in range(100):
            algorithm._request_times.append(start_time + i * 0.5)

        # Should be at exactly the limit
        assert algorithm._get_current_request_count() == 100

        # Next request should be throttled
        result = algorithm.should_throttle()
        assert result is not None

        # But after window slides, should be available
        mock_time.return_value = 111.0  # First request expires at 110
        algorithm._cleanup_old_requests()
        result = algorithm.should_throttle()
        assert result is None
