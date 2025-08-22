"""Tests for adaptive algorithm."""

from unittest.mock import Mock, patch

import pytest

from plsno429.algorithms import AdaptiveAlgorithm
from plsno429.exceptions import ConfigurationError, RateLimitExceeded


class TestAdaptiveAlgorithm:
    def test_initialization_with_defaults(self):
        algorithm = AdaptiveAlgorithm()

        assert algorithm.learning_window == 100
        assert algorithm.adaptation_rate == 0.1
        assert algorithm.min_delay == 0.1
        assert algorithm.max_delay == 300.0
        assert algorithm._current_delay == 0.1
        assert algorithm._success_rate == 1.0

    def test_initialization_with_custom_values(self):
        algorithm = AdaptiveAlgorithm(
            learning_window=50, adaptation_rate=0.2, min_delay=0.5, max_delay=120.0
        )

        assert algorithm.learning_window == 50
        assert algorithm.adaptation_rate == 0.2
        assert algorithm.min_delay == 0.5
        assert algorithm.max_delay == 120.0
        assert algorithm._current_delay == 0.5

    def test_invalid_configuration(self):
        with pytest.raises(ConfigurationError):
            AdaptiveAlgorithm(learning_window=0)

        with pytest.raises(ConfigurationError):
            AdaptiveAlgorithm(adaptation_rate=-0.1)

        with pytest.raises(ConfigurationError):
            AdaptiveAlgorithm(adaptation_rate=1.1)

        with pytest.raises(ConfigurationError):
            AdaptiveAlgorithm(min_delay=-1)

        with pytest.raises(ConfigurationError):
            AdaptiveAlgorithm(min_delay=10, max_delay=5)

    @patch('plsno429.algorithms.time.time')
    def test_record_request_successful(self, mock_time):
        mock_time.return_value = 100.0
        algorithm = AdaptiveAlgorithm()

        algorithm._record_request(success=True, delay_used=1.0, tokens=150)

        assert len(algorithm._request_history) == 1
        request = algorithm._request_history[0]
        assert request['timestamp'] == 100.0
        assert request['success'] is True
        assert request['delay_used'] == 1.0
        assert request['tokens'] == 150

    def test_record_request_updates_success_rate(self):
        algorithm = AdaptiveAlgorithm()

        # Add enough requests to trigger success rate calculation
        for i in range(15):
            success = i < 12  # 12 successes out of 15
            algorithm._record_request(success=success)

        # Should calculate success rate from last 20 requests (or all if less than 20)
        expected_rate = 12 / 15
        assert abs(algorithm._success_rate - expected_rate) < 0.01

    @patch('plsno429.algorithms.time.time')
    def test_analyze_patterns_insufficient_data(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = AdaptiveAlgorithm(min_delay=1.0)

        # Add only a few requests
        for _ in range(5):
            algorithm._record_request(success=True, delay_used=2.0)

        suggested_delay = algorithm._analyze_patterns()
        assert suggested_delay == algorithm._current_delay

    @patch('plsno429.algorithms.time.time')
    def test_analyze_patterns_with_hour_data(self, mock_time):
        # Set time to hour 10 (36000 seconds)
        mock_time.return_value = 36000.0
        algorithm = AdaptiveAlgorithm()

        # Add requests for current hour
        for _ in range(15):
            algorithm._record_request(success=True, delay_used=2.0)

        suggested_delay = algorithm._analyze_patterns()
        # Median delay is 2.0, but gets adjusted down by 0.8 due to high success rate (1.0 > 0.95)
        assert suggested_delay == 1.6  # 2.0 * 0.8

    def test_analyze_patterns_adjusts_for_success_rate(self):
        algorithm = AdaptiveAlgorithm()
        algorithm._current_delay = 10.0

        # Add enough request history for analysis
        for _ in range(15):
            algorithm._record_request(success=True, delay_used=10.0)

        # Low success rate should increase delay
        algorithm._success_rate = 0.7
        suggested_delay = algorithm._analyze_patterns()
        assert suggested_delay > 10.0  # Should be 10.0 * 1.5 = 15.0

        # High success rate should decrease delay
        algorithm._success_rate = 0.97
        suggested_delay = algorithm._analyze_patterns()
        assert suggested_delay < 10.0  # Should be 10.0 * 0.8 = 8.0

    def test_analyze_patterns_considers_consecutive_429s(self):
        algorithm = AdaptiveAlgorithm()
        algorithm._current_delay = 5.0
        algorithm._consecutive_429s = 4

        # Add enough request history for analysis
        for _ in range(15):
            algorithm._record_request(success=True, delay_used=5.0)

        suggested_delay = algorithm._analyze_patterns()
        # Should increase delay due to consecutive 429s: 5.0 * 1.2^4 ≈ 10.37
        assert suggested_delay > 5.0

    def test_update_delay_exponential_moving_average(self):
        algorithm = AdaptiveAlgorithm(adaptation_rate=0.2, min_delay=1.0)
        algorithm._current_delay = 10.0

        target_delay = 20.0
        algorithm._update_delay(target_delay)

        # Should be weighted average: 0.8 * 10 + 0.2 * 20 = 12
        assert algorithm._current_delay == 12.0

    def test_update_delay_respects_bounds(self):
        algorithm = AdaptiveAlgorithm(min_delay=5.0, max_delay=15.0, adaptation_rate=1.0)

        # Test minimum bound
        algorithm._update_delay(1.0)
        assert algorithm._current_delay == 5.0

        # Test maximum bound
        algorithm._update_delay(20.0)
        assert algorithm._current_delay == 15.0

    @patch('plsno429.algorithms.time.time')
    def test_should_throttle_no_delay_needed(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = AdaptiveAlgorithm(min_delay=1.0)
        algorithm._last_request_time = 0.0

        # First call sets _last_request_time
        mock_time.return_value = 2.0  # 2 seconds later
        result = algorithm.should_throttle()
        assert result is None

    @patch('plsno429.algorithms.time.time')
    def test_should_throttle_delay_needed(self, mock_time):
        mock_time.return_value = 0.0
        algorithm = AdaptiveAlgorithm(min_delay=5.0, jitter=False)

        # Set last request time
        algorithm._last_request_time = 0.0
        algorithm._last_cleanup = 0.0  # Initialize to avoid type error

        # Try to make request too soon
        mock_time.return_value = 2.0  # Only 2 seconds later, need 5
        result = algorithm.should_throttle()
        assert result == 3.0  # Should wait 3 more seconds

    def test_should_throttle_with_tpm_limit(self):
        algorithm = AdaptiveAlgorithm(tpm_limit=1000, safety_margin=0.9)

        # Add tokens to exceed TPM limit
        algorithm._add_token_usage(950)

        result = algorithm.should_throttle(estimated_tokens=100)
        assert result is not None
        assert result > 0

    def test_on_request_success(self):
        algorithm = AdaptiveAlgorithm()
        algorithm._consecutive_429s = 3

        algorithm.on_request_success(tokens_used=150)

        # Should reset consecutive 429 counter
        assert algorithm._consecutive_429s == 0

        # Should record the request
        assert len(algorithm._request_history) == 1
        assert algorithm._request_history[0]['success'] is True

    def test_on_request_failure_non_rate_limit_error(self):
        algorithm = AdaptiveAlgorithm()
        exception = Exception('Connection error')

        result = algorithm.on_request_failure(exception)
        assert result is None

        # Should still record the failure
        assert len(algorithm._request_history) == 1
        assert algorithm._request_history[0]['success'] is False

    def test_on_request_failure_rate_limit_with_retry_after(self):
        algorithm = AdaptiveAlgorithm(jitter=False)
        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '10'}

        result = algorithm.on_request_failure(exception)
        assert result == 10.0
        assert algorithm._consecutive_429s == 1

    def test_on_request_failure_rate_limit_adaptive_delay(self):
        algorithm = AdaptiveAlgorithm(jitter=False, min_delay=2.0)
        algorithm._consecutive_429s = 0

        exception = Mock()
        exception.status_code = 429
        exception.response = None

        result = algorithm.on_request_failure(exception)

        # Should use adaptive delay with multiplier for 429
        assert result >= 2.0  # At least min_delay
        assert algorithm._consecutive_429s == 1

    def test_on_request_failure_escalating_delays(self):
        algorithm = AdaptiveAlgorithm(jitter=False, min_delay=1.0)
        exception = Mock()
        exception.status_code = 429
        exception.response = None

        # First failure
        result1 = algorithm.on_request_failure(exception)

        # Second failure should have higher delay
        result2 = algorithm.on_request_failure(exception)
        assert result2 > result1

    def test_get_learning_stats(self):
        algorithm = AdaptiveAlgorithm()
        algorithm._success_rate = 0.85
        algorithm._current_delay = 5.0
        algorithm._consecutive_429s = 2

        stats = algorithm.get_learning_stats()

        assert stats['success_rate'] == 0.85
        assert stats['current_delay'] == 5.0
        assert stats['consecutive_429s'] == 2
        assert stats['requests_analyzed'] == 0
        assert stats['learning_window'] == 100

    def test_reset_learning(self):
        algorithm = AdaptiveAlgorithm(min_delay=2.0)

        # Set some state
        algorithm._record_request(success=True)
        algorithm._success_rate = 0.5
        algorithm._current_delay = 10.0
        algorithm._consecutive_429s = 3
        algorithm._last_request_time = 100.0

        algorithm.reset_learning()

        # Should reset to initial state
        assert len(algorithm._request_history) == 0
        assert algorithm._success_rate == 1.0
        assert algorithm._current_delay == 2.0  # min_delay
        assert algorithm._consecutive_429s == 0
        assert algorithm._last_request_time == 0.0

    def test_max_wait_time_exceeded(self):
        algorithm = AdaptiveAlgorithm(
            min_delay=1.0,
            max_delay=10.0,
            max_wait_minutes=0.01,  # 0.6 seconds
        )

        exception = Mock()
        exception.status_code = 429
        exception.response = Mock()
        exception.response.headers = {'Retry-After': '120'}  # 2 minutes

        with pytest.raises(RateLimitExceeded):
            algorithm.on_request_failure(exception)

    @patch('plsno429.algorithms.time.time')
    def test_learning_from_time_patterns(self, mock_time):
        algorithm = AdaptiveAlgorithm()

        # Simulate requests at different hours with different success patterns
        base_time = 0

        # Hour 0: successful with 1.0 delay
        mock_time.return_value = base_time
        for _ in range(15):
            algorithm._record_request(success=True, delay_used=1.0)

        # Hour 12: successful with 5.0 delay
        mock_time.return_value = base_time + 12 * 3600
        for _ in range(15):
            algorithm._record_request(success=True, delay_used=5.0)

        # Test at hour 0 - should suggest lower delay
        mock_time.return_value = base_time + 24 * 3600  # Next day, hour 0
        suggested_delay = algorithm._analyze_patterns()
        assert suggested_delay <= 2.0

        # Test at hour 12 - should suggest higher delay
        mock_time.return_value = base_time + 24 * 3600 + 12 * 3600  # Next day, hour 12
        suggested_delay = algorithm._analyze_patterns()
        assert suggested_delay >= 4.0
