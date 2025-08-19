"""Tests for minute boundary recovery behavior."""

from unittest.mock import patch

from plsno429.algorithms import AdaptiveAlgorithm, RetryAlgorithm, TokenBucketAlgorithm
from plsno429.utils import calculate_wait_until_next_minute, get_current_minute_boundary


class TestMinuteBoundaryRecovery:
    @patch('time.time')
    def test_calculate_wait_until_next_minute(self, mock_time):
        """Test calculation of wait time until next minute."""
        # Test at 30 seconds into minute
        mock_time.return_value = 1234.5  # 20:34.5
        wait_time = calculate_wait_until_next_minute()
        assert abs(wait_time - 25.5) < 0.1  # 60 - 34.5

        # Test at 59.9 seconds into minute
        mock_time.return_value = 1259.9  # 20:59.9
        wait_time = calculate_wait_until_next_minute()
        assert abs(wait_time - 0.1) < 0.001  # 60 - 59.9 (with floating point tolerance)

        # Test at exact minute boundary
        mock_time.return_value = 1260.0  # 21:00.0
        wait_time = calculate_wait_until_next_minute()
        assert wait_time == 0.0

    @patch('time.time')
    def test_get_current_minute_boundary(self, mock_time):
        """Test getting current minute boundary."""
        mock_time.return_value = 1234.567  # 20:34.567
        boundary = get_current_minute_boundary()
        assert boundary == 1200.0  # 20:00.0 (start of minute 20)

    @patch('time.time')
    def test_tpm_limit_triggers_minute_boundary_wait(self, mock_time):
        """Test that TPM limit triggers minute boundary wait."""
        mock_time.return_value = 1234.5  # 20:34.5

        algorithm = RetryAlgorithm(tpm_limit=1000, safety_margin=0.9)

        # Add tokens to exceed limit
        algorithm._add_token_usage(950)  # Over 900 limit

        # Should return wait time until next minute
        result = algorithm.should_throttle(estimated_tokens=50)
        assert abs(result - 25.5) < 0.1  # Time until next minute

    @patch('time.time')
    def test_minute_boundary_resets_tpm_tracking(self, mock_time):
        """Test that TPM tracking effectively resets at minute boundaries."""
        algorithm = RetryAlgorithm(tpm_limit=1000, safety_margin=0.9)

        # Start at beginning of minute 20
        mock_time.return_value = 1200.0  # 20:00.0
        int(1200.0 // 60)  # minute 20

        # Add tokens to near limit
        algorithm._add_token_usage(800)
        assert algorithm._get_current_tpm_usage() == 800

        # Move to next minute
        mock_time.return_value = 1260.0  # 21:00.0
        int(1260.0 // 60)  # minute 21

        # Usage in previous minute shouldn't affect new minute
        assert algorithm._get_current_tpm_usage() == 0

        # Can add tokens again
        result = algorithm.should_throttle(estimated_tokens=500)
        assert result is None

    @patch('time.time')
    def test_multiple_algorithms_minute_boundary_behavior(self, mock_time):
        """Test minute boundary behavior across different algorithms."""
        algorithms = [
            RetryAlgorithm(tpm_limit=1000, safety_margin=0.9),
            TokenBucketAlgorithm(tpm_limit=1000, safety_margin=0.9),
            AdaptiveAlgorithm(tpm_limit=1000, safety_margin=0.9),
        ]

        mock_time.return_value = 1234.5  # 20:34.5

        for algorithm in algorithms:
            # Add tokens to exceed limit
            algorithm._add_token_usage(950)

            # All should wait until minute boundary
            result = algorithm.should_throttle(estimated_tokens=50)
            assert abs(result - 25.5) < 0.1, f'Algorithm {type(algorithm).__name__} failed'

    @patch('time.time')
    def test_model_specific_minute_boundary_recovery(self, mock_time):
        """Test minute boundary recovery with model-specific limits."""
        mock_time.return_value = 1234.5  # 20:34.5

        model_limits = {'gpt-4': 500, 'gpt-3.5-turbo': 1000}
        algorithm = RetryAlgorithm(tpm_limit=1500, model_limits=model_limits, safety_margin=0.9)

        # Exceed gpt-4 model limit
        algorithm._add_token_usage(450, 'gpt-4')  # Over 450 limit (500 * 0.9)

        # Should wait for minute boundary for gpt-4
        result = algorithm.should_throttle(estimated_tokens=50, model='gpt-4')
        assert abs(result - 25.5) < 0.1

        # But gpt-3.5-turbo should be fine
        result = algorithm.should_throttle(estimated_tokens=50, model='gpt-3.5-turbo')
        assert result is None

    @patch('time.time')
    def test_cleanup_occurs_at_minute_boundaries(self, mock_time):
        """Test that cleanup occurs properly at minute boundaries."""
        algorithm = RetryAlgorithm(tpm_limit=1000)

        # Start at minute 20
        mock_time.return_value = 1200.0  # 20:00.0
        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._add_token_usage(200, 'gpt-3.5-turbo')

        # Move to minute 22 (2 minutes later)
        mock_time.return_value = 1320.0  # 22:00.0
        algorithm._last_cleanup = 1200.0  # Force cleanup

        # Trigger cleanup by checking usage
        algorithm._get_current_tpm_usage()

        # Old data (older than 2 minutes) should be cleaned up
        # Current minute (22) should have no usage yet
        assert algorithm._get_current_tpm_usage() == 0
        assert algorithm._get_current_tpm_usage('gpt-4') == 0

    @patch('time.time')
    def test_precise_minute_boundary_timing(self, mock_time):
        """Test precise timing behavior at minute boundaries."""
        algorithm = RetryAlgorithm(tpm_limit=100, safety_margin=1.0)  # Exact limit

        # At 59.9 seconds into minute
        mock_time.return_value = 1259.9
        algorithm._add_token_usage(100)  # At exact limit

        # Should wait 0.1 seconds for next minute
        result = algorithm.should_throttle(estimated_tokens=1)
        assert abs(result - 0.1) < 0.01

        # At exact minute boundary
        mock_time.return_value = 1260.0
        result = algorithm.should_throttle(estimated_tokens=50)
        assert result is None  # Should not throttle anymore

    @patch('time.time')
    def test_minute_boundary_with_multiple_models_interleaved(self, mock_time):
        """Test minute boundary behavior with multiple models used interleaved."""
        model_limits = {'gpt-4': 500, 'gpt-3.5-turbo': 800}
        algorithm = RetryAlgorithm(tpm_limit=2000, model_limits=model_limits, safety_margin=0.9)

        # Minute 20: Add usage for both models
        mock_time.return_value = 1200.0
        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._add_token_usage(400, 'gpt-3.5-turbo')

        # Still within minute 20 - check limits
        mock_time.return_value = 1230.0  # 20:30

        # GPT-4 should be close to limit (300/450)
        result = algorithm.should_throttle(estimated_tokens=200, model='gpt-4')
        assert result is not None  # Should throttle

        # Move to minute 21
        mock_time.return_value = 1260.0

        # Both models should be available again
        result = algorithm.should_throttle(estimated_tokens=200, model='gpt-4')
        assert result is None

        result = algorithm.should_throttle(estimated_tokens=200, model='gpt-3.5-turbo')
        assert result is None

    @patch('time.time')
    def test_total_limit_vs_model_limit_minute_boundary(self, mock_time):
        """Test minute boundary when both total and model limits are involved."""
        model_limits = {'gpt-4': 2000}  # High model limit
        algorithm = RetryAlgorithm(
            tpm_limit=500,  # Low total limit
            model_limits=model_limits,
            safety_margin=0.9,
        )

        mock_time.return_value = 1234.5  # 20:34.5

        # Add tokens under model limit but over total limit
        algorithm._add_token_usage(400, 'gpt-4')  # Under 1800 model limit, over 450 total limit

        # Should throttle due to total limit, not model limit
        result = algorithm.should_throttle(estimated_tokens=100, model='gpt-4')
        assert abs(result - 25.5) < 0.1  # Wait for minute boundary

    @patch('time.time')
    def test_gradual_approach_to_minute_boundary(self, mock_time):
        """Test behavior as we gradually approach minute boundary."""
        algorithm = RetryAlgorithm(tpm_limit=100, safety_margin=1.0)

        # Fill up tokens gradually
        mock_time.return_value = 1200.0  # Start of minute
        algorithm._add_token_usage(90)

        # At 45 seconds - should wait 15 seconds for next minute
        mock_time.return_value = 1245.0
        result = algorithm.should_throttle(estimated_tokens=20)
        assert abs(result - 15.0) < 0.1

        # At 50 seconds - should wait 10 seconds
        mock_time.return_value = 1250.0
        result = algorithm.should_throttle(estimated_tokens=20)
        assert abs(result - 10.0) < 0.1

        # At 58 seconds - should wait 2 seconds
        mock_time.return_value = 1258.0
        result = algorithm.should_throttle(estimated_tokens=20)
        assert abs(result - 2.0) < 0.1
