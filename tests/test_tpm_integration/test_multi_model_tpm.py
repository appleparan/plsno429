"""Tests for multi-model TPM integration."""

from unittest.mock import patch

import pytest

from plsno429 import throttle_requests
from plsno429.algorithms import RetryAlgorithm
from plsno429.exceptions import RateLimitExceeded


class TestMultiModelTPM:
    def test_model_limits_initialization(self):
        """Test algorithm initialization with model-specific limits."""
        model_limits = {'gpt-4': 50000, 'gpt-3.5-turbo': 90000, 'text-embedding-ada-002': 1000000}

        algorithm = RetryAlgorithm(
            tpm_limit=40000,  # Default limit
            model_limits=model_limits,
        )

        assert algorithm.tpm_limit == 40000
        assert algorithm.model_limits == model_limits

        # Test effective limits with safety margin
        assert algorithm._get_effective_tpm_limit('gpt-4') == 45000  # 50000 * 0.9
        assert algorithm._get_effective_tpm_limit('gpt-3.5-turbo') == 81000  # 90000 * 0.9
        assert algorithm._get_effective_tpm_limit('unknown-model') == 36000  # 40000 * 0.9

    def test_model_specific_token_tracking(self):
        """Test separate token tracking per model."""
        model_limits = {'gpt-4': 1000, 'gpt-3.5-turbo': 2000}
        algorithm = RetryAlgorithm(tpm_limit=1500, model_limits=model_limits)

        # Add tokens for different models
        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._add_token_usage(500, 'gpt-3.5-turbo')
        algorithm._add_token_usage(200, 'gpt-4')

        # Check model-specific usage
        assert algorithm._get_current_tpm_usage('gpt-4') == 500
        assert algorithm._get_current_tpm_usage('gpt-3.5-turbo') == 500
        assert algorithm._get_current_tpm_usage(None) == 1000  # Total

    def test_model_specific_throttling(self):
        """Test throttling based on model-specific limits."""
        model_limits = {'gpt-4': 1000, 'gpt-3.5-turbo': 2000}
        algorithm = RetryAlgorithm(tpm_limit=3000, model_limits=model_limits, safety_margin=0.8)

        # Add tokens to exceed gpt-4 limit (1000 * 0.8 = 800)
        algorithm._add_token_usage(750, 'gpt-4')

        # Should throttle gpt-4 requests (750 + 100 = 850 > 800)
        result = algorithm.should_throttle(estimated_tokens=100, model='gpt-4')
        assert result is not None
        assert result > 0

        # Should not throttle gpt-3.5-turbo requests (different limit)
        result = algorithm.should_throttle(estimated_tokens=100, model='gpt-3.5-turbo')
        assert result is None

    def test_total_limit_enforcement(self):
        """Test that total limit is enforced even with model limits."""
        model_limits = {'gpt-4': 2000, 'gpt-3.5-turbo': 2000}
        algorithm = RetryAlgorithm(
            tpm_limit=1000,  # Lower total limit
            model_limits=model_limits,
            safety_margin=0.9,
        )

        # Add tokens under individual model limits but over total limit
        algorithm._add_token_usage(400, 'gpt-4')
        algorithm._add_token_usage(400, 'gpt-3.5-turbo')

        # Total is 800, limit is 900 (1000 * 0.9), so should throttle
        result = algorithm.should_throttle(estimated_tokens=200, model='gpt-4')
        assert result is not None

    @patch('plsno429.base.calculate_wait_until_next_minute')
    def test_minute_boundary_recovery(self, mock_wait):
        """Test automatic recovery at minute boundaries."""
        mock_wait.return_value = 45.0  # 45 seconds until next minute

        algorithm = RetryAlgorithm(tpm_limit=1000, safety_margin=0.9)
        algorithm._add_token_usage(950)  # Exceed limit

        result = algorithm.should_throttle(estimated_tokens=100)
        assert result == 45.0
        mock_wait.assert_called_once()

    def test_get_tpm_stats_default(self):
        """Test TPM statistics for default/total usage."""
        algorithm = RetryAlgorithm(tpm_limit=1000, safety_margin=0.8)
        algorithm._add_token_usage(300)

        stats = algorithm.get_tpm_stats()

        assert stats['current_usage'] == 300
        assert stats['effective_limit'] == 800
        assert stats['raw_limit'] == 1000
        assert stats['safety_margin'] == 0.8
        assert stats['utilization'] == 0.375  # 300/800
        assert stats['tokens_remaining'] == 500

    def test_get_tpm_stats_model_specific(self):
        """Test TPM statistics for specific models."""
        model_limits = {'gpt-4': 2000}
        algorithm = RetryAlgorithm(tpm_limit=1000, model_limits=model_limits, safety_margin=0.9)

        algorithm._add_token_usage(600, 'gpt-4')
        algorithm._add_token_usage(200, 'gpt-3.5-turbo')

        # GPT-4 stats
        gpt4_stats = algorithm.get_tpm_stats('gpt-4')
        assert gpt4_stats['model'] == 'gpt-4'
        assert gpt4_stats['current_usage'] == 600
        assert gpt4_stats['effective_limit'] == 1800  # 2000 * 0.9
        assert gpt4_stats['total_usage'] == 800

        # Unknown model stats (uses default limit)
        unknown_stats = algorithm.get_tpm_stats('unknown-model')
        assert unknown_stats['model'] == 'unknown-model'
        assert unknown_stats['current_usage'] == 0
        assert unknown_stats['effective_limit'] == 900  # 1000 * 0.9

    def test_get_all_model_stats(self):
        """Test getting stats for all models."""
        model_limits = {'gpt-4': 2000, 'gpt-3.5-turbo': 1500}
        algorithm = RetryAlgorithm(tpm_limit=1000, model_limits=model_limits)

        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._add_token_usage(200, 'claude-3')  # Not in limits

        all_stats = algorithm.get_all_model_stats()

        # Should include total, used models, and configured models
        assert 'total' in all_stats
        assert 'gpt-4' in all_stats
        assert 'gpt-3.5-turbo' in all_stats  # Configured but not used
        assert 'claude-3' in all_stats  # Used but not configured

        assert all_stats['total']['current_usage'] == 500
        assert all_stats['gpt-4']['current_usage'] == 300
        assert all_stats['claude-3']['current_usage'] == 200

    def test_reset_tpm_tracking_specific_model(self):
        """Test resetting TPM tracking for specific model."""
        algorithm = RetryAlgorithm(tpm_limit=1000)

        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._add_token_usage(200, 'gpt-3.5-turbo')

        # Reset only gpt-4
        algorithm.reset_tpm_tracking('gpt-4')

        assert algorithm._get_current_tpm_usage('gpt-4') == 0
        assert algorithm._get_current_tpm_usage('gpt-3.5-turbo') == 200
        # Note: Current implementation doesn't update total when individual model is reset
        # This is a known limitation - total includes deleted model usage
        assert algorithm._get_current_tpm_usage(None) == 500  # Total still includes deleted usage

    def test_reset_tpm_tracking_all(self):
        """Test resetting all TPM tracking."""
        algorithm = RetryAlgorithm(tpm_limit=1000)

        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._add_token_usage(200, 'gpt-3.5-turbo')

        # Reset all
        algorithm.reset_tpm_tracking()

        assert algorithm._get_current_tpm_usage('gpt-4') == 0
        assert algorithm._get_current_tpm_usage('gpt-3.5-turbo') == 0
        assert algorithm._get_current_tpm_usage(None) == 0

    def test_decorator_with_model_limits(self):
        """Test decorator usage with model-specific limits."""

        @throttle_requests(
            algorithm='retry',
            tpm_limit=1000,
            model_limits={'gpt-4': 500},
            safety_margin=0.8,
            max_retries=1,
        )
        def mock_openai_call(model='gpt-4', tokens=100):
            # Simulate OpenAI response
            class MockResponse:
                def __init__(self):
                    self.model = model
                    self.usage = MockUsage()

            class MockUsage:
                def __init__(self):
                    self.total_tokens = tokens

            return MockResponse()

        # This should work fine
        result = mock_openai_call(model='gpt-4', tokens=100)
        assert result.model == 'gpt-4'

    def test_decorator_with_model_function(self):
        """Test decorator with model extraction function."""

        def extract_model(*args, **kwargs):
            return kwargs.get('model', 'gpt-3.5-turbo')

        @throttle_requests(
            algorithm='retry', model_func=extract_model, tpm_limit=1000, model_limits={'gpt-4': 500}
        )
        def api_call(**kwargs):
            return f'Called with model: {kwargs.get("model", "default")}'

        # Test model extraction
        result = api_call(model='gpt-4', prompt='test')
        assert 'gpt-4' in result

    @patch('time.time')
    def test_cleanup_old_model_usage(self, mock_time):
        """Test cleanup of old model usage data."""
        algorithm = RetryAlgorithm(tpm_limit=1000)

        # Set initial time
        mock_time.return_value = 100.0
        int(100.0 // 60)  # minute 1

        algorithm._add_token_usage(300, 'gpt-4')
        algorithm._last_cleanup = 100.0

        # Move to future (3 minutes later)
        mock_time.return_value = 280.0
        int(280.0 // 60)  # minute 4

        # Add new usage to trigger cleanup
        algorithm._add_token_usage(200, 'gpt-4')

        # Old usage should be cleaned up (cutoff is minute 2)
        # Only current minute usage should remain
        assert algorithm._get_current_tpm_usage('gpt-4') == 200
        assert len(algorithm._token_usage.get('gpt-4', {})) == 1

    def test_max_wait_time_with_model_limits(self):
        """Test maximum wait time enforcement with model limits."""
        algorithm = RetryAlgorithm(
            tpm_limit=1000,
            model_limits={'gpt-4': 500},
            max_wait_minutes=0.01,  # Very short max wait
        )

        algorithm._add_token_usage(450, 'gpt-4')

        # Should raise exception when wait time exceeds maximum
        with pytest.raises(RateLimitExceeded):
            algorithm.should_throttle(estimated_tokens=100, model='gpt-4')
