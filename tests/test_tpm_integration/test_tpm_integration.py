"""Integration tests for TPM functionality across all algorithms."""

import time
from unittest.mock import Mock, patch

import pytest

from plsno429 import throttle_requests
from plsno429.algorithms import (
    AdaptiveAlgorithm,
    CircuitBreakerAlgorithm,
    RetryAlgorithm,
    SlidingWindowAlgorithm,
    TokenBucketAlgorithm,
)


class TestTPMIntegration:
    @pytest.mark.parametrize("algorithm_class", [
        RetryAlgorithm,
        TokenBucketAlgorithm,
        AdaptiveAlgorithm,
        SlidingWindowAlgorithm,
        CircuitBreakerAlgorithm,
    ])
    def test_all_algorithms_support_model_limits(self, algorithm_class):
        """Test that all algorithms support model-specific limits."""
        model_limits = {
            'gpt-4': 50000,
            'gpt-3.5-turbo': 90000,
            'text-embedding-ada-002': 1000000
        }
        
        algorithm = algorithm_class(
            tpm_limit=40000,
            model_limits=model_limits
        )
        
        assert algorithm.model_limits == model_limits
        assert algorithm._get_effective_tpm_limit('gpt-4') == 45000  # 50000 * 0.9
        assert algorithm._get_effective_tpm_limit('unknown') == 36000  # 40000 * 0.9

    @pytest.mark.parametrize("algorithm_class", [
        RetryAlgorithm,
        TokenBucketAlgorithm,
        AdaptiveAlgorithm,
        SlidingWindowAlgorithm,
        CircuitBreakerAlgorithm,
    ])
    def test_all_algorithms_track_model_specific_usage(self, algorithm_class):
        """Test that all algorithms track usage per model."""
        algorithm = algorithm_class(
            tpm_limit=10000,
            model_limits={'gpt-4': 5000}
        )
        
        # Add usage for different models
        algorithm._add_token_usage(1000, 'gpt-4')
        algorithm._add_token_usage(500, 'gpt-3.5-turbo')
        
        assert algorithm._get_current_tpm_usage('gpt-4') == 1000
        assert algorithm._get_current_tpm_usage('gpt-3.5-turbo') == 500
        assert algorithm._get_current_tpm_usage(None) == 1500

    def test_decorator_with_real_model_extraction(self):
        """Test decorator with realistic model extraction."""
        def extract_model_from_openai_call(*args, **kwargs):
            # Simulate OpenAI client call pattern
            if 'model' in kwargs:
                return kwargs['model']
            if len(args) > 0 and hasattr(args[0], 'model'):
                return args[0].model
            return None
        
        call_count = 0
        
        @throttle_requests(
            algorithm='retry',
            model_func=extract_model_from_openai_call,
            tpm_limit=1000,
            model_limits={'gpt-4': 500},
            max_retries=0  # Don't actually retry
        )
        def openai_chat_completions(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Simulate OpenAI response
            class MockResponse:
                def __init__(self, model, tokens):
                    self.model = model
                    self.usage = MockUsage(tokens)
            
            class MockUsage:
                def __init__(self, tokens):
                    self.total_tokens = tokens
            
            return MockResponse(kwargs.get('model', 'gpt-3.5-turbo'), 150)
        
        # Make calls with different models
        result1 = openai_chat_completions(model='gpt-4', messages=[{'role': 'user', 'content': 'test'}])
        assert result1.model == 'gpt-4'
        
        result2 = openai_chat_completions(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': 'test'}])
        assert result2.model == 'gpt-3.5-turbo'
        
        assert call_count == 2

    @patch('time.time')
    def test_cross_algorithm_model_isolation(self, mock_time):
        """Test that different algorithm instances track models separately."""
        mock_time.return_value = 1000.0
        
        # Create two different algorithms
        retry_algo = RetryAlgorithm(tpm_limit=1000, model_limits={'gpt-4': 500})
        bucket_algo = TokenBucketAlgorithm(tpm_limit=1000, model_limits={'gpt-4': 500})
        
        # Add usage to one algorithm
        retry_algo._add_token_usage(300, 'gpt-4')
        
        # Other algorithm should not be affected
        assert retry_algo._get_current_tpm_usage('gpt-4') == 300
        assert bucket_algo._get_current_tpm_usage('gpt-4') == 0

    def test_decorator_model_auto_detection_from_response(self):
        """Test automatic model detection from response object."""
        @throttle_requests(
            algorithm='retry',
            tpm_limit=1000,
            model_limits={'gpt-4': 500}
        )
        def mock_api_call():
            # Return object that looks like OpenAI response
            class MockResponse:
                def __init__(self):
                    self.model = 'gpt-4'
                    self.usage = MockUsage()
            
            class MockUsage:
                def __init__(self):
                    self.total_tokens = 250
            
            return MockResponse()
        
        result = mock_api_call()
        assert result.model == 'gpt-4'

    def test_mixed_model_usage_patterns(self):
        """Test realistic mixed usage patterns across models."""
        model_limits = {
            'gpt-4': 1000,
            'gpt-3.5-turbo': 2000,
            'text-embedding-ada-002': 10000
        }
        
        algorithm = RetryAlgorithm(
            tpm_limit=5000,
            model_limits=model_limits,
            safety_margin=0.9
        )
        
        # Simulate realistic usage pattern
        algorithm._add_token_usage(400, 'gpt-4')  # Chat completion
        algorithm._add_token_usage(800, 'gpt-3.5-turbo')  # Multiple chat completions
        algorithm._add_token_usage(2000, 'text-embedding-ada-002')  # Embeddings
        
        # Check that each model is tracked correctly
        stats = algorithm.get_all_model_stats()
        
        assert stats['gpt-4']['current_usage'] == 400
        assert stats['gpt-3.5-turbo']['current_usage'] == 800
        assert stats['text-embedding-ada-002']['current_usage'] == 2000
        assert stats['total']['current_usage'] == 3200

    def test_tpm_stats_comprehensive(self):
        """Test comprehensive TPM statistics functionality."""
        model_limits = {'gpt-4': 2000, 'gpt-3.5-turbo': 3000}
        algorithm = RetryAlgorithm(
            tpm_limit=4000,
            model_limits=model_limits,
            safety_margin=0.8
        )
        
        # Add some usage
        algorithm._add_token_usage(600, 'gpt-4')
        algorithm._add_token_usage(1000, 'gpt-3.5-turbo')
        algorithm._add_token_usage(200, 'claude-3')  # Unknown model
        
        # Test individual model stats
        gpt4_stats = algorithm.get_tpm_stats('gpt-4')
        assert gpt4_stats['current_usage'] == 600
        assert gpt4_stats['effective_limit'] == 1600  # 2000 * 0.8
        assert gpt4_stats['utilization'] == 0.375  # 600/1600
        assert gpt4_stats['tokens_remaining'] == 1000
        
        # Test total stats
        total_stats = algorithm.get_tpm_stats()
        assert total_stats['current_usage'] == 1800
        assert total_stats['effective_limit'] == 3200  # 4000 * 0.8
        
        # Test all models stats
        all_stats = algorithm.get_all_model_stats()
        assert len(all_stats) == 5  # total + 4 models (2 configured + 2 used)

    @patch('plsno429.utils.calculate_wait_until_next_minute')
    def test_minute_boundary_recovery_across_algorithms(self, mock_wait):
        """Test minute boundary recovery works across all algorithms."""
        mock_wait.return_value = 30.0
        
        algorithms = [
            RetryAlgorithm(tpm_limit=100, safety_margin=1.0),
            TokenBucketAlgorithm(tpm_limit=100, safety_margin=1.0),
            AdaptiveAlgorithm(tpm_limit=100, safety_margin=1.0),
            SlidingWindowAlgorithm(tpm_limit=100, safety_margin=1.0),
            CircuitBreakerAlgorithm(tpm_limit=100, safety_margin=1.0),
        ]
        
        for algorithm in algorithms:
            # Exceed TPM limit
            algorithm._add_token_usage(100)
            
            # Should return minute boundary wait time
            result = algorithm.should_throttle(estimated_tokens=50)
            assert result == 30.0, f"Failed for {type(algorithm).__name__}"

    def test_error_handling_in_model_extraction(self):
        """Test graceful handling of errors in model extraction."""
        def buggy_model_extractor(*args, **kwargs):
            raise ValueError("Extraction failed")
        
        @throttle_requests(
            algorithm='retry',
            model_func=buggy_model_extractor,
            tpm_limit=1000
        )
        def api_call():
            return "success"
        
        # Should not raise, should fall back gracefully
        result = api_call()
        assert result == "success"

    @patch('time.time')
    def test_cleanup_behavior_with_models(self, mock_time):
        """Test cleanup behavior with model-specific tracking."""
        algorithm = RetryAlgorithm(tpm_limit=1000)
        
        # Start at time 0
        mock_time.return_value = 0.0
        algorithm._add_token_usage(100, 'gpt-4')
        algorithm._add_token_usage(200, 'gpt-3.5-turbo')
        
        # Move forward 3 minutes (beyond cleanup threshold)
        mock_time.return_value = 180.0
        algorithm._last_cleanup = 0.0  # Force cleanup
        
        # Trigger cleanup
        algorithm._get_current_tpm_usage()
        
        # Old data should be cleaned up
        assert algorithm._get_current_tpm_usage('gpt-4') == 0
        assert algorithm._get_current_tpm_usage('gpt-3.5-turbo') == 0

    def test_safety_margin_application_across_models(self):
        """Test that safety margin is applied correctly across different models."""
        model_limits = {
            'gpt-4': 1000,
            'gpt-3.5-turbo': 2000,
        }
        
        algorithm = RetryAlgorithm(
            tpm_limit=3000,
            model_limits=model_limits,
            safety_margin=0.5  # 50% safety margin
        )
        
        # Test effective limits with safety margin
        assert algorithm._get_effective_tpm_limit('gpt-4') == 500  # 1000 * 0.5
        assert algorithm._get_effective_tpm_limit('gpt-3.5-turbo') == 1000  # 2000 * 0.5
        assert algorithm._get_effective_tpm_limit('unknown') == 1500  # 3000 * 0.5
        
        # Test that limits are enforced with safety margin
        algorithm._add_token_usage(450, 'gpt-4')
        
        # Should throttle because 450 + 100 > 500 (effective limit)
        result = algorithm.should_throttle(estimated_tokens=100, model='gpt-4')
        assert result is not None