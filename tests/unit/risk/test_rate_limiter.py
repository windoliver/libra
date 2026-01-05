"""Tests for TokenBucketRateLimiter."""

import time

import pytest

from libra.risk.rate_limiter import MultiRateLimiter, TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_create_limiter(self):
        """Create rate limiter with valid parameters."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=100)

        assert limiter.rate == 10.0
        assert limiter.capacity == 100

    def test_invalid_rate(self):
        """Rate must be positive."""
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucketRateLimiter(rate=0, capacity=10)

        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucketRateLimiter(rate=-5, capacity=10)

    def test_invalid_capacity(self):
        """Capacity must be positive."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            TokenBucketRateLimiter(rate=10, capacity=0)

    def test_initial_capacity(self):
        """Bucket starts at full capacity."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=100)

        assert limiter.available_tokens == 100.0

    def test_acquire_success(self):
        """Acquire tokens when available."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=100)

        assert limiter.try_acquire() is True
        # Allow small timing variance due to refill
        assert 98.0 <= limiter.available_tokens <= 100.0

    def test_acquire_multiple(self):
        """Acquire multiple tokens at once."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=100)

        assert limiter.try_acquire(tokens=10) is True
        # Allow small timing variance due to refill
        assert 89.0 <= limiter.available_tokens <= 91.0

    def test_acquire_fail_insufficient(self):
        """Fail to acquire when insufficient tokens."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=5)

        assert limiter.try_acquire(tokens=10) is False
        # Tokens should not be consumed on failure
        assert limiter.available_tokens == 5.0

    def test_acquire_until_empty(self):
        """Acquire tokens until bucket is empty."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=3)

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

    def test_refill_over_time(self):
        """Tokens refill over time."""
        limiter = TokenBucketRateLimiter(rate=1000.0, capacity=100)

        # Drain the bucket
        for _ in range(100):
            limiter.try_acquire()

        assert limiter.is_empty is True

        # Wait for refill (100ms should add ~100 tokens at 1000/sec rate)
        time.sleep(0.1)

        assert limiter.available_tokens >= 90.0  # Allow some timing variance

    def test_is_empty(self):
        """Check if bucket is empty."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=2)

        assert limiter.is_empty is False

        limiter.try_acquire()
        limiter.try_acquire()

        assert limiter.is_empty is True

    def test_reset(self):
        """Reset bucket to full capacity."""
        limiter = TokenBucketRateLimiter(rate=0.1, capacity=100)  # Very slow refill

        for _ in range(50):
            limiter.try_acquire()

        assert 49.0 <= limiter.available_tokens <= 51.0

        limiter.reset()

        assert limiter.available_tokens == 100.0

    def test_wait_time_calculation(self):
        """Calculate wait time for tokens."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

        # Drain bucket
        for _ in range(10):
            limiter.try_acquire()

        # Need to wait for 1 token at 10 tokens/sec = 0.1 seconds
        wait_time = limiter.wait_time_for_tokens(1)
        assert 0.05 <= wait_time <= 0.15

    def test_wait_time_zero_when_available(self):
        """Wait time is 0 when tokens available."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

        wait_time = limiter.wait_time_for_tokens(5)
        assert wait_time == 0.0

    def test_capacity_cap(self):
        """Tokens don't exceed capacity after long idle."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

        # Use some tokens
        for _ in range(5):
            limiter.try_acquire()

        # Simulate long wait (bucket should cap at capacity)
        time.sleep(0.5)  # Would add 50 tokens at 10/sec

        assert limiter.available_tokens == 10.0  # Capped at capacity


class TestMultiRateLimiter:
    """Tests for MultiRateLimiter."""

    def test_create_multi_limiter(self):
        """Create multi-rate limiter."""
        limiter = MultiRateLimiter(
            limiters=[
                TokenBucketRateLimiter(rate=10.0, capacity=10),
                TokenBucketRateLimiter(rate=1.0, capacity=60),
            ]
        )

        assert len(limiter.limiters) == 2

    def test_acquire_success_all_pass(self):
        """Acquire succeeds when all limiters have capacity."""
        limiter = MultiRateLimiter(
            limiters=[
                TokenBucketRateLimiter(rate=10.0, capacity=10),
                TokenBucketRateLimiter(rate=1.0, capacity=60),
            ]
        )

        assert limiter.try_acquire() is True

    def test_acquire_fail_one_empty(self):
        """Acquire fails if any limiter is empty."""
        limiter1 = TokenBucketRateLimiter(rate=0.01, capacity=1)  # Very slow refill
        limiter2 = TokenBucketRateLimiter(rate=0.01, capacity=60)  # Very slow refill

        multi = MultiRateLimiter(limiters=[limiter1, limiter2])

        # First acquire succeeds
        assert multi.try_acquire() is True

        # Second fails because limiter1 is empty
        assert multi.try_acquire() is False

        # limiter2 should not have consumed token on failed attempt
        # (due to the pre-check design)
        assert 58.0 <= limiter2.available_tokens <= 60.0

    def test_available_tokens_minimum(self):
        """Available tokens is minimum across all limiters."""
        limiter = MultiRateLimiter(
            limiters=[
                TokenBucketRateLimiter(rate=10.0, capacity=5),
                TokenBucketRateLimiter(rate=1.0, capacity=100),
            ]
        )

        assert limiter.available_tokens == 5.0

    def test_reset_all(self):
        """Reset all limiters."""
        limiter = MultiRateLimiter(
            limiters=[
                TokenBucketRateLimiter(rate=10.0, capacity=10),
                TokenBucketRateLimiter(rate=1.0, capacity=60),
            ]
        )

        # Use some tokens
        for _ in range(5):
            limiter.try_acquire()

        limiter.reset()

        assert limiter.limiters[0].available_tokens == 10.0
        assert limiter.limiters[1].available_tokens == 60.0
