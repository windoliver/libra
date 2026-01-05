"""
Token Bucket Rate Limiter for LIBRA.

Prevents order spam and exchange rate limit violations using
the token bucket algorithm.

Performance: <1μs per check (simple arithmetic + monotonic clock).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for order submission.

    Prevents order spam and exchange rate limit violations.

    Algorithm:
    - Bucket holds up to `capacity` tokens
    - Tokens replenish at `rate` per second
    - Each operation consumes tokens (default: 1)
    - If bucket empty, operation is rejected

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded use,
        wrap calls in a lock or use the Rust implementation (Phase 1B).

    Examples:
        # Allow 10 orders/second with burst capacity of 100
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=100)

        if limiter.try_acquire():
            # Proceed with order
            ...
        else:
            # Rate limited, reject or queue

    Reference:
        https://en.wikipedia.org/wiki/Token_bucket
    """

    rate: float  # Tokens per second (replenishment rate)
    capacity: int  # Maximum tokens (burst capacity)

    # Internal state (initialized in __post_init__)
    _tokens: float = field(init=False, repr=False)
    _last_refill: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize bucket to full capacity."""
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from the bucket.

        This is the main method for rate limiting. Call before each
        operation that should be rate limited.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens acquired successfully, False if rate limited

        Performance:
            <1μs (simple arithmetic, no allocations)
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """
        Refill tokens based on elapsed time.

        Called automatically by try_acquire(). Uses monotonic clock
        for accurate timing regardless of system clock changes.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Add tokens based on elapsed time, cap at capacity
        self._tokens = min(self.capacity, self._tokens + (elapsed * self.rate))
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """
        Get current number of available tokens.

        Triggers a refill to ensure accurate count.
        """
        self._refill()
        return self._tokens

    @property
    def is_empty(self) -> bool:
        """Check if bucket has no tokens available."""
        self._refill()
        return self._tokens < 1.0

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def wait_time_for_tokens(self, tokens: int = 1) -> float:
        """
        Calculate time to wait for tokens to become available.

        Useful for implementing backpressure or queuing.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0.0 if tokens already available)
        """
        self._refill()

        if self._tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self._tokens
        return tokens_needed / self.rate


@dataclass
class MultiRateLimiter:
    """
    Combines multiple rate limiters for tiered limits.

    Useful when API has multiple rate limits like:
    - 10 requests per second
    - 100 requests per minute
    - 1000 requests per hour

    All limits must pass for an operation to proceed.

    Examples:
        limiter = MultiRateLimiter([
            TokenBucketRateLimiter(rate=10, capacity=10),   # 10/sec
            TokenBucketRateLimiter(rate=1.67, capacity=100), # 100/min
        ])

        if limiter.try_acquire():
            # Proceed
            ...
    """

    limiters: list[TokenBucketRateLimiter]

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from all limiters.

        Only succeeds if ALL limiters have sufficient tokens.
        If any limiter fails, no tokens are consumed from any limiter.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if all limiters approved, False otherwise
        """
        # First check if all limiters have capacity
        for limiter in self.limiters:
            limiter._refill()
            if limiter._tokens < tokens:
                return False

        # All have capacity, consume from all
        for limiter in self.limiters:
            limiter._tokens -= tokens

        return True

    @property
    def available_tokens(self) -> float:
        """Get minimum available tokens across all limiters."""
        return min(limiter.available_tokens for limiter in self.limiters)

    def reset(self) -> None:
        """Reset all limiters to full capacity."""
        for limiter in self.limiters:
            limiter.reset()
