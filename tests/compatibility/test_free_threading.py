"""
Python 3.13 Free-Threading Compatibility Tests.

Tests GIL status and basic thread safety for LIBRA dependencies.
Run with: pytest tests/compatibility/ -v

For free-threaded Python:
    python3.13t -m pytest tests/compatibility/ -v

See: https://py-free-threading.github.io/
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest


def is_gil_enabled() -> bool:
    """Check if GIL is currently enabled."""
    if hasattr(sys, "_is_gil_enabled"):
        return sys._is_gil_enabled()
    # Pre-3.13 or non-free-threaded build always has GIL
    return True


def is_free_threaded_build() -> bool:
    """Check if running on free-threaded Python build."""
    # Free-threaded builds have sys._is_gil_enabled
    return hasattr(sys, "_is_gil_enabled")


class TestGILStatus:
    """Tests for GIL status verification."""

    def test_python_version(self) -> None:
        """Verify Python version is 3.13+."""
        assert sys.version_info >= (3, 13), (
            f"Python 3.13+ required, got {sys.version_info}"
        )

    def test_gil_status_detection(self) -> None:
        """Test that we can detect GIL status."""
        gil_enabled = is_gil_enabled()
        free_threaded = is_free_threaded_build()

        print(f"\nPython version: {sys.version}")
        print(f"Free-threaded build: {free_threaded}")
        print(f"GIL enabled: {gil_enabled}")

        # This test always passes - it's informational
        assert isinstance(gil_enabled, bool)

    @pytest.mark.skipif(
        not is_free_threaded_build(),
        reason="Only runs on free-threaded Python build"
    )
    def test_gil_disabled_on_free_threaded(self) -> None:
        """On free-threaded build, GIL should be disabled by default."""
        # Note: Importing certain C extensions may re-enable the GIL
        assert not is_gil_enabled(), (
            "GIL is enabled on free-threaded build. "
            "A C extension may have re-enabled it."
        )


class TestBasicThreading:
    """Tests for basic threading operations."""

    def test_thread_creation(self) -> None:
        """Test basic thread creation works."""
        results: list[int] = []
        lock = threading.Lock()

        def worker(n: int) -> None:
            with lock:
                results.append(n)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(10))

    def test_thread_pool_executor(self) -> None:
        """Test ThreadPoolExecutor works correctly."""
        def cpu_work(n: int) -> int:
            total = 0
            for i in range(n):
                total += i * i
            return total

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_work, 10000) for _ in range(8)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 8
        assert all(r > 0 for r in results)

    def test_concurrent_dict_access(self) -> None:
        """Test concurrent dictionary access (common thread safety issue)."""
        shared_dict: dict[str, int] = {}
        lock = threading.Lock()
        errors: list[Exception] = []

        def writer(key_prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    with lock:
                        shared_dict[f"{key_prefix}_{i}"] = i
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"thread_{t}", 100))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(shared_dict) == 400


class TestPolarsThreading:
    """Tests for Polars with threading."""

    def test_polars_import(self) -> None:
        """Test Polars can be imported."""
        import polars as pl

        # Check if importing Polars re-enabled GIL
        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter Polars import, GIL is: {gil_status}")

        df = pl.DataFrame({"a": [1, 2, 3]})
        assert len(df) == 3

    def test_polars_parallel_operations(self) -> None:
        """Test Polars parallel operations."""
        import polars as pl

        # Create large DataFrame
        df = pl.DataFrame({
            "a": range(100_000),
            "b": range(100_000),
        })

        # Operations that Polars parallelizes internally
        result = df.select([
            pl.col("a").sum().alias("sum_a"),
            pl.col("b").mean().alias("mean_b"),
            (pl.col("a") * pl.col("b")).sum().alias("product_sum"),
        ])

        assert result["sum_a"][0] == sum(range(100_000))

    def test_polars_concurrent_dataframe_creation(self) -> None:
        """Test creating DataFrames from multiple threads."""
        import polars as pl

        results: list[pl.DataFrame] = []
        lock = threading.Lock()

        def create_df(n: int) -> None:
            df = pl.DataFrame({"col": range(n)})
            with lock:
                results.append(df)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_df, 1000) for _ in range(8)]
            for f in as_completed(futures):
                f.result()  # Raise any exceptions

        assert len(results) == 8


class TestNumPyThreading:
    """Tests for NumPy with threading (if installed)."""

    @pytest.fixture
    def numpy(self):
        """Import NumPy if available."""
        pytest.importorskip("numpy")
        import numpy as np
        return np

    def test_numpy_import(self, numpy) -> None:
        """Test NumPy can be imported."""
        # Check if importing NumPy re-enabled GIL
        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter NumPy import, GIL is: {gil_status}")

        arr = numpy.array([1, 2, 3])
        assert len(arr) == 3

    def test_numpy_concurrent_reads(self, numpy) -> None:
        """Test concurrent reads from NumPy arrays."""
        shared_array = numpy.random.rand(10000)
        results: list[float] = []
        lock = threading.Lock()

        def read_sum(start: int, end: int) -> None:
            total = numpy.sum(shared_array[start:end])
            with lock:
                results.append(total)

        threads = [
            threading.Thread(target=read_sum, args=(i * 2500, (i + 1) * 2500))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4
        # Sum of parts should equal total
        assert abs(sum(results) - numpy.sum(shared_array)) < 1e-10


class TestAsyncIOStack:
    """Tests for async I/O libraries."""

    def test_aiohttp_import(self) -> None:
        """Test aiohttp import and GIL status."""
        import aiohttp

        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter aiohttp import, GIL is: {gil_status}")
            # aiohttp currently re-enables GIL due to C extensions
            if is_gil_enabled():
                pytest.skip(
                    "aiohttp re-enables GIL on free-threaded build. "
                    "See: https://github.com/aio-libs/aiohttp/issues/8796"
                )

    def test_uvloop_import(self) -> None:
        """Test uvloop import and GIL status."""
        pytest.importorskip("uvloop")
        import uvloop

        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter uvloop import, GIL is: {gil_status}")

    @pytest.mark.asyncio
    async def test_async_basic(self) -> None:
        """Test basic async operations."""
        import asyncio

        async def worker(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * n

        results = await asyncio.gather(*[worker(i) for i in range(10)])
        assert results == [i * i for i in range(10)]


class TestMsgspecThreading:
    """Tests for msgspec with threading."""

    def test_msgspec_import(self) -> None:
        """Test msgspec can be imported."""
        import msgspec

        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter msgspec import, GIL is: {gil_status}")

    def test_msgspec_concurrent_encoding(self) -> None:
        """Test concurrent JSON encoding with msgspec."""
        import msgspec

        data = {"key": "value", "numbers": list(range(100))}
        results: list[bytes] = []
        lock = threading.Lock()

        def encode_many(count: int) -> None:
            for _ in range(count):
                encoded = msgspec.json.encode(data)
                with lock:
                    results.append(encoded)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(encode_many, 100) for _ in range(4)]
            for f in as_completed(futures):
                f.result()

        assert len(results) == 400


class TestOrjsonThreading:
    """Tests for orjson with threading."""

    def test_orjson_import(self) -> None:
        """Test orjson can be imported."""
        import orjson

        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter orjson import, GIL is: {gil_status}")

    def test_orjson_concurrent_encoding(self) -> None:
        """Test concurrent JSON encoding with orjson."""
        import orjson

        data = {"key": "value", "numbers": list(range(100))}
        results: list[bytes] = []
        lock = threading.Lock()

        def encode_many(count: int) -> None:
            for _ in range(count):
                encoded = orjson.dumps(data)
                with lock:
                    results.append(encoded)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(encode_many, 100) for _ in range(4)]
            for f in as_completed(futures):
                f.result()

        assert len(results) == 400


class TestCCXTThreading:
    """Tests for CCXT with threading (if installed)."""

    @pytest.fixture
    def ccxt(self):
        """Import CCXT if available."""
        pytest.importorskip("ccxt")
        import ccxt
        return ccxt

    def test_ccxt_import(self, ccxt) -> None:
        """Test CCXT can be imported."""
        if is_free_threaded_build():
            gil_status = "enabled" if is_gil_enabled() else "disabled"
            print(f"\nAfter CCXT import, GIL is: {gil_status}")

    def test_ccxt_concurrent_exchange_creation(self, ccxt) -> None:
        """Test creating exchange instances from multiple threads."""
        results: list[Any] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def create_exchange() -> None:
            try:
                # Create exchange without connecting
                exchange = ccxt.binance({"enableRateLimit": True})
                with lock:
                    results.append(exchange.id)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_exchange) for _ in range(8)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 8


def generate_compatibility_report() -> str:
    """Generate a compatibility report for all tested libraries."""
    report_lines = [
        "=" * 60,
        "Python 3.13 Free-Threading Compatibility Report",
        "=" * 60,
        f"Python Version: {sys.version}",
        f"Free-threaded build: {is_free_threaded_build()}",
        f"GIL initially enabled: {is_gil_enabled()}",
        "",
        "Library Import Results:",
        "-" * 40,
    ]

    libraries = [
        ("polars", "polars"),
        ("numpy", "numpy"),
        ("aiohttp", "aiohttp"),
        ("uvloop", "uvloop"),
        ("msgspec", "msgspec"),
        ("orjson", "orjson"),
        ("ccxt", "ccxt"),
        ("asyncpg", "asyncpg"),
    ]

    for name, module in libraries:
        try:
            __import__(module)
            gil_after = is_gil_enabled() if is_free_threaded_build() else "N/A"
            report_lines.append(f"  {name}: OK (GIL after import: {gil_after})")
        except ImportError:
            report_lines.append(f"  {name}: NOT INSTALLED")
        except Exception as e:
            report_lines.append(f"  {name}: ERROR - {e}")

    report_lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    print(generate_compatibility_report())
