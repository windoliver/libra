# ADR-008: Python 3.13 Free-Threading Decision

**Status:** Accepted
**Date:** 2026-01-08
**Decision Makers:** Libra Team

## Context

Python 3.13 introduced experimental support for "free-threaded" execution (PEP 703)
where the Global Interpreter Lock (GIL) is disabled, enabling true parallel execution
of Python threads on multi-core CPUs.

Issue #26 required testing whether Python 3.13t (free-threaded build) is suitable
for LIBRA's high-frequency trading requirements before making architectural decisions.

### Key Concerns

1. **Experimental Status**: Free-threading is experimental in 3.13, officially supported in 3.14
2. **C Extension Compatibility**: Many C extensions re-enable the GIL when imported
3. **Performance Overhead**: Single-threaded code runs ~40% slower in 3.13t (~9% in 3.14t)

## Decision

**Use Python 3.13 (standard build with GIL) + Rust for performance-critical paths.**

Do NOT adopt Python 3.13t (free-threaded) for production until Python 3.14+.

## Analysis

### Dependency Compatibility (Tested January 2026)

| Library | Status | Notes |
|---------|--------|-------|
| **NumPy 2.3+** | ✅ Working | Preliminary free-threaded support |
| **Polars** | ⚠️ Build issues | Needs rust-numpy 0.24 upgrade |
| **msgspec** | ✅ Working | Rust-based, GIL-independent |
| **orjson** | ✅ Working | Rust-based, GIL-independent |
| **aiohttp** | ❌ Re-enables GIL | C extensions not yet compatible |
| **uvloop** | ❌ Not supported | No free-threaded wheels available |
| **asyncpg** | ❓ Unknown | Needs testing |
| **CCXT** | ❓ Unknown | Async support via aiohttp |

### Critical Blocker: I/O Stack

LIBRA's architecture relies heavily on async I/O:
- `aiohttp` for HTTP client (exchange APIs)
- `uvloop` for high-performance event loop
- `asyncpg` for database queries

**All three currently re-enable the GIL**, negating any benefit from free-threading.

### Performance Trade-offs

| Metric | Python 3.13 (GIL) | Python 3.13t (no GIL) | Python 3.14t |
|--------|-------------------|----------------------|--------------|
| Single-threaded | Baseline | ~40% slower | ~9% slower |
| Multi-threaded CPU-bound | GIL-limited | 2-5x faster | 2-5x faster |
| Multi-threaded I/O-bound | async works | GIL re-enabled | TBD |

### Why Not Free-Threading Now?

1. **I/O Stack Incompatibility**: aiohttp/uvloop re-enable GIL
2. **40% Single-Thread Overhead**: Unacceptable for latency-sensitive trading
3. **Experimental Status**: Not production-ready
4. **Rust Already Solves This**: Hot paths in Rust bypass GIL entirely

## Implementation

### Current Architecture (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python 3.13 (with GIL)                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   asyncio    │  │   aiohttp    │  │   uvloop     │          │
│  │  (I/O bound) │  │ (HTTP/WS)    │  │ (event loop) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Rust Extensions (via PyO3)                  │   │
│  │  • Order matching engine      • Indicator calculations  │   │
│  │  • Risk calculations          • Hot data structures     │   │
│  │  (GIL released during Rust execution)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │   Polars     │  │   orjson     │  ← Rust-backed, fast      │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Future Path (When Ready)

Re-evaluate free-threading when:
1. Python 3.14 is released (late 2025)
2. aiohttp supports free-threading
3. uvloop has free-threaded wheels
4. Single-thread overhead drops to <10%

### Test Suite

Tests created in `tests/compatibility/`:
- `test_free_threading.py` - Compatibility tests for all dependencies
- `bench_gil_comparison.py` - Performance benchmarks

Run on both standard and free-threaded builds:
```bash
# Standard Python
pytest tests/compatibility/ -v

# Free-threaded Python (when available)
python3.13t -m pytest tests/compatibility/ -v
```

## Consequences

### Positive
- Production-ready today with standard Python
- Predictable performance (no experimental surprises)
- Rust hot paths already bypass GIL
- Clear migration path when ecosystem matures

### Negative
- Cannot exploit multi-core for pure Python CPU-bound code
- Need to maintain Rust extensions for performance

### Neutral
- CI should test both Python builds when 3.14 is released
- Re-evaluate decision every 6 months

## References

- [PEP 703 - Making the Global Interpreter Lock Optional](https://peps.python.org/pep-0703/)
- [Python Free-Threading Guide](https://py-free-threading.github.io/)
- [State of Python 3.13 Performance](https://codspeed.io/blog/state-of-python-3-13-performance-free-threading)
- [aiohttp Free-Threading Issue #8796](https://github.com/aio-libs/aiohttp/issues/8796)
- [NumPy Free-Threading Issue #26157](https://github.com/numpy/numpy/issues/26157)
- [Polars Build Issue #21889](https://github.com/pola-rs/polars/issues/21889)

## Related Issues

- Issue #26: Python 3.13 Free-Threaded Compatibility Testing
- ADR-001: Hybrid Rust/Python Architecture
