"""
Execution Algorithm Registry.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

The registry provides a centralized way to discover and instantiate
execution algorithms by name. This enables:
- Runtime algorithm selection via configuration
- Plugin-style algorithm extensions
- Easy enumeration of available algorithms

Example:
    # Register algorithms
    registry = AlgorithmRegistry()
    registry.register("twap", TWAPAlgorithm)
    registry.register("vwap", VWAPAlgorithm)
    registry.register("iceberg", IcebergAlgorithm)

    # Create by name
    algo = registry.create("twap", horizon_secs=120, interval_secs=10)

    # Or use global registry
    from libra.execution import algorithm_registry
    algo = algorithm_registry.create("vwap", num_intervals=12)

References:
- NautilusTrader: ExecAlgorithmFactory pattern
- QuantConnect: ExecutionModelFactory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from libra.execution.algorithm import BaseExecAlgorithm


if TYPE_CHECKING:
    from libra.clients.execution_client import ExecutionClient


# =============================================================================
# Algorithm Registry
# =============================================================================


@dataclass
class AlgorithmRegistry:
    """
    Registry for execution algorithms.

    Allows registration and lookup of algorithm classes by name,
    enabling runtime algorithm selection and factory pattern.

    Example:
        registry = AlgorithmRegistry()

        # Register custom algorithm
        registry.register("my_algo", MyCustomAlgorithm)

        # Create instance
        algo = registry.create("my_algo", param1=value1)

        # List available
        print(registry.list_algorithms())
    """

    _algorithms: dict[str, type[BaseExecAlgorithm]] = field(default_factory=dict)

    def register(
        self,
        name: str,
        algorithm_class: type[BaseExecAlgorithm],
    ) -> None:
        """
        Register an algorithm class.

        Args:
            name: Unique name for the algorithm (e.g., "twap", "vwap")
            algorithm_class: The algorithm class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' is already registered")
        self._algorithms[name] = algorithm_class

    def unregister(self, name: str) -> bool:
        """
        Unregister an algorithm.

        Args:
            name: Name of algorithm to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._algorithms:
            del self._algorithms[name]
            return True
        return False

    def get(self, name: str) -> type[BaseExecAlgorithm] | None:
        """
        Get an algorithm class by name.

        Args:
            name: Algorithm name

        Returns:
            Algorithm class or None if not found
        """
        return self._algorithms.get(name)

    def create(
        self,
        name: str,
        execution_client: ExecutionClient | None = None,
        **config_kwargs: Any,
    ) -> BaseExecAlgorithm:
        """
        Create an algorithm instance by name.

        Args:
            name: Algorithm name (e.g., "twap", "vwap", "iceberg")
            execution_client: Optional execution client
            **config_kwargs: Configuration parameters for the algorithm

        Returns:
            Configured algorithm instance

        Raises:
            KeyError: If algorithm name not found

        Example:
            # Create TWAP with custom config
            algo = registry.create(
                "twap",
                execution_client=client,
                horizon_secs=300,
                interval_secs=30,
                randomize_size=True,
            )
        """
        algorithm_class = self._algorithms.get(name)
        if algorithm_class is None:
            available = ", ".join(self._algorithms.keys())
            raise KeyError(
                f"Algorithm '{name}' not found. Available: {available}"
            )

        # Import config classes dynamically based on algorithm type
        # This avoids circular imports
        if name == "twap" and config_kwargs:
            from libra.execution.twap import TWAPConfig

            config = TWAPConfig(**config_kwargs)
            return algorithm_class(config=config, execution_client=execution_client)

        elif name == "vwap" and config_kwargs:
            from libra.execution.vwap import VWAPConfig

            config = VWAPConfig(**config_kwargs)
            return algorithm_class(config=config, execution_client=execution_client)

        elif name == "iceberg" and config_kwargs:
            from libra.execution.iceberg import IcebergConfig

            config = IcebergConfig(**config_kwargs)
            return algorithm_class(config=config, execution_client=execution_client)

        # Generic instantiation for algorithms without config
        return algorithm_class(execution_client=execution_client)

    def list_algorithms(self) -> list[str]:
        """List all registered algorithm names."""
        return list(self._algorithms.keys())

    def __contains__(self, name: str) -> bool:
        """Check if algorithm is registered."""
        return name in self._algorithms

    def __len__(self) -> int:
        """Number of registered algorithms."""
        return len(self._algorithms)


# =============================================================================
# Global Registry Instance
# =============================================================================


def _create_default_registry() -> AlgorithmRegistry:
    """Create and populate the default algorithm registry."""
    from libra.execution.iceberg import IcebergAlgorithm
    from libra.execution.twap import TWAPAlgorithm
    from libra.execution.vwap import VWAPAlgorithm

    registry = AlgorithmRegistry()
    registry.register("twap", TWAPAlgorithm)
    registry.register("vwap", VWAPAlgorithm)
    registry.register("iceberg", IcebergAlgorithm)

    return registry


# Lazy initialization to avoid import cycles
_algorithm_registry: AlgorithmRegistry | None = None


def get_algorithm_registry() -> AlgorithmRegistry:
    """
    Get the global algorithm registry.

    Returns a singleton registry with default algorithms registered.

    Example:
        registry = get_algorithm_registry()
        algo = registry.create("twap", horizon_secs=60)
    """
    global _algorithm_registry
    if _algorithm_registry is None:
        _algorithm_registry = _create_default_registry()
    return _algorithm_registry


# =============================================================================
# Convenience Functions
# =============================================================================


def create_algorithm(
    name: str,
    execution_client: ExecutionClient | None = None,
    **config_kwargs: Any,
) -> BaseExecAlgorithm:
    """
    Create an execution algorithm by name.

    Convenience function using the global registry.

    Args:
        name: Algorithm name ("twap", "vwap", "iceberg")
        execution_client: Optional execution client
        **config_kwargs: Configuration parameters

    Returns:
        Configured algorithm instance

    Example:
        # Create a TWAP algorithm
        algo = create_algorithm(
            "twap",
            horizon_secs=120,
            interval_secs=10,
        )

        # Create with execution client
        algo = create_algorithm(
            "vwap",
            execution_client=my_client,
            num_intervals=12,
            max_participation_pct=0.01,
        )
    """
    return get_algorithm_registry().create(
        name,
        execution_client=execution_client,
        **config_kwargs,
    )


def list_algorithms() -> list[str]:
    """
    List available execution algorithms.

    Returns:
        List of algorithm names

    Example:
        >>> list_algorithms()
        ['twap', 'vwap', 'iceberg']
    """
    return get_algorithm_registry().list_algorithms()
