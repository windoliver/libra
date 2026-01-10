"""Tests for FinRL model registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from libra.plugins.finrl_adapter.models.registry import ModelMetadata, ModelRegistry


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_create(self) -> None:
        """Test creating model metadata."""
        metadata = ModelMetadata.create(
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
            description="Test model",
        )

        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.algorithm == "ppo"
        assert metadata.description == "Test model"
        assert len(metadata.model_id) == 12  # SHA256[:12]
        assert metadata.status == "ready"

    def test_create_with_metrics(self) -> None:
        """Test creating metadata with metrics."""
        metadata = ModelMetadata.create(
            name="test_model",
            version="1.0.0",
            algorithm="sac",
            performance_metrics={"sharpe": 1.5, "return": 0.15},
        )

        assert metadata.performance_metrics["sharpe"] == 1.5
        assert metadata.performance_metrics["return"] == 0.15

    def test_create_with_tags(self) -> None:
        """Test creating metadata with tags."""
        metadata = ModelMetadata.create(
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
            tags=["production", "crypto"],
        )

        assert "production" in metadata.tags
        assert "crypto" in metadata.tags

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        metadata = ModelMetadata.create(
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        result = metadata.to_dict()

        assert result["name"] == "test_model"
        assert result["version"] == "1.0.0"
        assert result["algorithm"] == "ppo"
        assert isinstance(result["tags"], list)

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "model_id": "abc123",
            "name": "test_model",
            "version": "1.0.0",
            "algorithm": "ppo",
            "tags": ["test"],
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.model_id == "abc123"
        assert metadata.name == "test_model"
        assert metadata.tags == ("test",)


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def registry_dir(self) -> Path:
        """Create temporary registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry(self, registry_dir: Path) -> ModelRegistry:
        """Create test registry."""
        return ModelRegistry(registry_dir)

    @pytest.fixture
    def model_file(self, registry_dir: Path) -> Path:
        """Create dummy model file."""
        model_path = registry_dir / "test_model.zip"
        model_path.write_text("dummy model data")
        return model_path

    def test_init_creates_dirs(self, registry_dir: Path) -> None:
        """Test initialization creates directories."""
        registry = ModelRegistry(registry_dir)

        assert registry.registry_path.exists()
        assert registry.models_dir.exists()

    def test_register_model(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test registering a model."""
        metadata = registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
            description="Test model",
        )

        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"

    def test_get_model(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test getting a model by ID."""
        metadata = registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        result = registry.get_model(metadata.model_id)

        assert result is not None
        assert result.name == "test_model"

    def test_get_model_not_found(self, registry: ModelRegistry) -> None:
        """Test getting non-existent model."""
        result = registry.get_model("nonexistent")
        assert result is None

    def test_get_model_by_name(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test getting model by name."""
        registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        result = registry.get_model_by_name("test_model")

        assert result is not None
        assert result.name == "test_model"

    def test_get_model_by_name_with_version(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test getting model by name and version."""
        # Register multiple versions
        registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )
        registry.register_model(
            model_path=model_file,
            name="test_model",
            version="2.0.0",
            algorithm="ppo",
        )

        result = registry.get_model_by_name("test_model", version="1.0.0")

        assert result is not None
        assert result.version == "1.0.0"

    def test_list_models(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test listing models."""
        registry.register_model(
            model_path=model_file,
            name="model_a",
            version="1.0.0",
            algorithm="ppo",
        )
        registry.register_model(
            model_path=model_file,
            name="model_b",
            version="1.0.0",
            algorithm="sac",
        )

        models = registry.list_models()

        assert len(models) == 2

    def test_list_models_filter_by_algorithm(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test filtering models by algorithm."""
        registry.register_model(
            model_path=model_file,
            name="model_a",
            version="1.0.0",
            algorithm="ppo",
        )
        registry.register_model(
            model_path=model_file,
            name="model_b",
            version="1.0.0",
            algorithm="sac",
        )

        models = registry.list_models(algorithm="ppo")

        assert len(models) == 1
        assert models[0].algorithm == "ppo"

    def test_update_model(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test updating model metadata."""
        metadata = registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        updated = registry.update_model(
            model_id=metadata.model_id,
            status="production",
            performance_metrics={"sharpe": 2.0},
        )

        assert updated is not None
        assert updated.status == "production"
        assert updated.performance_metrics["sharpe"] == 2.0

    def test_delete_model(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test deleting a model."""
        metadata = registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        result = registry.delete_model(metadata.model_id)

        assert result is True
        assert registry.get_model(metadata.model_id) is None

    def test_promote_model(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test promoting model to production."""
        metadata = registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        promoted = registry.promote_model(metadata.model_id, stage="production")

        assert promoted is not None
        assert promoted.status == "production"
        assert "production" in promoted.tags

    def test_get_production_model(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test getting production model."""
        metadata = registry.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )
        registry.promote_model(metadata.model_id, stage="production")

        production = registry.get_production_model("test_model")

        assert production is not None
        assert production.status == "production"

    def test_compare_models(
        self,
        registry: ModelRegistry,
        model_file: Path,
    ) -> None:
        """Test comparing models."""
        m1 = registry.register_model(
            model_path=model_file,
            name="model_a",
            version="1.0.0",
            algorithm="ppo",
            performance_metrics={"sharpe": 1.5},
        )
        m2 = registry.register_model(
            model_path=model_file,
            name="model_b",
            version="1.0.0",
            algorithm="sac",
            performance_metrics={"sharpe": 2.0},
        )

        comparison = registry.compare_models([m1.model_id, m2.model_id])

        assert len(comparison["models"]) == 2
        assert "sharpe" in comparison["metrics"]

    def test_persistence(self, registry_dir: Path, model_file: Path) -> None:
        """Test registry persistence across instances."""
        # Create registry and register model
        registry1 = ModelRegistry(registry_dir)
        metadata = registry1.register_model(
            model_path=model_file,
            name="test_model",
            version="1.0.0",
            algorithm="ppo",
        )

        # Create new registry instance
        registry2 = ModelRegistry(registry_dir)
        result = registry2.get_model(metadata.model_id)

        assert result is not None
        assert result.name == "test_model"
