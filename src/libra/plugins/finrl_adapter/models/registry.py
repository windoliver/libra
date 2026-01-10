"""
Model Registry for FinRL Adapter.

Provides model versioning, tracking, and management functionality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import msgspec

logger = logging.getLogger(__name__)


class ModelMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """
    Metadata for a registered model.

    Attributes:
        model_id: Unique identifier for the model.
        name: Human-readable model name.
        version: Model version string.
        algorithm: RL algorithm used.
        description: Optional description.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        training_config: Training configuration used.
        performance_metrics: Performance metrics from evaluation.
        tags: Custom tags for organization.
        parent_model_id: ID of parent model (for fine-tuned models).
        status: Model status (training, ready, deployed, archived).
    """

    model_id: str
    name: str
    version: str
    algorithm: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    training_config: dict[str, Any] = {}
    performance_metrics: dict[str, float] = {}
    tags: tuple[str, ...] = ()
    parent_model_id: str | None = None
    status: str = "ready"

    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        algorithm: str,
        **kwargs: Any,
    ) -> ModelMetadata:
        """
        Create new model metadata with auto-generated ID.

        Args:
            name: Model name.
            version: Version string.
            algorithm: RL algorithm.
            **kwargs: Additional metadata fields.

        Returns:
            New ModelMetadata instance.
        """
        now = datetime.utcnow().isoformat()
        model_id = cls._generate_id(name, version, now)

        # Convert lists to tuples
        if "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"] = tuple(kwargs["tags"])

        return cls(
            model_id=model_id,
            name=name,
            version=version,
            algorithm=algorithm,
            created_at=now,
            updated_at=now,
            **kwargs,
        )

    @staticmethod
    def _generate_id(name: str, version: str, timestamp: str) -> str:
        """Generate unique model ID."""
        content = f"{name}:{version}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field in msgspec.structs.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, tuple):
                result[field.name] = list(value)
            else:
                result[field.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = tuple(data["tags"])
        known_fields = {f.name for f in msgspec.structs.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class ModelRegistry:
    """
    Registry for managing RL model versions.

    Provides:
    - Model registration and versioning
    - Model metadata storage
    - Model artifact management
    - A/B testing support
    - Model promotion workflows

    Usage:
        registry = ModelRegistry("./models")
        metadata = registry.register_model(
            model_path="trained_model.zip",
            name="ppo_trading",
            version="1.0.0",
            algorithm="ppo",
        )
        model_path = registry.get_model_path(metadata.model_id)
    """

    METADATA_FILE = "registry.json"
    MODELS_DIR = "artifacts"

    def __init__(self, registry_path: Path | str) -> None:
        """
        Initialize the model registry.

        Args:
            registry_path: Path to the registry directory.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.registry_path / self.MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)

        self.metadata_path = self.registry_path / self.METADATA_FILE
        self._models: dict[str, ModelMetadata] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    data = json.load(f)
                self._models = {
                    model_id: ModelMetadata.from_dict(meta)
                    for model_id, meta in data.get("models", {}).items()
                }
                logger.info("Loaded %d models from registry", len(self._models))
            except Exception as e:
                logger.warning("Failed to load registry: %s", e)
                self._models = {}
        else:
            self._models = {}

    def _save_registry(self) -> None:
        """Save registry metadata to disk."""
        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "models": {
                model_id: meta.to_dict()
                for model_id, meta in self._models.items()
            },
        }
        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_path: Path | str,
        name: str,
        version: str,
        algorithm: str,
        description: str = "",
        training_config: dict[str, Any] | None = None,
        performance_metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        copy_artifacts: bool = True,
    ) -> ModelMetadata:
        """
        Register a new model in the registry.

        Args:
            model_path: Path to the model file.
            name: Model name.
            version: Version string.
            algorithm: RL algorithm used.
            description: Optional description.
            training_config: Training configuration.
            performance_metrics: Performance metrics.
            tags: Custom tags.
            copy_artifacts: Whether to copy model files to registry.

        Returns:
            Created ModelMetadata.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create metadata
        metadata = ModelMetadata.create(
            name=name,
            version=version,
            algorithm=algorithm,
            description=description,
            training_config=training_config or {},
            performance_metrics=performance_metrics or {},
            tags=tags or [],
        )

        # Copy artifacts
        if copy_artifacts:
            artifact_dir = self.models_dir / metadata.model_id
            artifact_dir.mkdir(exist_ok=True)

            # Copy model file
            dest_path = artifact_dir / model_path.name
            shutil.copy2(model_path, dest_path)

            # Copy normalization stats if present
            stats_path = model_path.with_suffix(".pkl")
            if stats_path.exists():
                shutil.copy2(stats_path, artifact_dir / stats_path.name)

            logger.info("Copied model artifacts to %s", artifact_dir)

        # Register
        self._models[metadata.model_id] = metadata
        self._save_registry()

        logger.info(
            "Registered model: %s (id=%s, version=%s)",
            name,
            metadata.model_id,
            version,
        )
        return metadata

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """
        Get model metadata by ID.

        Args:
            model_id: Model ID.

        Returns:
            ModelMetadata or None if not found.
        """
        return self._models.get(model_id)

    def get_model_by_name(
        self,
        name: str,
        version: str | None = None,
    ) -> ModelMetadata | None:
        """
        Get model by name and optionally version.

        Args:
            name: Model name.
            version: Optional version (returns latest if not specified).

        Returns:
            ModelMetadata or None if not found.
        """
        matches = [m for m in self._models.values() if m.name == name]

        if not matches:
            return None

        if version:
            for m in matches:
                if m.version == version:
                    return m
            return None

        # Return latest version
        return max(matches, key=lambda m: m.created_at)

    def get_model_path(self, model_id: str) -> Path | None:
        """
        Get path to model artifacts.

        Args:
            model_id: Model ID.

        Returns:
            Path to model directory or None if not found.
        """
        metadata = self.get_model(model_id)
        if metadata is None:
            return None

        artifact_dir = self.models_dir / model_id
        if not artifact_dir.exists():
            return None

        # Find the model file
        for ext in [".zip", ".pt", ".pth", ".onnx"]:
            files = list(artifact_dir.glob(f"*{ext}"))
            if files:
                return files[0]

        return artifact_dir

    def list_models(
        self,
        name: str | None = None,
        algorithm: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
    ) -> list[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            name: Filter by name.
            algorithm: Filter by algorithm.
            tags: Filter by tags (any match).
            status: Filter by status.

        Returns:
            List of matching ModelMetadata.
        """
        results = list(self._models.values())

        if name:
            results = [m for m in results if m.name == name]
        if algorithm:
            results = [m for m in results if m.algorithm == algorithm]
        if tags:
            tag_set = set(tags)
            results = [m for m in results if tag_set & set(m.tags)]
        if status:
            results = [m for m in results if m.status == status]

        return sorted(results, key=lambda m: m.created_at, reverse=True)

    def update_model(
        self,
        model_id: str,
        status: str | None = None,
        performance_metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
    ) -> ModelMetadata | None:
        """
        Update model metadata.

        Args:
            model_id: Model ID.
            status: New status.
            performance_metrics: New/updated metrics.
            tags: New tags (replaces existing).

        Returns:
            Updated ModelMetadata or None if not found.
        """
        metadata = self._models.get(model_id)
        if metadata is None:
            return None

        # Create updated metadata
        updates: dict[str, Any] = {
            "updated_at": datetime.utcnow().isoformat(),
        }
        if status:
            updates["status"] = status
        if performance_metrics:
            updates["performance_metrics"] = {
                **metadata.performance_metrics,
                **performance_metrics,
            }
        if tags is not None:
            updates["tags"] = tuple(tags)

        # Create new metadata with updates
        new_data = metadata.to_dict()
        new_data.update(updates)
        new_metadata = ModelMetadata.from_dict(new_data)

        self._models[model_id] = new_metadata
        self._save_registry()

        return new_metadata

    def delete_model(
        self,
        model_id: str,
        delete_artifacts: bool = True,
    ) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID.
            delete_artifacts: Whether to delete model files.

        Returns:
            True if deleted, False if not found.
        """
        if model_id not in self._models:
            return False

        # Delete artifacts
        if delete_artifacts:
            artifact_dir = self.models_dir / model_id
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
                logger.info("Deleted artifacts: %s", artifact_dir)

        # Remove from registry
        del self._models[model_id]
        self._save_registry()

        logger.info("Deleted model: %s", model_id)
        return True

    def promote_model(
        self,
        model_id: str,
        stage: str = "production",
    ) -> ModelMetadata | None:
        """
        Promote a model to a deployment stage.

        Args:
            model_id: Model ID.
            stage: Target stage (staging, production, etc.).

        Returns:
            Updated ModelMetadata or None if not found.
        """
        return self.update_model(
            model_id,
            status=stage,
            tags=[stage],
        )

    def get_production_model(self, name: str) -> ModelMetadata | None:
        """
        Get the production model for a given name.

        Args:
            name: Model name.

        Returns:
            Production ModelMetadata or None.
        """
        models = self.list_models(name=name, status="production")
        return models[0] if models else None

    def compare_models(
        self,
        model_ids: list[str],
    ) -> dict[str, Any]:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs to compare.

        Returns:
            Comparison dictionary with metrics.
        """
        models = [self.get_model(mid) for mid in model_ids]
        models = [m for m in models if m is not None]

        if not models:
            return {}

        # Collect all metric names
        all_metrics: set[str] = set()
        for m in models:
            all_metrics.update(m.performance_metrics.keys())

        # Build comparison
        comparison = {
            "models": [m.to_dict() for m in models],
            "metrics": {},
        }

        for metric in all_metrics:
            values = [
                m.performance_metrics.get(metric) for m in models
            ]
            comparison["metrics"][metric] = {
                "values": values,
                "best_idx": values.index(max(v for v in values if v is not None))
                if any(v is not None for v in values)
                else None,
            }

        return comparison
