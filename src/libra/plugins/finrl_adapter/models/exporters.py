"""
Model Exporters for FinRL Adapter.

Provides functionality to export trained models to various formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    import onnx

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelExporter:
    """
    Export trained RL models to various formats.

    Supports:
    - ONNX export for deployment
    - PyTorch JIT (TorchScript) for C++ inference
    - Policy-only export for lightweight inference

    Usage:
        exporter = ModelExporter()
        exporter.to_onnx(model, "model.onnx", observation_shape=(84,))
    """

    def __init__(self) -> None:
        """Initialize the model exporter."""
        pass

    def to_onnx(
        self,
        model: Any,
        output_path: Path | str,
        observation_shape: tuple[int, ...],
        opset_version: int = 14,
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            model: Trained SB3 model.
            output_path: Path for output ONNX file.
            observation_shape: Shape of observation space.
            opset_version: ONNX opset version.

        Returns:
            Path to exported ONNX file.

        Raises:
            ImportError: If ONNX or PyTorch not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ONNX export")
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required for ONNX export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the policy network
        policy = model.policy

        # Create dummy input
        dummy_input = torch.randn(1, *observation_shape)

        # Extract just the actor network for inference
        class ActorWrapper(torch.nn.Module):
            def __init__(self, policy: Any) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                # Get deterministic action
                features = self.policy.extract_features(obs)
                if hasattr(self.policy, "mlp_extractor"):
                    latent_pi, _ = self.policy.mlp_extractor(features)
                else:
                    latent_pi = features

                if hasattr(self.policy, "action_net"):
                    return self.policy.action_net(latent_pi)
                return latent_pi

        # Wrap and export
        actor = ActorWrapper(policy)
        actor.eval()

        torch.onnx.export(
            actor,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
        )

        logger.info("Exported model to ONNX: %s", output_path)

        # Verify the model
        if ONNX_AVAILABLE:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified successfully")

        return output_path

    def to_torchscript(
        self,
        model: Any,
        output_path: Path | str,
        observation_shape: tuple[int, ...],
        optimize: bool = True,
    ) -> Path:
        """
        Export model to TorchScript format.

        Args:
            model: Trained SB3 model.
            output_path: Path for output .pt file.
            observation_shape: Shape of observation space.
            optimize: Whether to optimize for inference.

        Returns:
            Path to exported TorchScript file.

        Raises:
            ImportError: If PyTorch not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchScript export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the policy
        policy = model.policy
        policy.eval()

        # Create dummy input
        dummy_input = torch.randn(1, *observation_shape)

        # Trace the model
        traced = torch.jit.trace(policy, dummy_input)

        if optimize:
            # Optimize for inference
            traced = torch.jit.freeze(traced)
            traced = torch.jit.optimize_for_inference(traced)

        # Save
        torch.jit.save(traced, str(output_path))

        logger.info("Exported model to TorchScript: %s", output_path)
        return output_path

    def to_policy_weights(
        self,
        model: Any,
        output_path: Path | str,
    ) -> Path:
        """
        Export just the policy network weights.

        This creates a lightweight file containing only the policy
        parameters needed for inference.

        Args:
            model: Trained SB3 model.
            output_path: Path for output file.

        Returns:
            Path to exported weights file.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for weight export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get policy state dict
        policy = model.policy
        state_dict = policy.state_dict()

        # Filter to only policy-related weights (not value function)
        policy_keys = [k for k in state_dict.keys() if "critic" not in k.lower()]
        policy_weights = {k: state_dict[k] for k in policy_keys}

        # Save
        torch.save(policy_weights, str(output_path))

        logger.info(
            "Exported policy weights to %s (%d parameters)",
            output_path,
            len(policy_weights),
        )
        return output_path

    def export_all(
        self,
        model: Any,
        output_dir: Path | str,
        observation_shape: tuple[int, ...],
        model_name: str = "model",
    ) -> dict[str, Path]:
        """
        Export model to all supported formats.

        Args:
            model: Trained SB3 model.
            output_dir: Directory for output files.
            observation_shape: Shape of observation space.
            model_name: Base name for output files.

        Returns:
            Dictionary mapping format to output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        # ONNX
        if ONNX_AVAILABLE and TORCH_AVAILABLE:
            try:
                results["onnx"] = self.to_onnx(
                    model,
                    output_dir / f"{model_name}.onnx",
                    observation_shape,
                )
            except Exception as e:
                logger.warning("ONNX export failed: %s", e)

        # TorchScript
        if TORCH_AVAILABLE:
            try:
                results["torchscript"] = self.to_torchscript(
                    model,
                    output_dir / f"{model_name}.pt",
                    observation_shape,
                )
            except Exception as e:
                logger.warning("TorchScript export failed: %s", e)

        # Weights only
        if TORCH_AVAILABLE:
            try:
                results["weights"] = self.to_policy_weights(
                    model,
                    output_dir / f"{model_name}_weights.pt",
                )
            except Exception as e:
                logger.warning("Weight export failed: %s", e)

        return results


class ONNXInference:
    """
    ONNX inference runtime for exported models.

    Provides lightweight inference without SB3 dependencies.

    Usage:
        runtime = ONNXInference("model.onnx")
        action = runtime.predict(observation)
    """

    def __init__(self, model_path: Path | str) -> None:
        """
        Initialize ONNX inference runtime.

        Args:
            model_path: Path to ONNX model file.
        """
        try:
            import onnxruntime as ort

            self.ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create inference session
        self.session = ort.InferenceSession(str(model_path))

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info("Loaded ONNX model from %s", model_path)

    def predict(
        self,
        observation: Any,
    ) -> Any:
        """
        Run inference on observation.

        Args:
            observation: Input observation array.

        Returns:
            Action array.
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for inference")

        # Ensure proper shape and type
        if isinstance(observation, list):
            observation = np.array(observation, dtype=np.float32)
        elif not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        observation = observation.astype(np.float32)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: observation},
        )

        return outputs[0]

    def predict_batch(
        self,
        observations: Any,
    ) -> Any:
        """
        Run inference on batch of observations.

        Args:
            observations: Batch of observations (N, obs_dim).

        Returns:
            Batch of actions (N, action_dim).
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for inference")

        observations = np.array(observations, dtype=np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: observations},
        )

        return outputs[0]


def export_model(
    model: Any,
    output_path: Path | str,
    observation_shape: tuple[int, ...],
    format: str = "onnx",
) -> Path:
    """
    Convenience function to export a model.

    Args:
        model: Trained SB3 model.
        output_path: Output file path.
        observation_shape: Shape of observations.
        format: Export format (onnx, torchscript, weights).

    Returns:
        Path to exported file.
    """
    exporter = ModelExporter()

    if format == "onnx":
        return exporter.to_onnx(model, output_path, observation_shape)
    elif format == "torchscript":
        return exporter.to_torchscript(model, output_path, observation_shape)
    elif format == "weights":
        return exporter.to_policy_weights(model, output_path)
    else:
        raise ValueError(f"Unknown export format: {format}")
