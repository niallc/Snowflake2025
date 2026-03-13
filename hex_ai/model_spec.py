"""
Model specification and checkpoint metadata helpers.

This module provides a small explicit contract for describing model
architectures in checkpoints and in-memory objects, so loaders do not have
to rely on hardcoded defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Union

import torch

from .config import BOARD_SIZE

DEFAULT_MODEL_TYPE = "katago_inspired"
MODEL_SPEC_VERSION = 1
_TRUNK_BLOCK_KEY_RE = re.compile(r"^trunk\.(\d+)\.")


@dataclass(frozen=True)
class ModelSpec:
    """Minimal structural description required to rebuild a model."""

    model_type: str = DEFAULT_MODEL_TYPE
    num_blocks: int = 7
    trunk_channels: int = 128
    board_size: int = BOARD_SIZE
    spec_version: int = MODEL_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.model_type:
            raise ValueError("model_type must be a non-empty string")
        if int(self.num_blocks) <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if int(self.trunk_channels) <= 0:
            raise ValueError(
                f"trunk_channels must be positive, got {self.trunk_channels}"
            )
        if int(self.board_size) <= 0:
            raise ValueError(f"board_size must be positive, got {self.board_size}")
        if int(self.spec_version) <= 0:
            raise ValueError(
                f"spec_version must be positive, got {self.spec_version}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "model_type": self.model_type,
            "num_blocks": int(self.num_blocks),
            "trunk_channels": int(self.trunk_channels),
            "board_size": int(self.board_size),
            "spec_version": int(self.spec_version),
        }

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Return kwargs accepted by the current model factory."""
        return {
            "model_type": self.model_type,
            "num_blocks": int(self.num_blocks),
            "trunk_channels": int(self.trunk_channels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        """Construct a spec from checkpoint or metadata payload."""
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"ModelSpec payload must be a mapping, got {type(payload)!r}"
            )
        model_type = payload.get("model_type", DEFAULT_MODEL_TYPE)
        return cls(
            model_type=str(model_type),
            num_blocks=int(payload.get("num_blocks", 7)),
            trunk_channels=int(payload.get("trunk_channels", 128)),
            board_size=int(payload.get("board_size", BOARD_SIZE)),
            spec_version=int(payload.get("spec_version", MODEL_SPEC_VERSION)),
        )


def build_model_from_spec(model_spec: ModelSpec):
    """Instantiate a model from a spec."""
    from .models import create_model

    return create_model(**model_spec.to_model_kwargs())


def model_spec_from_model(model: Any) -> ModelSpec:
    """Extract a spec from a live model instance."""
    model_type = str(getattr(model, "model_type", DEFAULT_MODEL_TYPE))
    num_blocks = getattr(model, "num_blocks", None)
    trunk_channels = getattr(model, "trunk_channels", None)
    policy_head = getattr(model, "policy_head", None)
    board_size = getattr(policy_head, "board_size", BOARD_SIZE)
    if num_blocks is None or trunk_channels is None:
        raise ValueError(
            f"Model {type(model).__name__} does not expose num_blocks/trunk_channels"
        )
    return ModelSpec(
        model_type=model_type,
        num_blocks=int(num_blocks),
        trunk_channels=int(trunk_channels),
        board_size=int(board_size),
    )


def model_spec_from_hyperparameters(hyperparameters: Mapping[str, Any]) -> ModelSpec:
    """Build a spec from experiment hyperparameters."""
    if not isinstance(hyperparameters, Mapping):
        raise TypeError(
            f"hyperparameters must be a mapping, got {type(hyperparameters)!r}"
        )
    return ModelSpec(
        model_type=str(hyperparameters.get("model_type", DEFAULT_MODEL_TYPE)),
        num_blocks=int(hyperparameters.get("num_blocks", 7)),
        trunk_channels=int(hyperparameters.get("trunk_channels", 128)),
        board_size=int(hyperparameters.get("board_size", BOARD_SIZE)),
    )


def load_checkpoint_payload(
    checkpoint_path: Union[str, Path],
    map_location: Optional[Any] = None,
) -> Mapping[str, Any]:
    """Load a checkpoint or raw state dict from disk."""
    path = Path(checkpoint_path)
    if _is_gzipped(path):
        with gzip.open(path, "rb") as handle:
            payload = torch.load(
                handle, map_location=map_location, weights_only=False
            )
    else:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Checkpoint payload must be a mapping, got {type(payload)!r}"
        )
    return payload


def extract_state_dict_from_checkpoint_payload(
    checkpoint_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the state dict whether payload is wrapped or raw."""
    if "model_state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["model_state_dict"]
    else:
        state_dict = checkpoint_payload
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"State dict must be a mapping, got {type(state_dict)!r}"
        )
    return state_dict


def resolve_model_spec_from_checkpoint_payload(
    checkpoint_payload: Mapping[str, Any],
) -> ModelSpec:
    """Resolve model spec from explicit metadata or legacy state-dict inference."""
    if "model_spec" in checkpoint_payload:
        return ModelSpec.from_dict(checkpoint_payload["model_spec"])
    state_dict = extract_state_dict_from_checkpoint_payload(checkpoint_payload)
    return infer_model_spec_from_state_dict(state_dict)


def infer_model_spec_from_state_dict(state_dict: Mapping[str, Any]) -> ModelSpec:
    """
    Infer a legacy model spec from checkpoint weights.

    This is intentionally narrow: it recognizes the current KataGo-inspired
    family by its key structure and infers only the structural fields that are
    actually needed to rebuild the model.
    """
    input_conv_weight = state_dict.get("input_conv.weight")
    if input_conv_weight is None:
        raise ValueError("Cannot infer model spec: missing input_conv.weight")
    if not hasattr(input_conv_weight, "shape") or len(input_conv_weight.shape) != 4:
        raise ValueError(
            "Cannot infer model spec: input_conv.weight does not look like a conv kernel"
        )
    trunk_channels = int(input_conv_weight.shape[0])

    block_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict.keys()
            for match in [_TRUNK_BLOCK_KEY_RE.match(str(key))]
            if match is not None
        }
    )
    if not block_indices:
        raise ValueError("Cannot infer model spec: no trunk block keys found")
    expected_indices = list(range(block_indices[-1] + 1))
    if block_indices != expected_indices:
        raise ValueError(
            "Cannot infer model spec from non-contiguous trunk block indices: "
            f"{block_indices}"
        )

    return ModelSpec(
        model_type=DEFAULT_MODEL_TYPE,
        num_blocks=len(block_indices),
        trunk_channels=trunk_channels,
        board_size=BOARD_SIZE,
    )


def _is_gzipped(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"
