"""
Shared runtime contract for model families.

This module gives training and inference a small stable interface so the rest
of the codebase does not need to know per-family forward-input details or
optimizer grouping rules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _model_type_label(model: nn.Module) -> str:
    return str(getattr(model, "model_type", type(model).__name__))


def _require_callable(model: nn.Module, method_name: str):
    method = getattr(model, method_name, None)
    if callable(method):
        return method
    raise ValueError(
        f"Model {_model_type_label(model)!r} does not implement required runtime "
        f"method {method_name}()."
    )


def forward_model(
    model: nn.Module,
    boards: torch.Tensor,
    *,
    move_stage: Optional[torch.Tensor] = None,
    batch_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the standard policy/value forward pass through the model family API.
    """
    forward_from_boards = _require_callable(model, "forward_from_boards")
    return forward_from_boards(boards, move_stage=move_stage, batch_idx=batch_idx)


def forward_value_only_model(
    model: nn.Module,
    boards: torch.Tensor,
    *,
    move_stage: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run value-only inference through the model family API.
    """
    forward_value_only_from_boards = _require_callable(
        model, "forward_value_only_from_boards"
    )
    return forward_value_only_from_boards(boards, move_stage=move_stage)


def build_optimizer_param_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    value_learning_rate_factor: float,
    value_weight_decay_factor: float,
) -> List[Dict[str, Any]]:
    """
    Delegate optimizer grouping to the model family.
    """
    build_groups = _require_callable(model, "build_optimizer_param_groups")
    return build_groups(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_learning_rate_factor=value_learning_rate_factor,
        value_weight_decay_factor=value_weight_decay_factor,
    )
