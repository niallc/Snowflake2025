"""Shared helpers for optional inline move heatmap responses."""

from typing import Any, Callable, Mapping, MutableMapping

from hex_ai.web.move_heatmap import build_policy_value_heatmap


INLINE_MOVE_HEATMAP_OPTIONAL_FIELDS = [
    "heatmap_enabled",
    "heatmap_selection_mode",
    "heatmap_top_k",
    "heatmap_policy_temperature",
]


def parse_inline_move_heatmap_options(request_data: Mapping[str, Any]) -> dict:
    """Parse optional inline heatmap settings from request payload fields."""
    enabled = bool(request_data.get("heatmap_enabled", False))
    if not enabled:
        return {"enabled": False}

    selection_mode = request_data.get("heatmap_selection_mode", "policy_top_k")
    top_k = request_data.get("heatmap_top_k", 12)
    policy_temperature = request_data.get("heatmap_policy_temperature", 1.0)

    if selection_mode not in {"policy_top_k", "all_legal"}:
        raise ValueError(f"Invalid heatmap_selection_mode: {selection_mode}")

    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("heatmap_top_k must be an integer") from exc
    if top_k < 1:
        raise ValueError("heatmap_top_k must be >= 1")

    try:
        policy_temperature = float(policy_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("heatmap_policy_temperature must be numeric") from exc
    if policy_temperature <= 0:
        raise ValueError("heatmap_policy_temperature must be > 0")

    return {
        "enabled": True,
        "selection_mode": selection_mode,
        "top_k": top_k,
        "policy_temperature": policy_temperature,
    }


def maybe_attach_inline_move_heatmap(
    result: MutableMapping[str, Any],
    *,
    state,
    model_id: str,
    heatmap_options: Mapping[str, Any],
    model_getter: Callable[[str], Any],
    logger=None,
) -> None:
    """Attach `move_heatmap` payload to successful move responses when requested."""
    if not heatmap_options.get("enabled"):
        return
    if not result.get("success"):
        return

    try:
        model = model_getter(model_id)
        heatmap = build_policy_value_heatmap(
            state=state,
            model=model,
            selection_mode=heatmap_options["selection_mode"],
            top_k=heatmap_options["top_k"],
            policy_temperature=heatmap_options["policy_temperature"],
        )
        result["move_heatmap"] = heatmap.to_dict()
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Inline move heatmap generation failed (model=%s): %s",
                model_id,
                exc,
            )
        result["move_heatmap_error"] = "Failed to compute move heatmap"
