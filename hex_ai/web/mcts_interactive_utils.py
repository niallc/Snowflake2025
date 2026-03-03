"""Shared utilities for interactive MCTS endpoint execution."""

from __future__ import annotations

from hex_ai.inference.game_engine import HexGameEngine
from hex_ai.inference.mcts import create_mcts_config, run_mcts_move


INTERACTIVE_DEAD_CELL_CONFIG = {
    "enable_dead_cell_pruning": True,
    "dead_cell_enable_four_run": True,
    "dead_cell_enable_two_two_split": True,
    "dead_cell_enable_three_plus_one": True,
    "dead_cell_enable_a1b2a3_discouraged": True,
    "dead_cell_enable_double_dead_pairs": False,
}


def get_interactive_dead_cell_config() -> dict:
    """Return dead-cell rule defaults used by interactive web MCTS."""
    return dict(INTERACTIVE_DEAD_CELL_CONFIG)


def create_interactive_mcts_config(
    *,
    num_simulations: int,
    exploration_constant: float,
    temperature: float,
    temperature_end: float,
    enable_gumbel: bool,
    gumbel_max_sims: int,
    confidence_termination_threshold: float,
    logger=None,
):
    """
    Build the MCTS config used by interactive endpoints.

    Returns:
        tuple: (mcts_config, adjusted_temperature_end)
    """
    adjusted_temperature_end = temperature_end
    if adjusted_temperature_end > temperature:
        adjusted_temperature_end = temperature / 10
        if logger is not None:
            logger.info(
                "Adjusting temperature_end from %s to %s (temperature_start/10)",
                temperature_end,
                adjusted_temperature_end,
            )

    if temperature < 0.02 and logger is not None:
        logger.info(
            "Temperature %s is very low (< 0.02), using deterministic move selection safeguards",
            temperature,
        )

    mcts_config = create_mcts_config(
        config_type="tournament",
        confidence_termination_threshold=confidence_termination_threshold,
        sims=num_simulations,
        c_puct=exploration_constant,
        temperature_start=temperature,
        temperature_end=adjusted_temperature_end,
        enable_gumbel_root_selection=enable_gumbel,
        gumbel_sim_threshold=gumbel_max_sims,
        **INTERACTIVE_DEAD_CELL_CONFIG,
    )
    return mcts_config, adjusted_temperature_end


def run_interactive_mcts_search(
    *,
    state,
    model_wrapper,
    mcts_config,
    verbose: int = 0,
    logger=None,
):
    """
    Run one interactive MCTS search and return raw outputs from `run_mcts_move`.
    """
    if logger is not None:
        logger.info("Creating game engine")
    engine = HexGameEngine()

    if logger is not None:
        logger.info("Starting MCTS search")

    try:
        return run_mcts_move(
            engine,
            model_wrapper,
            state,
            mcts_config,
            verbose=verbose,
        )
    except Exception as exc:
        if logger is not None:
            logger.error("run_mcts_move failed with exception: %s", exc)
        raise
