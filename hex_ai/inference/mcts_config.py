"""
Configuration and preset factory utilities for MCTS.

This module isolates config structure/defaults from the runtime search implementation
to keep `mcts.py` focused on search flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hex_ai.config import (
    DEFAULT_BATCH_CAP,
    DEFAULT_C_PUCT,
    DEFAULT_GUMBEL_SIM_THRESHOLD,
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    DEFAULT_GUMBEL_CANDIDATE_MIN,
    DEFAULT_GUMBEL_CANDIDATE_MAX,
    DEFAULT_GUMBEL_C_VISIT,
    DEFAULT_MCTS_DIRICHLET_ALPHA,
    DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_MCTS_ENABLE_TERMINAL_MOVE_DETECTION,
    DEFAULT_GUMBEL_USE_GUMBEL_IN_FINAL_EVAL,
    TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
)
from hex_ai.utils.temperature import SUPPORTED_TEMPERATURE_DECAY_TYPES


# Local MCTS config defaults (distinct from broader app/model defaults in hex_ai.config).
DEFAULT_CONFIDENCE_TERMINATION_THRESHOLD = 0.9
DEFAULT_TERMINAL_MOVE_BOOST = 2.0
DEFAULT_DEPTH_DISCOUNT_FACTOR = 0.97
DEFAULT_CACHE_SIZE = 100000
DEFAULT_DIRICHLET_EPS = 0.25
DEFAULT_TEMPERATURE_START = 1.0
DEFAULT_TEMPERATURE_END = 0.1
DEFAULT_TEMPERATURE_DECAY_TYPE = "exponential"
DEFAULT_TEMPERATURE_DECAY_MOVES = 50
DEFAULT_TERMINAL_DETECTION_MAX_DEPTH = 3
DEFAULT_GUMBEL_TEMPERATURE_ENABLED = True
DEFAULT_TEMPERATURE_DETERMINISTIC_CUTOFF = 0.02
DEFAULT_VISIT_SAMPLING_TOP_K = 5
DEFAULT_GUMBEL_ROOT_TEMPERATURE = 1.0
DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF = -1.0


@dataclass
class BaselineMCTSConfig:
    sims: int = 200
    batch_cap: int = DEFAULT_BATCH_CAP
    c_puct: float = DEFAULT_C_PUCT
    cache_size: int = DEFAULT_CACHE_SIZE
    dirichlet_alpha: float = DEFAULT_MCTS_DIRICHLET_ALPHA
    dirichlet_eps: float = DEFAULT_DIRICHLET_EPS
    add_root_noise: bool = False
    # Temperature scaling parameters used for visit-count move sampling.
    temperature_start: float = DEFAULT_TEMPERATURE_START
    temperature_end: float = DEFAULT_TEMPERATURE_END
    temperature_decay_type: str = DEFAULT_TEMPERATURE_DECAY_TYPE
    temperature_decay_moves: int = DEFAULT_TEMPERATURE_DECAY_MOVES
    temperature_step_thresholds: List[int] = field(default_factory=lambda: [10, 25, 50])
    temperature_step_values: List[float] = field(default_factory=lambda: [0.8, 0.5, 0.2])
    # Terminal move detection parameters
    enable_terminal_move_detection: bool = DEFAULT_MCTS_ENABLE_TERMINAL_MOVE_DETECTION
    terminal_detection_max_depth: int = DEFAULT_TERMINAL_DETECTION_MAX_DEPTH
    terminal_move_boost: float = DEFAULT_TERMINAL_MOVE_BOOST
    # New terminal move handling
    prefer_immediate_terminal: bool = True
    terminal_win_score_bonus: float = 0.25
    # Adaptive batch selection for low simulation counts
    adaptive_distinct_target: bool = False
    distinct_target_min: int = 8
    distinct_target_max: int = 16

    # Confidence-based termination parameters
    enable_confidence_termination: bool = False
    confidence_termination_threshold: float = DEFAULT_CONFIDENCE_TERMINATION_THRESHOLD
    confidence_termination_probability: float = 0.98

    # Depth-based discounting parameters
    enable_depth_discounting: bool = True
    depth_discount_factor: float = DEFAULT_DEPTH_DISCOUNT_FACTOR

    # Top-k filtering for visit count sampling
    visit_sampling_top_k: int = DEFAULT_VISIT_SAMPLING_TOP_K

    # Gumbel-AlphaZero root selection parameters
    enable_gumbel_root_selection: bool = True
    gumbel_sim_threshold: int = DEFAULT_GUMBEL_SIM_THRESHOLD
    gumbel_c_visit: float = DEFAULT_GUMBEL_C_VISIT
    gumbel_c_scale: float = DEFAULT_GUMBEL_C_SCALE
    gumbel_m_candidates: Optional[int] = None

    # Gumbel candidate scaling parameters (power-law scaling)
    gumbel_candidate_power_scale: float = DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE
    gumbel_candidate_power_rate: float = DEFAULT_GUMBEL_CANDIDATE_POWER_RATE
    gumbel_candidate_power_offset: float = DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET
    gumbel_candidate_min: int = DEFAULT_GUMBEL_CANDIDATE_MIN
    gumbel_candidate_max: int = DEFAULT_GUMBEL_CANDIDATE_MAX
    # Gumbel ranking stabilization parameters
    gumbel_use_gumbel_in_final_eval: bool = DEFAULT_GUMBEL_USE_GUMBEL_IN_FINAL_EVAL

    # Deterministic cutoff for visit-count sampling (non-Gumbel move selection path).
    temperature_deterministic_cutoff: float = DEFAULT_TEMPERATURE_DETERMINISTIC_CUTOFF

    # Legacy Gumbel temperature controls. These are intentionally fixed:
    # Gumbel root selection currently runs with temperature=1.0 and no cutoff path.
    gumbel_temperature_enabled: bool = DEFAULT_GUMBEL_TEMPERATURE_ENABLED
    gumbel_temperature_deterministic_cutoff: float = DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF

    # Batch flushing control parameters
    distinct_target: int = 32
    enable_low_distinct_ratio_flush: bool = False

    def __post_init__(self):
        if self.sims <= 0:
            raise ValueError(f"sims must be positive, got {self.sims}")
        if self.batch_cap <= 0:
            raise ValueError(f"batch_cap must be positive, got {self.batch_cap}")
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be positive, got {self.c_puct}")
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be positive, got {self.dirichlet_alpha}")
        if not 0 <= self.dirichlet_eps <= 1:
            raise ValueError(f"dirichlet_eps must be between 0 and 1, got {self.dirichlet_eps}")
        if self.temperature_start <= 0:
            raise ValueError(f"temperature_start must be positive, got {self.temperature_start}")
        if self.temperature_end <= 0:
            raise ValueError(f"temperature_end must be positive, got {self.temperature_end}")
        if self.temperature_start < self.temperature_end:
            raise ValueError(
                f"temperature_start ({self.temperature_start}) must be >= temperature_end ({self.temperature_end})"
            )
        if self.temperature_decay_moves <= 0:
            raise ValueError(f"temperature_decay_moves must be positive, got {self.temperature_decay_moves}")
        if self.temperature_decay_type not in SUPPORTED_TEMPERATURE_DECAY_TYPES:
            raise ValueError(
                f"temperature_decay_type must be one of {SUPPORTED_TEMPERATURE_DECAY_TYPES}, "
                f"got {self.temperature_decay_type}"
            )
        if self.terminal_move_boost < 0:
            raise ValueError(f"terminal_move_boost must be non-negative, got {self.terminal_move_boost}")
        if self.terminal_detection_max_depth < 0:
            raise ValueError(f"terminal_detection_max_depth must be non-negative, got {self.terminal_detection_max_depth}")
        if not 0 <= self.confidence_termination_threshold <= 1:
            raise ValueError(
                "confidence_termination_threshold must be between 0 and 1 "
                f"(represents distance from neutral), got {self.confidence_termination_threshold}"
            )
        if not 0 <= self.confidence_termination_probability <= 0.98:
            raise ValueError(
                "confidence_termination_probability must be between 0 and 1 "
                f"(probability of early termination when threshold is exceeded), got {self.confidence_termination_probability}"
            )
        if not 0 < self.depth_discount_factor <= 1:
            raise ValueError(f"depth_discount_factor must be between 0 and 1, got {self.depth_discount_factor}")

        if self.distinct_target <= 0:
            raise ValueError(f"distinct_target must be positive, got {self.distinct_target}")
        if self.distinct_target > self.batch_cap:
            raise ValueError(f"distinct_target ({self.distinct_target}) cannot exceed batch_cap ({self.batch_cap})")

        if self.distinct_target_min <= 0:
            raise ValueError(f"distinct_target_min must be positive, got {self.distinct_target_min}")
        if self.distinct_target_max <= 0:
            raise ValueError(f"distinct_target_max must be positive, got {self.distinct_target_max}")
        if self.distinct_target_min > self.distinct_target_max:
            raise ValueError(
                f"distinct_target_min ({self.distinct_target_min}) cannot exceed distinct_target_max ({self.distinct_target_max})"
            )

        if self.gumbel_sim_threshold <= 0:
            raise ValueError(f"gumbel_sim_threshold must be positive, got {self.gumbel_sim_threshold}")
        if self.gumbel_c_visit <= 0:
            raise ValueError(f"gumbel_c_visit must be positive, got {self.gumbel_c_visit}")
        if self.gumbel_c_scale <= 0:
            raise ValueError(f"gumbel_c_scale must be positive, got {self.gumbel_c_scale}")
        if self.gumbel_m_candidates is not None and self.gumbel_m_candidates <= 0:
            raise ValueError(f"gumbel_m_candidates must be positive, got {self.gumbel_m_candidates}")

        if self.gumbel_candidate_power_scale <= 0.0:
            raise ValueError(f"gumbel_candidate_power_scale must be > 0.0, got {self.gumbel_candidate_power_scale}")
        if self.gumbel_candidate_power_rate <= 0.0:
            raise ValueError(f"gumbel_candidate_power_rate must be > 0.0, got {self.gumbel_candidate_power_rate}")
        if self.gumbel_candidate_min <= 0:
            raise ValueError(f"gumbel_candidate_min must be positive, got {self.gumbel_candidate_min}")
        if self.gumbel_candidate_max <= 0:
            raise ValueError(f"gumbel_candidate_max must be positive, got {self.gumbel_candidate_max}")
        if self.gumbel_candidate_min > self.gumbel_candidate_max:
            raise ValueError(
                f"gumbel_candidate_min ({self.gumbel_candidate_min}) cannot exceed gumbel_candidate_max ({self.gumbel_candidate_max})"
            )

        if self.temperature_deterministic_cutoff <= 0:
            raise ValueError(f"temperature_deterministic_cutoff must be positive, got {self.temperature_deterministic_cutoff}")
        if not self.gumbel_temperature_enabled:
            raise ValueError(
                "gumbel_temperature_enabled=False is unsupported: "
                f"Gumbel root selection uses fixed temperature={DEFAULT_GUMBEL_ROOT_TEMPERATURE}."
            )
        if self.gumbel_temperature_deterministic_cutoff != DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF:
            raise ValueError(
                "gumbel_temperature_deterministic_cutoff is unsupported and must remain "
                f"{DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF}."
            )

        if self.temperature_decay_type == "step":
            if len(self.temperature_step_thresholds) != len(self.temperature_step_values):
                raise ValueError("temperature_step_thresholds and temperature_step_values must have the same length")
            if not all(t >= 0 for t in self.temperature_step_thresholds):
                raise ValueError("All temperature_step_thresholds must be non-negative")
            if not all(0 < v <= 1 for v in self.temperature_step_values):
                raise ValueError("All temperature_step_values must be between 0 and 1")


def create_mcts_config(
    config_type: str = "tournament",
    sims: Optional[int] = None,
    confidence_termination_threshold: Optional[float] = None,
    confidence_termination_probability: Optional[float] = None,
    cache_size: Optional[int] = None,
    **kwargs
) -> BaselineMCTSConfig:
    """
    Create an MCTS configuration with preset defaults for different use cases.

    Args:
        config_type: Type of configuration ("tournament", "selfplay", "fast_selfplay")
        sims: Number of simulations (overrides preset default)
        confidence_termination_threshold: Distance from neutral for confidence termination
        confidence_termination_probability: Probability of early termination when threshold is exceeded
        cache_size: Cache size for MCTS evaluation cache
        **kwargs: Additional parameters to override in the configuration
    """
    presets = {
        "tournament": {
            "sims": 200,
            "confidence_termination_threshold": TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
            "confidence_termination_probability": 0.95,
            "temperature_start": 1.0,
            "temperature_end": 1.0,
            "add_root_noise": True,
        },
        "selfplay": {
            "sims": 500,
            "confidence_termination_threshold": 0.85,
            "confidence_termination_probability": 0.95,
            "temperature_start": 0.5,
            "temperature_end": 0.01,
            "add_root_noise": True,
        },
        "fast_selfplay": {
            "sims": 200,
            "confidence_termination_threshold": 0.8,
            "temperature_start": 0.5,
            "temperature_end": 0.01,
            "add_root_noise": True,
        },
    }

    if config_type not in presets:
        raise ValueError(f"Unknown config_type: {config_type}. Must be one of {list(presets.keys())}")

    config_params = presets[config_type].copy()

    if sims is not None:
        config_params["sims"] = sims
    if confidence_termination_threshold is not None:
        config_params["confidence_termination_threshold"] = confidence_termination_threshold
    if confidence_termination_probability is not None:
        config_params["confidence_termination_probability"] = confidence_termination_probability
    if cache_size is not None:
        config_params["cache_size"] = cache_size

    config_params.update(kwargs)

    default_params = {
        "c_puct": DEFAULT_C_PUCT,
        "batch_cap": DEFAULT_BATCH_CAP,
        "dirichlet_alpha": DEFAULT_MCTS_DIRICHLET_ALPHA,
        "dirichlet_eps": DEFAULT_DIRICHLET_EPS,
        "temperature_decay_type": DEFAULT_TEMPERATURE_DECAY_TYPE,
        "temperature_decay_moves": DEFAULT_TEMPERATURE_DECAY_MOVES,
        "enable_terminal_move_detection": DEFAULT_MCTS_ENABLE_TERMINAL_MOVE_DETECTION,
        "terminal_move_boost": DEFAULT_TERMINAL_MOVE_BOOST,
        "terminal_detection_max_depth": DEFAULT_TERMINAL_DETECTION_MAX_DEPTH,
        "prefer_immediate_terminal": True,
        "terminal_win_score_bonus": 0.25,
        "adaptive_distinct_target": False,
        "distinct_target_min": 8,
        "distinct_target_max": 16,
        "enable_confidence_termination": True,
        "enable_depth_discounting": True,
        "depth_discount_factor": DEFAULT_DEPTH_DISCOUNT_FACTOR,
        "gumbel_temperature_enabled": DEFAULT_GUMBEL_TEMPERATURE_ENABLED,
        "temperature_deterministic_cutoff": DEFAULT_TEMPERATURE_DETERMINISTIC_CUTOFF,
        "gumbel_temperature_deterministic_cutoff": DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF,
        "distinct_target": 32,
        "enable_low_distinct_ratio_flush": False,
    }

    explicit_distinct_target = "distinct_target" in config_params
    for key, value in default_params.items():
        config_params.setdefault(key, value)

    if not explicit_distinct_target:
        config_params["distinct_target"] = min(int(config_params["distinct_target"]), int(config_params["batch_cap"]))

    return BaselineMCTSConfig(**config_params)
