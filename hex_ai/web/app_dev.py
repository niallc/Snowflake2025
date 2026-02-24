from flask import Flask, request, jsonify, send_from_directory
import os
import json
import numpy as np
from flask_cors import CORS
import logging
from datetime import datetime
import time
import random

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import apply_move_to_state_trmph

from hex_ai.inference.fixed_tree_search import run_fixed_tree_search, create_fixed_tree_config
from hex_ai.value_utils import (
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_policy_move,
)
from hex_ai.enums import Piece, Player
from hex_ai.inference.mcts_utils import (
    compute_win_probability_from_tree_data,
    compute_best_child_win_probability_from_tree_data,
)
from hex_ai.config import BOARD_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN, FIXED_TREE_MAX_PRODUCT, FIXED_TREE_DEFAULT_WIDTH, FIXED_TREE_DEFAULT_TEMPERATURE
from hex_ai.web.model_browser import create_model_browser
from hex_ai.file_utils import add_recent_model
from hex_ai.inference.model_config import get_model_path, get_all_model_info, is_valid_model_id
from hex_ai.inference.model_cache import get_model_cache
from hex_ai.web.web_config import INTERACTIVE_CONFIDENCE_TERMINATION_THRESHOLD
from hex_ai.web.move_heatmap import (
    build_policy_value_heatmap,
    parse_move_heatmap_request_params,
)
from hex_ai.web.inline_move_heatmap import (
    validate_request_with_inline_heatmap,
    maybe_attach_inline_move_heatmap,
)
from hex_ai.web.gameplay_response import (
    apply_trmph_sequence_to_state,
    build_engine_error_payload,
    build_engine_move_response,
    build_game_state_response,
)
from hex_ai.web.mcts_interactive_utils import (
    create_interactive_mcts_config,
    run_interactive_mcts_search,
)
from hex_ai.web.interactive_core import (
    create_game_state_from_trmph_input as core_create_game_state_from_trmph_input,
    validate_api_input as core_validate_api_input,
    validate_trmph_input as core_validate_trmph_input,
)

app = Flask(__name__, static_folder="static")
CORS(app)

# API contract for player fields
# - player: UI-friendly color string ("blue"|"red")
# - player_enum: canonical enum name ("BLUE"|"RED")
# - player_index: canonical numeric (0=BLUE, 1=RED)
# - player_raw: remove after frontend migrates

# NOTE: Value head terminology - We use 'value_signed' as a shorthand for [-1, 1] scores
# returned by the value head (tanh activated) and used by MCTS, as opposed to 'value_logits'
# which were the old sigmoid-based outputs.

# Add debug logging for incoming requests
@app.before_request
def log_request_info():
    app.logger.debug(f"Request: {request.method} {request.path}")
    # Store start time for timing analysis
    request.start_time = time.time()

@app.after_request
def log_response_info(response):
    if hasattr(request, 'start_time'):
        request_time = time.time() - request.start_time
        app.logger.info(f"HTTP Request/Response cycle: {request.method} {request.path} took {request_time:.3f}s")
    return response

# Global model browser instance
MODEL_BROWSER = create_model_browser()

# Dynamic model registry for user-selected models (these override the central registry)
DYNAMIC_MODELS = {}

# Get centralized model cache
MODEL_CACHE = get_model_cache()

# Preload default models on startup to avoid first-move delays
def preload_default_models():
    """Preload the default models to avoid loading delays on first move."""
    try:
        app.logger.info("Preloading default models...")
        default_models = ["best", "model2"]  # Current and previous best models
        
        for model_id in default_models:
            try:
                model_path = get_model_path(model_id)
                app.logger.info(f"Preloading {model_id} from {model_path}")
                # Preload both simple and wrapper models
                MODEL_CACHE.get_simple_model(model_path)
                MODEL_CACHE.get_wrapper_model(model_path)
                app.logger.info(f"Successfully preloaded {model_id}")
            except Exception as e:
                app.logger.warning(f"Failed to preload {model_id}: {e}")
        
        app.logger.info("Default model preloading complete")
    except Exception as e:
        app.logger.error(f"Error during model preloading: {e}")

# Preload models on startup
preload_default_models()

DEFAULT_PIE_RULE_ENABLED = True
PIE_RULE_OPENING_WEIGHT_EXPONENT = 6.0

# --- Input Validation ---
def validate_trmph_input(trmph):
    """Validate TRMPH format."""
    return core_validate_trmph_input(trmph, board_size=BOARD_SIZE)

def validate_api_input(data, required_fields=None, optional_fields=None, *, reject_unexpected=False):
    """Centralized validation for API endpoints."""
    return core_validate_api_input(
        data,
        required_fields=required_fields,
        optional_fields=optional_fields,
        logger=app.logger,
        reject_unexpected=reject_unexpected,
        trmph_validator=validate_trmph_input,
        boolean_fields={"pie_rule_enabled"},
    )

# --- Model Management ---
def _resolve_model_path(model_id: str) -> str:
    """Resolve dynamic/registered/direct model inputs to a concrete file path."""
    if model_id in DYNAMIC_MODELS:
        model_path = DYNAMIC_MODELS[model_id]
        if not os.path.isabs(model_path):
            model_path = os.path.join("checkpoints", model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        return model_path

    if is_valid_model_id(model_id):
        return get_model_path(model_id)

    if os.path.exists(model_id):
        return model_id

    app.logger.error(f"Unknown model_id: {model_id}")
    app.logger.error(
        f"Available options: dynamic={list(DYNAMIC_MODELS.keys())}, registered={get_all_model_info()}"
    )
    raise ValueError(f"Unknown model_id: {model_id}")


def get_model(model_id="best"):
    """Get or create a model instance for the given model_id using centralized cache."""
    app.logger.debug(f"get_model called with model_id: {model_id}")

    model_path = _resolve_model_path(model_id)
    app.logger.debug(f"Resolved model {model_id} -> {model_path}")
    return MODEL_CACHE.get_simple_model(model_path)

def register_dynamic_model(model_id: str, model_path: str):
    """Register a dynamically selected model."""
    DYNAMIC_MODELS[model_id] = model_path
    app.logger.info(f"Registered dynamic model {model_id} -> {model_path}")

def get_available_models():
    """Return list of available model configurations."""
    # Get all models from central registry
    all_models = get_all_model_info()
    
    # Filter to only show the main model IDs (best, model2) to avoid duplicates
    # since best/previous_best point to the same files
    main_models = [model for model in all_models if model['id'] in ['best', 'model2']]
    
    # Add 'name' field that the frontend expects
    for model in main_models:
        # Create a user-friendly name from the filename
        model["name"] = f"Model {model['id']} ({model['filename']})"
    
    # Add dynamic models
    for model_id, model_path in DYNAMIC_MODELS.items():
        filename = os.path.basename(model_path)
        main_models.append({
            "id": model_id,
            "name": f"Dynamic ({filename})",
            "path": model_path
        })
    
    return main_models

def get_cached_model_wrapper(model_id: str):
    """Get or create a cached ModelWrapper instance for the given model_id using centralized cache."""
    app.logger.debug(f"get_cached_model_wrapper called with model_id: {model_id}")

    model_path = _resolve_model_path(model_id)
    app.logger.debug(f"Getting ModelWrapper for path: {model_path}")
    return MODEL_CACHE.get_wrapper_model(model_path)

def clear_model_wrapper_cache():
    """Clear the model cache to free memory."""
    MODEL_CACHE.clear_cache()
    app.logger.info("Model cache cleared")
    return 0  # Return 0 since we don't track individual cache sizes anymore

def create_game_state_from_trmph(trmph, context=""):
    """Create game state from normalized TRMPH input."""
    return core_create_game_state_from_trmph_input(trmph, context=context)


def _safe_count_trmph_moves(trmph_text):
    try:
        return fc.count_trmph_moves(trmph_text)
    except Exception:
        return None


def _is_pie_rule_opening_window(state, trmph, pie_rule_enabled):
    """True when pie-rule opening selection should run for this position."""
    if not pie_rule_enabled:
        return False
    if state.game_over:
        return False
    if _safe_count_trmph_moves(trmph) != 0:
        return False
    return state.current_player_enum == Player.BLUE


def _pie_rule_opening_weight_from_probability(
    opening_win_prob,
    *,
    exponent=PIE_RULE_OPENING_WEIGHT_EXPONENT,
):
    """Weight for pie-rule opening sampling based on closeness to 50%."""
    p = max(0.0, min(1.0, float(opening_win_prob)))
    symmetry_distance = min(p, 1.0 - p)
    if symmetry_distance <= 0.0:
        return 0.0
    return float(symmetry_distance ** exponent)


def _select_pie_rule_balanced_opening_move(state, trmph, model_id, pie_rule_enabled):
    """Sample opening move with weights favoring value estimates near 50%."""
    if not _is_pie_rule_opening_window(state, trmph, pie_rule_enabled):
        return None

    model = get_model(model_id)
    heatmap = build_policy_value_heatmap(
        state=state,
        model=model,
        selection_mode="all_legal",
        top_k=None,
        policy_temperature=1.0,
    )

    all_candidates = []
    weighted_candidates = []
    for move, score in sorted(heatmap.scores.items()):
        opening_win_prob = float(score)
        distance = abs(opening_win_prob - 0.5)
        weight = _pie_rule_opening_weight_from_probability(opening_win_prob)
        candidate = {
            "move": move,
            "opening_win_probability": opening_win_prob,
            "distance_to_even": distance,
            "weight": weight,
        }
        all_candidates.append(candidate)
        if weight > 0.0:
            weighted_candidates.append(candidate)

    if not all_candidates:
        return None

    if weighted_candidates:
        selected = random.choices(
            weighted_candidates,
            weights=[c["weight"] for c in weighted_candidates],
            k=1,
        )[0]
        sampling_total_weight = float(sum(c["weight"] for c in weighted_candidates))
        sampling_mode = "weighted"
    else:
        selected = random.choice(all_candidates)
        sampling_total_weight = 0.0
        sampling_mode = "uniform_zero_weights"

    return {
        "move": selected["move"],
        "opening_win_probability": selected["opening_win_probability"],
        "distance_to_even": selected["distance_to_even"],
        "weight": selected["weight"],
        "weight_exponent": PIE_RULE_OPENING_WEIGHT_EXPONENT,
        "sampling_mode": sampling_mode,
        "candidate_count": len(all_candidates),
        "weighted_candidate_count": len(weighted_candidates),
        "sampling_total_weight": sampling_total_weight,
    }


def _build_pie_rule_balanced_opening_response(
    *,
    state,
    selected_move_trmph,
    opening_win_probability,
    distance_to_even,
    opening_weight,
    weight_exponent,
    sampling_mode,
    candidate_count,
    weighted_candidate_count,
    sampling_total_weight,
    model_id,
    num_simulations,
    exploration_constant,
    temperature,
    temperature_end,
    enable_gumbel,
    gumbel_max_sims,
):
    """Build move response for pie-rule balanced opening without running MCTS."""
    new_state = _apply_selected_move(state, selected_move_trmph)
    result = build_engine_move_response(
        new_state,
        new_trmph=new_state.to_trmph(),
        move_made=selected_move_trmph,
        additional_fields={
            "pie_rule_enabled": True,
            "pie_rule_action": "balanced_opening",
            "pie_rule_opening_move": selected_move_trmph,
            "pie_rule_opening_win_probability": opening_win_probability,
            "pie_rule_opening_distance_to_even": distance_to_even,
            "pie_rule_opening_weight": opening_weight,
            "pie_rule_opening_weight_exponent": weight_exponent,
            "pie_rule_opening_sampling_mode": sampling_mode,
            "pie_rule_opening_candidate_count": candidate_count,
            "pie_rule_opening_weighted_candidate_count": weighted_candidate_count,
            "pie_rule_opening_sampling_total_weight": sampling_total_weight,
        },
    )
    result["mcts_config"] = {
        "model": model_id,
        "num_simulations": num_simulations,
        "exploration_constant": exploration_constant,
        "temperature": temperature,
        "temperature_end": temperature_end,
        "enable_gumbel": enable_gumbel,
        "gumbel_max_sims": gumbel_max_sims,
        "algorithm": "mcts",
    }
    return result


def build_game_response(state, model_id, temperature, trmph_for_inference=None, additional_fields=None):
    """Build standardized state payload for dev endpoints."""
    model = get_model(model_id)
    return build_game_state_response(
        state,
        model=model,
        temperature=temperature,
        trmph_for_inference=trmph_for_inference,
        additional_fields=additional_fields,
    )


ENGINE_NUMERIC_DEBUG_FIELDS = {
    "search_time",
    "total_compute_ms",
    "encode_ms",
    "forward_ms",
    "expand_ms",
    "backprop_ms",
    "batch_count",
    "cache_hits",
    "cache_misses",
    "root_value",
    "best_child_value",
    "win_probability",
    "best_child_win_probability",
    "pv_length",
    "search_efficiency",
    "mcts_probability",
    "direct_probability",
    "difference",
    "total_visits",
    "total_nodes",
    "max_depth",
    "inferences",
    "unique_evals_total",
    "effective_sims_total",
    "unique_evals_per_sec",
    "effective_sims_per_sec",
    "deduplication_ratio",
    "efficiency_gain_percent",
}


def _coerce_verbose_level(verbose):
    """Parse verbose flag into an integer level."""
    try:
        return int(verbose)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid verbose value: {verbose}") from exc


def _build_game_over_engine_result(state, trmph, debug_field):
    """Return consistent game-over payload for engine move endpoints."""
    app.logger.info("Game is over, returning current state")
    result = build_engine_move_response(
        state,
        new_trmph=trmph,
        move_made=None,
        additional_fields={debug_field: {}},
    )
    app.logger.info(f"Returning early result: {result}")
    return result


def _attach_inline_heatmap_if_requested(
    *,
    result,
    inline_heatmap_options,
    model_id,
    state=None,
):
    """Attach inline move heatmap when requested and the move response succeeded."""
    if not result.get("success"):
        return
    if not inline_heatmap_options.get("enabled"):
        return

    try:
        heatmap_state = state
        if heatmap_state is None:
            new_trmph = result.get("new_trmph")
            if not isinstance(new_trmph, str):
                app.logger.warning(
                    "Skipping inline move heatmap attachment: result missing new_trmph"
                )
                return
            heatmap_state = create_game_state_from_trmph(
                new_trmph,
                context="for inline move heatmap",
            )

        maybe_attach_inline_move_heatmap(
            result=result,
            state=heatmap_state,
            model_id=model_id,
            heatmap_options=inline_heatmap_options,
            model_getter=get_model,
            logger=app.logger,
        )
    except Exception as exc:
        app.logger.warning("Inline move heatmap attachment failed: %s", exc)
        result["move_heatmap_error"] = "Failed to compute move heatmap"


def _load_model_or_error(model_id):
    """Load a model and return a response payload on failure."""
    try:
        model = get_model(model_id)
    except Exception as e:
        app.logger.error(f"Failed to get model {model_id}: {e}")
        return None, build_engine_error_payload(
            f"Model loading failed: {e}",
            reason="model_load_failed",
        )

    app.logger.info(f"Model loaded successfully: {type(model).__name__}")
    return model, None


def _build_legal_move_probabilities(state, policy_probs):
    """Build mapping from legal TRMPH moves to policy probabilities."""
    legal_move_probs = {}
    for move in state.get_legal_moves():
        move_trmph = fc.rowcol_to_trmph(*move)
        tensor_idx = fc.rowcol_to_tensor(*move)
        if 0 <= tensor_idx < len(policy_probs):
            legal_move_probs[move_trmph] = float(policy_probs[tensor_idx])
    return legal_move_probs


def _compute_direct_policy_analysis(state, model, trmph, temperature):
    """Compute direct policy probabilities for legal moves and log summary."""
    app.logger.info("Getting direct policy comparison...")
    policy_start = time.time()
    policy_logits, value_output = model.simple_infer(trmph)
    policy_time = time.time() - policy_start
    app.logger.info(f"Direct policy inference took {policy_time:.3f}s")

    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    app.logger.info(f"Policy logits shape: {policy_logits.shape}, value_output: {value_output}")

    legal_moves = state.get_legal_moves()
    original_legal_moves_count = len(legal_moves)
    app.logger.info(f"Legal moves count: {original_legal_moves_count}")

    legal_move_probs = _build_legal_move_probabilities(state, policy_probs)
    sorted_moves = sorted(legal_move_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    top_moves_str = {move: f"{prob:.3f}" for move, prob in sorted_moves}
    app.logger.info(f"Top 10 legal move probabilities: {top_moves_str}")

    return legal_move_probs, original_legal_moves_count


def _apply_selected_move(state, selected_move_trmph):
    """Apply selected move with consistent logging."""
    app.logger.info(f"Applying move: {selected_move_trmph}")
    new_state = apply_move_to_state_trmph(state, selected_move_trmph)
    app.logger.info(f"Move applied. New state game_over: {new_state.game_over}")
    return new_state


def _sanitize_numeric_debug_fields(obj, path=""):
    """Replace unsupported numeric debug values with 0.0 for frontend safety."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if key in ENGINE_NUMERIC_DEBUG_FIELDS:
                if value is None:
                    app.logger.warning(
                        f"Found None value in numeric field {current_path}, replacing with 0.0"
                    )
                    obj[key] = 0.0
                elif isinstance(value, str):
                    app.logger.warning(
                        f"Found string value '{value}' in numeric field {current_path}, replacing with 0.0"
                    )
                    obj[key] = 0.0
            _sanitize_numeric_debug_fields(value, current_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _sanitize_numeric_debug_fields(item, f"{path}[{i}]")


def _resolve_mcts_algorithm_label(stats, algorithm_termination_info):
    """Resolve human-readable algorithm label for diagnostics."""
    if algorithm_termination_info:
        if algorithm_termination_info.reason == "terminal_move":
            return "Terminal Move Detection"
        if algorithm_termination_info.reason == "neural_network_confidence":
            return "Confidence-Based Termination"
        return "MCTS (Algorithm Termination)"

    return "MCTS (Gumbel on)" if stats.get("gumbel_candidates_m", 0) > 0 else "MCTS (Gumbel off)"


def _build_mcts_debug_info(
    *,
    stats,
    tree_data,
    algorithm_termination_info,
    selected_move_trmph,
    selected_move,
    legal_move_probs,
    original_legal_moves_count,
    num_simulations,
    exploration_constant,
    temperature,
    temperature_end,
    enable_gumbel,
    gumbel_max_sims,
    mcts_search_time,
):
    """Assemble MCTS diagnostics payload."""
    algorithm = _resolve_mcts_algorithm_label(stats, algorithm_termination_info)
    root_win_prob = compute_win_probability_from_tree_data(tree_data)
    best_child_signed_value = tree_data["v_ptm_ref_signed_best_child"]
    best_child_win_prob = compute_best_child_win_probability_from_tree_data(tree_data)
    algorithm_win_prob = (
        algorithm_termination_info.win_probability if algorithm_termination_info else None
    )
    temperature_scaled_mcts_probs = tree_data.get("temperature_scaled_probabilities", {})

    app.logger.debug(
        f"Value conversion: root_prob={root_win_prob:.3f}, "
        f"best_child_signed={best_child_signed_value:.3f} -> best_child_prob={best_child_win_prob:.3f}"
    )
    if algorithm_termination_info:
        app.logger.debug(
            f"Algorithm termination: reason={algorithm_termination_info.reason}, "
            f"termination_win_probability={algorithm_win_prob}"
        )

    mcts_debug_info = {
        "algorithm_info": {
            "algorithm": algorithm,
            "early_termination": algorithm_termination_info is not None,
            "early_termination_reason": algorithm_termination_info.reason if algorithm_termination_info else "none",
            "early_termination_details": {
                "reason": algorithm_termination_info.reason if algorithm_termination_info else "none",
                "win_probability": algorithm_win_prob,
                "move": algorithm_termination_info.move if algorithm_termination_info else None,
            },
            "parameters": {
                "simulations": num_simulations,
                "exploration_constant": exploration_constant,
                "temperature": temperature,
                "temperature_end": temperature_end,
                "gumbel_enabled": enable_gumbel,
                "gumbel_max_sims": gumbel_max_sims,
            },
        },
        "search_stats": {
            "num_simulations": num_simulations if not algorithm_termination_info else 0,
            "search_time": mcts_search_time,
            "exploration_constant": exploration_constant,
            "temperature": temperature,
            "mcts_stats": stats,
            "inferences": tree_data.get("inferences", 0) if not algorithm_termination_info else 0,
            "algorithm_used": algorithm,
        },
        "tree_statistics": {
            "total_visits": tree_data["total_visits"] if not algorithm_termination_info else 0,
            "total_nodes": tree_data["total_nodes"] if not algorithm_termination_info else 0,
            "max_depth": tree_data["max_depth"] if not algorithm_termination_info else 0,
            "inferences": tree_data["inferences"] if not algorithm_termination_info else 0,
            "algorithm_note": "No tree search performed" if algorithm_termination_info else "Full MCTS tree search",
        },
        "move_selection": {
            "selected_move": selected_move_trmph,
            "selected_move_coords": selected_move,
        },
        "move_probabilities": {
            "direct_policy": legal_move_probs,
            "mcts_visits": tree_data["visit_counts"],
            "mcts_probabilities": tree_data["mcts_probabilities"],
            "mcts_temperature_scaled_probabilities": temperature_scaled_mcts_probs,
        },
        "comparison": {
            "mcts_vs_direct": {},
        },
        "win_rate_analysis": {
            "root_value": tree_data["v_ptm_ref_signed_root"],
            "best_child_value": tree_data["v_ptm_ref_signed_best_child"],
            "win_probability": root_win_prob,
            "best_child_win_probability": best_child_win_prob,
        },
        "move_sequence_analysis": {
            "principal_variation": [fc.rowcol_to_trmph(*move) for move in tree_data["principal_variation"]],
            "alternative_lines": [],
            "pv_length": len(tree_data["principal_variation"]),
        },
        "gumbel_analysis": {
            "gumbel_used": stats.get("gumbel_candidates_m", 0) > 0,
            "gumbel_candidates_m": stats.get("gumbel_candidates_m", 0),
            "gumbel_rounds_R": stats.get("gumbel_rounds_R", 0),
            "gumbel_nn_calls_per_move": stats.get("gumbel_nn_calls_per_move", 0),
            "gumbel_total_leaves_evaluated": stats.get("gumbel_total_leaves_evaluated", 0),
            "gumbel_distinct_leaves_evaluated": stats.get("gumbel_distinct_leaves_evaluated", 0),
            "gumbel_avg_nn_batch_size": stats.get("gumbel_avg_nn_batch_size", 0.0),
            "gumbel_leaves_distinct_ratio": stats.get("gumbel_leaves_distinct_ratio", 0.0),
            "gumbel_selected_tensor_action": stats.get("gumbel_selected_tensor_action", None),
            "gumbel_v_pi_01": stats.get("gumbel_v_pi_01", None),
            "gumbel_final_rank_top_move": stats.get("gumbel_final_rank_top_move", None),
            "gumbel_final_rank_top5": stats.get("gumbel_final_rank_top5", None),
            "gumbel_selection_note": (
                "Gumbel final selection uses log(prior) + c_scale*(q - v_pi) over a candidate set; "
                "it is not the visit-count argmax."
            ),
        },
        "summary": {
            "selected_move": selected_move_trmph,
            "top_direct_move": max(legal_move_probs.items(), key=lambda x: x[1])[0] if legal_move_probs else None,
            "top_mcts_move": (
                max(tree_data["mcts_probabilities"].items(), key=lambda x: x[1])[0]
                if tree_data["mcts_probabilities"]
                else None
            ),
            "top_mcts_move_temperature_scaled": (
                max(temperature_scaled_mcts_probs.items(), key=lambda x: x[1])[0]
                if temperature_scaled_mcts_probs
                else None
            ),
            "top_mcts_move_raw_visits": (
                max(tree_data["visit_counts"].items(), key=lambda x: x[1])[0]
                if tree_data.get("visit_counts")
                else None
            ),
            "total_legal_moves": original_legal_moves_count,
            "moves_explored": tree_data["total_visits"] if not algorithm_termination_info else None,
            "search_efficiency": (
                tree_data["inferences"] / max(1, tree_data["total_visits"])
                if not algorithm_termination_info
                else 0.0
            ),
            "algorithm_summary": algorithm,
            "gumbel_summary": (
                f"Gumbel: {'ON' if stats.get('gumbel_candidates_m', 0) > 0 else 'OFF'} "
                f"(candidates: {stats.get('gumbel_candidates_m', 0)}, rounds: {stats.get('gumbel_rounds_R', 0)})"
            ),
            "move_selection_explanation": (
                "Selected Move is the move played (may be sampled). Top MCTS Move (Raw Visits) is the "
                "visit-count argmax. Top MCTS Move (Temperature Scaled) is the argmax after temperature "
                "scaling (closer to move selection)."
            ),
        },
        "profiling_summary": {
            "total_compute_ms": int(mcts_search_time * 1000.0),
            "encode_ms": stats.get("encode_ms", 0),
            "stack_ms": stats.get("stack_ms", 0),
            "forward_ms": stats.get("forward_ms", 0),
            "pure_forward_ms": stats.get("pure_forward_ms", 0),
            "terminal_detect_ms": stats.get("terminal_detect_ms", 0),
            "puct_calc_ms": stats.get("puct_calc_ms", 0),
            "sync_ms": stats.get("sync_ms", 0),
            "d2h_ms": stats.get("d2h_ms", 0),
            "expand_ms": stats.get("expand_ms", 0),
            "backprop_ms": stats.get("backprop_ms", 0),
            "select_ms": stats.get("select_ms", 0),
            "cache_lookup_ms": stats.get("cache_lookup_ms", 0),
            "state_creation_ms": stats.get("state_creation_ms", 0),
            "batch_count": stats.get("batch_count", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
            "simulations_per_second": stats.get("simulations_per_second", 0),
            "median_forward_ms": stats.get("median_forward_ms_ex_warm", 0),
            "median_select_ms": stats.get("median_select_ms", 0),
            "median_cache_hit_ms": stats.get("median_cache_hit_ms", 0),
            "median_cache_miss_ms": stats.get("median_cache_miss_ms", 0),
            "unique_evals_total": stats.get("unique_evals_total", 0),
            "effective_sims_total": stats.get("effective_sims_total", 0),
            "unique_evals_per_sec": stats.get("unique_evals_per_sec", 0),
            "effective_sims_per_sec": stats.get("effective_sims_per_sec", 0),
            "deduplication_ratio": stats.get("unique_evals_total", 0) / max(1, stats.get("effective_sims_total", 1)),
            "efficiency_gain_percent": (
                1.0 - (stats.get("unique_evals_total", 0) / max(1, stats.get("effective_sims_total", 1)))
            ) * 100,
        },
    }

    mcts_probs = tree_data["mcts_probabilities"]
    for move_trmph, direct_prob in legal_move_probs.items():
        mcts_prob = mcts_probs.get(move_trmph, 0.0)
        temp_scaled_prob = temperature_scaled_mcts_probs.get(move_trmph, 0.0)
        mcts_debug_info["comparison"]["mcts_vs_direct"][move_trmph] = {
            "direct_probability": direct_prob,
            "mcts_probability": mcts_prob,
            "mcts_temperature_scaled_probability": temp_scaled_prob,
            "difference": mcts_prob - direct_prob,
        }
    return mcts_debug_info


def _log_mcts_timing_summary(stats, mcts_search_time):
    """Emit summary logs for MCTS performance."""
    forward_percentage = (
        (stats.get("forward_ms", 0) / mcts_search_time / 10) if mcts_search_time > 0 else 0
    )
    app.logger.info("=== PERFORMANCE SUMMARY ===")
    app.logger.info(f"Forward pass uses: {forward_percentage:.1f}% of total time")
    app.logger.info(
        f"Cache efficiency: {stats.get('cache_hits', 0)} hits, {stats.get('cache_misses', 0)} misses"
    )
    app.logger.info(
        "Batch efficiency: %s batches, avg size %.1f",
        stats.get("batch_count", 0),
        sum(stats.get("batch_sizes", [0])) / max(1, len(stats.get("batch_sizes", []))),
    )
    app.logger.info(f"Simulations per second: {stats.get('simulations_per_second', 0):.1f}")
    app.logger.info("=== END PERFORMANCE SUMMARY ===")


def make_mcts_move(trmph, model_id, num_simulations, exploration_constant,
                   temperature, temperature_end, verbose, enable_gumbel, gumbel_max_sims):
    """Make one computer move using MCTS and return the new state with diagnostics."""
    try:
        mcts_verbose = _coerce_verbose_level(verbose)
        app.logger.info("=== MCTS MOVE START ===")
        app.logger.info(
            "Input: model_id=%s, sims=%s, temp=%s->%s, verbose=%s, gumbel=%s, gumbel_max_sims=%s",
            model_id,
            num_simulations,
            temperature,
            temperature_end,
            mcts_verbose,
            enable_gumbel,
            gumbel_max_sims,
        )
        app.logger.info(f"Input TRMPH: {trmph}")

        state = create_game_state_from_trmph(trmph, context="for MCTS move")
        app.logger.info(
            f"Game state created: game_over={state.game_over}, current_player={state.current_player_enum}"
        )
        if state.game_over:
            return _build_game_over_engine_result(state, trmph, "mcts_debug_info")

        model, error_result = _load_model_or_error(model_id)
        if error_result:
            return error_result

        mcts_config, temperature_end = create_interactive_mcts_config(
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            temperature=temperature,
            temperature_end=temperature_end,
            enable_gumbel=enable_gumbel,
            gumbel_max_sims=gumbel_max_sims,
            confidence_termination_threshold=INTERACTIVE_CONFIDENCE_TERMINATION_THRESHOLD,
            logger=app.logger,
        )
        app.logger.info(f"MCTS config created: {mcts_config}")

        app.logger.info(f"Getting cached ModelWrapper for model_id={model_id}")
        model_wrapper_start = time.time()
        model_wrapper = get_cached_model_wrapper(model_id)
        model_wrapper_time = time.time() - model_wrapper_start
        app.logger.info(f"ModelWrapper retrieval took {model_wrapper_time:.3f}s")

        app.logger.info("Starting MCTS search...")
        total_start_time = time.time()
        mcts_start_time = time.time()
        app.logger.info("About to run interactive MCTS search...")
        selected_move, stats, tree_data, algorithm_termination_info = run_interactive_mcts_search(
            state=state,
            model_wrapper=model_wrapper,
            mcts_config=mcts_config,
            verbose=mcts_verbose,
            logger=app.logger,
        )
        app.logger.info("run_mcts_move completed successfully")
        mcts_search_time = time.time() - mcts_start_time

        app.logger.debug("=== DETAILED TIMING BREAKDOWN ===")
        app.logger.debug(f"MCTS search completed in {mcts_search_time:.3f}s")
        app.logger.debug(f"Total wall time so far: {time.time() - total_start_time:.3f}s")
        app.logger.debug(f"MCTS selected move: {selected_move}")
        app.logger.debug(f"Simulations per second: {stats.get('simulations_per_second', 0):.2f}")
        _log_mcts_timing_summary(stats, mcts_search_time)

        selected_move_trmph = fc.rowcol_to_trmph(*selected_move)
        app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")

        legal_move_probs, original_legal_moves_count = _compute_direct_policy_analysis(
            state, model, trmph, temperature
        )
        state = _apply_selected_move(state, selected_move_trmph)

        mcts_debug_info = _build_mcts_debug_info(
            stats=stats,
            tree_data=tree_data,
            algorithm_termination_info=algorithm_termination_info,
            selected_move_trmph=selected_move_trmph,
            selected_move=selected_move,
            legal_move_probs=legal_move_probs,
            original_legal_moves_count=original_legal_moves_count,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            temperature=temperature,
            temperature_end=temperature_end,
            enable_gumbel=enable_gumbel,
            gumbel_max_sims=gumbel_max_sims,
            mcts_search_time=mcts_search_time,
        )

        result = build_engine_move_response(
            state,
            new_trmph=state.to_trmph(),
            move_made=selected_move_trmph,
            additional_fields={
                "mcts_debug_info": mcts_debug_info,
                "tree_data": tree_data,
            },
        )
        _sanitize_numeric_debug_fields(result)

        total_wall_time = time.time() - total_start_time
        post_mcts_time = total_wall_time - mcts_search_time
        app.logger.debug("=== MCTS MOVE COMPLETE ===")
        app.logger.debug("=== WALL TIME BREAKDOWN ===")
        app.logger.debug(f"ModelWrapper retrieval: {model_wrapper_time:.3f}s")
        app.logger.debug(f"MCTS search time: {mcts_search_time:.3f}s")
        app.logger.debug(f"Post-MCTS processing: {post_mcts_time:.3f}s")
        app.logger.debug(f"TOTAL WALL TIME: {total_wall_time:.3f}s")
        app.logger.debug("=== END WALL TIME BREAKDOWN ===")
        app.logger.debug(f"Final result keys: {list(result.keys())}")
        app.logger.debug(f"Move made: {result['move_made']}")
        app.logger.debug(f"Game over: {result['game_over']}")
        app.logger.debug(f"Winner: {result['winner']}")

        if "tree_data" in result and "detailed_exploration" in result["tree_data"]:
            de = result["tree_data"]["detailed_exploration"]
            app.logger.debug(
                "Detailed exploration: enabled=%s, simulations=%s, trace_length=%s",
                de.get("enabled"),
                de.get("total_simulations"),
                len(de.get("trace", [])),
            )
        else:
            app.logger.debug("No detailed exploration data found in tree_data")

        try:
            response_json = json.dumps(result)
            response_size = len(response_json)
            app.logger.debug(
                f"Response JSON size: {response_size:,} bytes ({response_size/1024:.1f} KB)"
            )
        except Exception as e:
            app.logger.warning(f"Could not serialize response for size measurement: {e}")

        return result
    except Exception as e:
        app.logger.error("=== MCTS MOVE ERROR ===")
        app.logger.error(f"Error in make_mcts_move: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return build_engine_error_payload(
            f"MCTS move generation failed: {e}",
            reason="engine_failure",
        )


def _build_fixed_tree_debug_info(
    *,
    search_result,
    stats,
    search_widths,
    temperature,
    search_time,
    selected_move_trmph,
    legal_move_probs,
    original_legal_moves_count,
):
    """Assemble fixed-tree diagnostics payload."""
    early_termination_info = search_result.early_termination_info

    def _early_term_field(field_name, default=None):
        if early_termination_info is None:
            return default
        if isinstance(early_termination_info, dict):
            return early_termination_info.get(field_name, default)
        return getattr(early_termination_info, field_name, default)

    algorithm = "Fixed Tree Search"
    fixed_tree_debug_info = {
        "algorithm_info": {
            "algorithm": algorithm,
            "early_termination": early_termination_info is not None,
            "early_termination_reason": _early_term_field("reason", "none"),
            "early_termination_details": {
                "reason": _early_term_field("reason", "none"),
                "win_probability": _early_term_field("win_probability", None),
                "move": _early_term_field("move", None),
            },
            "parameters": {
                "search_widths": search_widths,
                "temperature": temperature,
            },
        },
        "search_stats": {
            "search_time": search_time,
            "search_widths": search_widths,
            "temperature": temperature,
            "fixed_tree_stats": stats,
            "algorithm_used": algorithm,
        },
        "tree_statistics": {
            "total_positions": stats.get("total_positions", 0),
            "tree_depth": stats.get("tree_depth", 0),
            "tree_width": stats.get("tree_width", 0),
            "policy_evaluations": stats.get("policy_evaluations", 0),
            "value_evaluations": stats.get("value_evaluations", 0),
            "early_terminations": stats.get("early_terminations", 0),
        },
        "move_selection": {
            "selected_move": selected_move_trmph,
            "selected_move_coords": search_result.move,
        },
        "move_probabilities": {
            "direct_policy": legal_move_probs,
        },
        "comparison": {
            "fixed_tree_vs_direct": {},
        },
        "win_rate_analysis": {
            "root_value": search_result.value,
            "win_probability": search_result.win_probability,
        },
        "summary": {
            "top_direct_move": max(legal_move_probs.items(), key=lambda x: x[1])[0] if legal_move_probs else None,
            "total_legal_moves": original_legal_moves_count,
            "moves_explored": f"{stats.get('total_positions', 0)}/{original_legal_moves_count}",
            "search_efficiency": stats.get("total_positions", 0) / max(1, original_legal_moves_count),
            "algorithm_summary": algorithm,
        },
        "profiling_summary": {
            "total_compute_ms": int(search_time * 1000.0),
            "search_time_ms": int(search_time * 1000.0),
            "memory_usage_mb": stats.get("memory_usage_mb", 0.0),
            "tree_building_time_ms": int(stats.get("tree_building_time", 0.0) * 1000.0),
            "leaf_evaluation_time_ms": int(stats.get("leaf_evaluation_time", 0.0) * 1000.0),
            "backup_time_ms": int(stats.get("backup_time", 0.0) * 1000.0),
            "policy_nn_time_ms": int(stats.get("policy_nn_time", 0.0) * 1000.0),
            "value_nn_time_ms": int(stats.get("value_nn_time", 0.0) * 1000.0),
        },
    }

    for move_trmph, direct_prob in legal_move_probs.items():
        fixed_tree_debug_info["comparison"]["fixed_tree_vs_direct"][move_trmph] = {
            "direct_probability": direct_prob,
            "fixed_tree_selected": move_trmph == selected_move_trmph,
        }
    return fixed_tree_debug_info


def make_fixed_tree_move(trmph, model_id, search_widths, temperature, verbose):
    """Make one computer move using Fixed Tree Search and return the new state with diagnostics."""
    try:
        app.logger.info("=== FIXED TREE MOVE START ===")
        app.logger.info(
            f"Input: model_id={model_id}, search_widths={search_widths}, temp={temperature}, verbose={verbose}"
        )
        app.logger.info(f"Input TRMPH: {trmph}")

        state = create_game_state_from_trmph(trmph, context="for Fixed Tree move")
        app.logger.info(
            f"Game state created: game_over={state.game_over}, current_player={state.current_player_enum}"
        )
        if state.game_over:
            return _build_game_over_engine_result(state, trmph, "fixed_tree_debug_info")

        model, error_result = _load_model_or_error(model_id)
        if error_result:
            return error_result

        search_config = create_fixed_tree_config(
            search_widths=search_widths,
            temperature=temperature,
            batch_size=1000,
            # Interactive fixed-tree should run the requested tree search rather
            # than short-circuiting to a policy-only pick from value confidence.
            enable_early_termination=False,
            early_termination_threshold=0.95,
        )
        app.logger.info(f"Fixed tree config created: {search_config}")

        app.logger.info("Starting Fixed Tree Search...")
        total_start_time = time.time()
        search_start_time = time.time()
        app.logger.info("About to call run_fixed_tree_search...")
        try:
            search_result = run_fixed_tree_search(state, model, search_config, verbose)
            app.logger.info("run_fixed_tree_search completed successfully")
        except Exception as e:
            app.logger.error(f"run_fixed_tree_search failed with exception: {e}")
            import traceback
            app.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        search_time = time.time() - search_start_time
        stats = search_result.stats

        app.logger.debug("=== DETAILED TIMING BREAKDOWN ===")
        app.logger.debug(f"Fixed tree search completed in {search_time:.3f}s")
        app.logger.debug(f"Total wall time so far: {time.time() - total_start_time:.3f}s")
        app.logger.debug(f"Fixed tree selected move: {search_result.move}")
        app.logger.debug(f"Total positions: {stats.get('total_positions', 0)}")
        app.logger.debug(f"Policy evaluations: {stats.get('policy_evaluations', 0)}")
        app.logger.debug(f"Value evaluations: {stats.get('value_evaluations', 0)}")
        app.logger.debug(f"Tree depth: {stats.get('tree_depth', 0)}")
        app.logger.debug(f"Tree width: {stats.get('tree_width', 0)}")
        app.logger.debug(f"Memory usage: {stats.get('memory_usage_mb', 0):.1f}MB")

        app.logger.info("=== PERFORMANCE SUMMARY ===")
        app.logger.info(f"Total positions evaluated: {stats.get('total_positions', 0)}")
        app.logger.info(f"Search time: {search_time:.3f}s")
        app.logger.info(f"Tree depth: {stats.get('tree_depth', 0)}, max width: {stats.get('tree_width', 0)}")
        app.logger.info(f"Memory usage: {stats.get('memory_usage_mb', 0):.1f}MB")
        app.logger.info("=== END PERFORMANCE SUMMARY ===")

        selected_move_trmph = fc.rowcol_to_trmph(*search_result.move)
        app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")

        legal_move_probs, original_legal_moves_count = _compute_direct_policy_analysis(
            state, model, trmph, temperature
        )
        state = _apply_selected_move(state, selected_move_trmph)

        fixed_tree_debug_info = _build_fixed_tree_debug_info(
            search_result=search_result,
            stats=stats,
            search_widths=search_widths,
            temperature=temperature,
            search_time=search_time,
            selected_move_trmph=selected_move_trmph,
            legal_move_probs=legal_move_probs,
            original_legal_moves_count=original_legal_moves_count,
        )

        result_data = build_engine_move_response(
            state,
            new_trmph=state.to_trmph(),
            move_made=selected_move_trmph,
            additional_fields={
                "fixed_tree_debug_info": fixed_tree_debug_info,
                "tree_data": search_result.tree_data,
            },
        )
        _sanitize_numeric_debug_fields(result_data)

        total_wall_time = time.time() - total_start_time
        post_search_time = total_wall_time - search_time
        app.logger.debug("=== FIXED TREE MOVE COMPLETE ===")
        app.logger.debug("=== WALL TIME BREAKDOWN ===")
        app.logger.debug(f"Fixed tree search time: {search_time:.3f}s")
        app.logger.debug(f"Post-search processing: {post_search_time:.3f}s")
        app.logger.debug(f"TOTAL WALL TIME: {total_wall_time:.3f}s")
        app.logger.debug("=== END WALL TIME BREAKDOWN ===")
        app.logger.debug(f"Final result keys: {list(result_data.keys())}")
        app.logger.debug(f"Move made: {result_data['move_made']}")
        app.logger.debug(f"Game over: {result_data['game_over']}")
        app.logger.debug(f"Winner: {result_data['winner']}")
        return result_data
    except Exception as e:
        app.logger.error("=== FIXED TREE MOVE ERROR ===")
        app.logger.error(f"Error in make_fixed_tree_move: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return build_engine_error_payload(
            f"Fixed tree move generation failed: {e}",
            reason="engine_failure",
        )


@app.route("/api/constants", methods=["GET"])
def api_constants():
    """Return game constants for frontend use."""
    return jsonify({
        "BOARD_SIZE": BOARD_SIZE,
        "PIECE_VALUES": {
            "EMPTY": Piece.EMPTY.value,
            "BLUE": Piece.BLUE.value,
            "RED": Piece.RED.value
        },
        "PLAYER_VALUES": {
            "BLUE": Player.BLUE.value,
            "RED": Player.RED.value,
        },
        "WINNER_VALUES": {
            "BLUE": TRMPH_BLUE_WIN,
            "RED": TRMPH_RED_WIN
        },
        "FIXED_TREE": {
            "MAX_PRODUCT": FIXED_TREE_MAX_PRODUCT,
            "DEFAULT_WIDTH": FIXED_TREE_DEFAULT_WIDTH,
            "DEFAULT_TEMPERATURE": FIXED_TREE_DEFAULT_TEMPERATURE
        }
    })

@app.route("/api/models", methods=["GET"])
def api_models():
    """Get available models."""
    return jsonify({"models": get_available_models()})

@app.route("/api/model-browser/recent", methods=["GET"])
def api_recent_models():
    """Get recently used models."""
    try:
        recent_models = MODEL_BROWSER.get_recent_models()
        return jsonify({"recent_models": recent_models})
    except Exception as e:
        app.logger.error(f"Error getting recent models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-browser/directories", methods=["GET"])
def api_model_directories():
    """Get all directories containing models."""
    try:
        directories = MODEL_BROWSER.get_directories()
        return jsonify({"directories": directories})
    except Exception as e:
        app.logger.error(f"Error getting directories: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-browser/directory/<path:directory>", methods=["GET"])
def api_models_in_directory(directory):
    """Get all models in a specific directory."""
    try:
        models = MODEL_BROWSER.get_models_in_directory(directory)
        return jsonify({"models": models})
    except Exception as e:
        app.logger.error(f"Error getting models in directory {directory}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-browser/search", methods=["GET"])
def api_search_models():
    """Search models by query."""
    query = request.args.get("q", "")
    try:
        models = MODEL_BROWSER.search_models(query)
        return jsonify({"models": models})
    except Exception as e:
        app.logger.error(f"Error searching models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/cache/clear", methods=["POST"])
def api_clear_cache():
    """Clear the ModelWrapper cache."""
    try:
        cache_size = clear_model_wrapper_cache()
        return jsonify({"success": True, "cleared_instances": cache_size})
    except Exception as e:
        app.logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/cache/status", methods=["GET"])
def api_cache_status():
    """Get cache status."""
    try:
        # Get cache statistics from the centralized cache
        # Note: The centralized cache doesn't expose individual model keys,
        # so we return basic cache information
        return jsonify({
            "cache_type": "centralized_model_cache",
            "cache_status": "active",
            "note": "Using centralized ModelCache - individual model tracking not available"
        })
    except Exception as e:
        app.logger.error(f"Error getting cache status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-browser/validate", methods=["POST"])
def api_validate_model():
    """Validate a model path."""
    data = request.get_json()
    model_path = data.get("model_path")
    
    if not model_path:
        return jsonify({"error": "model_path required"}), 400
    
    try:
        validation = MODEL_BROWSER.validate_model(model_path)
        return jsonify(validation)
    except Exception as e:
        app.logger.error(f"Error validating model {model_path}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-browser/select", methods=["POST"])
def api_select_model():
    """Select a model and add it to recent models."""
    data = request.get_json()
    model_path = data.get("model_path")
    model_id = data.get("model_id")  # Optional: client can specify model_id
    
    app.logger.debug(f"api_select_model called with data: {data}")
    
    if not model_path:
        app.logger.error("No model_path provided")
        return jsonify({"error": "model_path required"}), 400
    
    try:
        app.logger.debug(f"Validating model path: {model_path}")
        # Validate the model
        validation = MODEL_BROWSER.validate_model(model_path)
        app.logger.debug(f"Validation result: {validation}")
        
        if not validation['valid']:
            app.logger.error(f"Model validation failed: {validation['error']}")
            return jsonify({"success": False, "error": validation['error']}), 400
        
        # Generate model_id if not provided
        if not model_id:
            model_id = f"model_{int(datetime.now().timestamp())}"
        
        app.logger.debug(f"Using model_id: {model_id}")
        app.logger.debug(f"Model path: {model_path}")
        
        # Register the dynamic model
        register_dynamic_model(model_id, model_path)
        app.logger.debug(f"Registered dynamic model. Current DYNAMIC_MODELS: {DYNAMIC_MODELS}")
        
        # Add to recent models
        add_recent_model(model_path)
        
        # Test loading the model immediately to catch any issues
        app.logger.debug("Testing model loading...")
        try:
            test_model = get_model(model_id)
            app.logger.debug(f"Model loading test successful: {type(test_model)}")
        except Exception as e:
            app.logger.error(f"Model loading test failed: {e}")
            # Remove from dynamic models if loading fails
            if model_id in DYNAMIC_MODELS:
                del DYNAMIC_MODELS[model_id]
            return jsonify({"success": False, "error": f"Model loading test failed: {e}"}), 500
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "model_path": model_path,
            "model_info": validation
        })
        
    except Exception as e:
        app.logger.error(f"Error selecting model {model_path}: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/model-browser/refresh", methods=["POST"])
def api_refresh_models():
    """Force refresh of model cache."""
    try:
        models = MODEL_BROWSER.get_all_models(force_refresh=True)
        return jsonify({"models": models, "count": len(models)})
    except Exception as e:
        app.logger.error(f"Error refreshing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json()
    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=["trmph"],
        optional_fields=["model_id", "temperature", "verbose"],
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    trmph = validated_data.get("trmph")
    model_id = validated_data.get("model_id", "best")
    temperature = validated_data.get("temperature", 1.0)
    verbose = validated_data.get("verbose", 0)
    try:
        verbose = _coerce_verbose_level(verbose)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    try:
        state = create_game_state_from_trmph(trmph, context="for state endpoint")
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400

    response = build_game_response(
        state,
        model_id,
        temperature,
        trmph_for_inference=trmph,
        additional_fields={"trmph": trmph},
    )

    # Debug logging for board data (only if verbose >= 4)
    if verbose >= 4:
        board = response.get("board", [])
        app.logger.debug(f"Board data being sent to frontend: {board}")
        app.logger.debug(f"Board type: {type(board)}, Board shape: {len(board)}x{len(board[0]) if board else 0}")
        if board and len(board) > 0 and len(board[0]) > 0:
            app.logger.debug(f"Sample board values: [0,0]='{board[0][0]}', [0,1]='{board[0][1]}', [1,0]='{board[1][0]}'")
    return jsonify(response)


@app.route("/api/move_heatmap", methods=["POST"])
def api_move_heatmap():
    """
    Return value-head win-rate heatmap for every legal next move.

    Scores are always from the *current player to move* perspective.
    """
    data = request.get_json()
    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=["trmph"],
        optional_fields=["model_id", "score_type", "selection_mode", "top_k", "policy_temperature"],
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    trmph = validated_data.get("trmph")
    try:
        parsed_heatmap_params = parse_move_heatmap_request_params(
            validated_data,
            default_model_id="best",
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    model_id = parsed_heatmap_params["model_id"]
    score_type = parsed_heatmap_params["score_type"]
    selection_mode = parsed_heatmap_params["selection_mode"]
    top_k = parsed_heatmap_params["top_k"]
    policy_temperature = parsed_heatmap_params["policy_temperature"]

    try:
        state = create_game_state_from_trmph(trmph, context="for move heatmap")
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid TRMPH: {e}"}), 400

    try:
        model = get_model(model_id)
        heatmap = build_policy_value_heatmap(
            state=state,
            model=model,
            selection_mode=selection_mode,
            top_k=top_k,
            policy_temperature=policy_temperature,
        )
        response = {
            "success": True,
            "trmph": trmph,
            "model_id": model_id,
            "score_type": score_type,
        }
        response.update(heatmap.to_dict())
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in api_move_heatmap: {e}")
        return jsonify({"success": False, "error": f"Failed to compute move heatmap: {e}"}), 500

@app.route("/api/apply_move", methods=["POST"])
def api_apply_move():
    """Apply only a human move without making a computer move."""
    data = request.get_json()
    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=["trmph", "move"],
        optional_fields=["model_id", "temperature", "verbose"],
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    trmph = validated_data.get("trmph")
    move = validated_data.get("move")
    model_id = validated_data.get("model_id", "best")
    temperature = validated_data.get("temperature", 1.0)
    verbose = validated_data.get("verbose", 0)
    try:
        verbose = _coerce_verbose_level(verbose)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    try:
        state = create_game_state_from_trmph(trmph, context="for apply_move")
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400
    
    try:
        state = apply_move_to_state_trmph(state, move)
    except Exception as e:
        return jsonify({"error": f"Invalid move: {e}"}), 400

    new_trmph = state.to_trmph()
    response = build_game_response(
        state,
        model_id,
        temperature,
        trmph_for_inference=new_trmph,
        additional_fields={
            "new_trmph": new_trmph,
            "model_move": None,
        },
    )

    # Debug logging for board data (only if verbose >= 4)
    if verbose >= 4:
        board = response.get("board", [])
        app.logger.debug(f"Apply move - Board data being sent to frontend: {board}")
        app.logger.debug(f"Apply move - Board data type: {type(board)}, Board shape: {len(board[0]) if board else 0}")
        if board and len(board) > 0 and len(board[0]) > 0:
            app.logger.debug(f"Apply move - Sample board values: [0,0]='{board[0][0]}', [0,1]='{board[0][1]}', [1,0]='{board[1][0]}'")
    return jsonify(response)

@app.route("/api/apply_trmph_sequence", methods=["POST"])
def api_apply_trmph_sequence():
    """Apply a sequence of TRMPH moves to the board state."""
    data = request.get_json()
    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=["trmph"],
        optional_fields=["trmph_sequence", "model_id", "temperature", "verbose"],
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    trmph = validated_data.get("trmph")
    trmph_sequence = validated_data.get("trmph_sequence", "")
    model_id = validated_data.get("model_id", "best")
    temperature = validated_data.get("temperature", 1.0)
    verbose = validated_data.get("verbose", 0)
    try:
        verbose = _coerce_verbose_level(verbose)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not trmph_sequence.strip():
        return jsonify({"error": "No TRMPH sequence provided"}), 400
    
    try:
        # Start with the current state
        state = create_game_state_from_trmph(trmph, context="for apply_trmph_sequence")
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400
    
    try:
        state, _, moves_applied = apply_trmph_sequence_to_state(state, trmph_sequence)
    except ValueError as e:
        return jsonify({"error": f"Invalid TRMPH sequence format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH sequence: {e}"}), 400

    new_trmph = state.to_trmph()
    response = build_game_response(
        state,
        model_id,
        temperature,
        trmph_for_inference=new_trmph,
        additional_fields={
            "new_trmph": new_trmph,
            "moves_applied": moves_applied,
            "game_over": state.game_over,
        },
    )

    if verbose >= 4:
        board = response.get("board", [])
        app.logger.debug(f"Apply sequence - Board data being sent to frontend: {board}")
    return jsonify(response)


def _validate_engine_request(
    data,
    *,
    required_fields,
    optional_fields,
    supports_inline_heatmap=True,
):
    """Validate engine endpoint payloads with one shared code path."""
    def _strict_validate_api_input(payload, *, required_fields=None, optional_fields=None):
        return validate_api_input(
            payload,
            required_fields=required_fields,
            optional_fields=optional_fields,
            reject_unexpected=True,
        )

    if supports_inline_heatmap:
        validated_data, inline_heatmap_options, error_msg = validate_request_with_inline_heatmap(
            data,
            required_fields=required_fields,
            optional_fields=optional_fields,
            validate_api_input_fn=_strict_validate_api_input,
        )
        if error_msg:
            return None, None, (
                jsonify(
                    build_engine_error_payload(
                        error_msg,
                        reason="validation_error",
                    )
                ),
                400,
            )
        return validated_data, inline_heatmap_options, None

    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=required_fields,
        optional_fields=optional_fields,
        reject_unexpected=True,
    )
    if not is_valid:
        return None, {"enabled": False}, (
            jsonify(
                build_engine_error_payload(
                    error_msg,
                    reason="validation_error",
                )
            ),
            400,
        )

    return validated_data, {"enabled": False}, None


def _validate_fixed_tree_search_widths(search_widths):
    """
    Validate fixed-tree search widths and return a normalized integer list.

    Enforces positive integer widths and the configured product cap.
    """
    if not isinstance(search_widths, (list, tuple)) or not search_widths:
        raise ValueError("search_widths must be a non-empty array of positive integers")

    normalized_widths = []
    product = 1

    for idx, width in enumerate(search_widths):
        if isinstance(width, bool):
            raise ValueError(f"search_widths[{idx}] must be a positive integer")

        try:
            width_float = float(width)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search_widths[{idx}] must be a positive integer") from exc

        if not np.isfinite(width_float) or not width_float.is_integer():
            raise ValueError(f"search_widths[{idx}] must be a positive integer")

        width_int = int(width_float)
        if width_int <= 0:
            raise ValueError(f"search_widths[{idx}] must be > 0")

        normalized_widths.append(width_int)
        product *= width_int
        if product > FIXED_TREE_MAX_PRODUCT:
            raise ValueError(
                f"Product of search widths ({product}) exceeds limit of {FIXED_TREE_MAX_PRODUCT}"
            )

    return normalized_widths



@app.route("/api/policy_move", methods=["POST"])
def api_policy_move():
    """Make a computer move using policy sampling."""
    data = request.get_json()
    app.logger.info("=== POLICY API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    # Validate input
    validated_data, inline_heatmap_options, error_response = _validate_engine_request(
        data,
        required_fields=["trmph"],
        optional_fields=["model_id", "temperature", "verbose"],
    )
    if error_response:
        return error_response

    trmph = validated_data.get("trmph")
    model_id = validated_data.get("model_id", "best")
    temperature = validated_data.get("temperature", 0.15)  # Default policy temperature
    verbose = validated_data.get("verbose", 0)
    try:
        verbose = _coerce_verbose_level(verbose)
    except ValueError as e:
        return jsonify(
            build_engine_error_payload(
                str(e),
                reason="validation_error",
            )
        ), 400
    
    app.logger.info(f"Parsed parameters: trmph={trmph[:50]}..., model_id={model_id}, temp={temperature}, verbose={verbose}")
    
    try:
        # Parse TRMPH and get game state
        state = create_game_state_from_trmph(trmph, context="for policy move")
        
        # Get model and make policy move
        model = get_model(model_id)
        move = select_policy_move(state, model, temperature)
        
        if move is None:
            return jsonify(
                build_engine_error_payload(
                    "No valid moves available",
                    reason="no_valid_moves",
                )
            ), 400
        
        # Apply the move
        move_trmph = fc.rowcol_to_trmph(move[0], move[1])
        new_state = apply_move_to_state_trmph(state, move_trmph)
        new_trmph = new_state.to_trmph()
        
        # Get policy information for debug output
        policy_logits, value_signed = model.simple_infer(trmph)
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        legal_moves = state.get_legal_moves()
        legal_policy = get_legal_policy_probs(policy_probs, legal_moves, state.board.shape[0])
        
        # Create policy dictionary for debug
        policy_dict = {}
        for i, (row, col) in enumerate(legal_moves):
            trmph_move = fc.rowcol_to_trmph(row, col)
            policy_dict[trmph_move] = float(legal_policy[i])
        
        # Sort by probability for debug output
        sorted_policy = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)
        
        result = build_engine_move_response(
            new_state,
            new_trmph=new_trmph,
            move_made=move_trmph,
            additional_fields={
                "policy_info": {
                    "selected_move": move_trmph,
                    "selected_probability": policy_dict.get(move_trmph, 0.0),
                    "top_moves": sorted_policy[:5],  # Top 5 moves for debug
                    "temperature": temperature,
                },
            },
        )
        
        if verbose >= 1:
            result["debug_info"] = {
                "algorithm_info": {
                    "algorithm": "policy",
                    "parameters": {
                        "temperature": temperature
                    }
                },
                "policy_analysis": {
                    "top_moves": sorted_policy[:10],
                    "total_legal_moves": len(legal_moves)
                }
            }

        _attach_inline_heatmap_if_requested(
            result=result,
            inline_heatmap_options=inline_heatmap_options,
            model_id=model_id,
            state=new_state,
        )
        
        app.logger.info("=== POLICY API RESPONSE ===")
        app.logger.info(f"Selected move: {move_trmph} (prob: {policy_dict.get(move_trmph, 0.0):.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Policy move error: {e}")
        return jsonify(
            build_engine_error_payload(
                str(e),
                reason="engine_failure",
            )
        ), 500

@app.route("/api/mcts_move", methods=["POST"])
def api_mcts_move():
    """Make a computer move using MCTS with diagnostic output."""
    data = request.get_json()
    app.logger.info("=== MCTS API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    # Validate input
    validated_data, inline_heatmap_options, error_response = _validate_engine_request(
        data,
        required_fields=["trmph"],
        optional_fields=[
            "model_id",
            "num_simulations",
            "exploration_constant",
            "temperature",
            "temperature_end",
            "verbose",
            "enable_gumbel",
            "gumbel_max_sims",
            "pie_rule_enabled",
        ],
    )
    if error_response:
        return error_response

    trmph = validated_data.get("trmph")
    model_id = validated_data.get("model_id", "best")
    num_simulations = validated_data.get("num_simulations", 200)
    exploration_constant = validated_data.get("exploration_constant", 2.8)
    temperature = validated_data.get("temperature", 1.0)
    temperature_end = validated_data.get("temperature_end", 0.1)  # Default final temperature
    verbose = validated_data.get("verbose", 0)
    try:
        verbose = _coerce_verbose_level(verbose)
    except ValueError as e:
        return jsonify(
            build_engine_error_payload(
                str(e),
                reason="validation_error",
            )
        ), 400
    enable_gumbel = validated_data.get("enable_gumbel", True)
    gumbel_max_sims = validated_data.get("gumbel_max_sims", 500)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    
    app.logger.info(f"Parsed parameters: trmph={trmph[:50]}..., model_id={model_id}, sims={num_simulations}, temp={temperature}->{temperature_end}, verbose={verbose}, gumbel={enable_gumbel}, gumbel_max_sims={gumbel_max_sims}")

    opening_choice = None
    opening_state = None
    try:
        opening_state = create_game_state_from_trmph(
            trmph,
            context="for pie-rule balanced opening",
        )
        opening_choice = _select_pie_rule_balanced_opening_move(
            opening_state,
            trmph,
            model_id,
            pie_rule_enabled,
        )
    except Exception as e:
        app.logger.warning("Pie-rule opening selection failed; falling back to MCTS: %s", e)

    if opening_choice:
        app.logger.info(
            "Pie-rule opening selected (MCTS bypass) move=%s opening_p=%.4f distance_to_even=%.4f weight=%.8f mode=%s",
            opening_choice["move"],
            opening_choice["opening_win_probability"],
            opening_choice["distance_to_even"],
            opening_choice["weight"],
            opening_choice["sampling_mode"],
        )
        result = _build_pie_rule_balanced_opening_response(
            state=opening_state,
            selected_move_trmph=opening_choice["move"],
            opening_win_probability=opening_choice["opening_win_probability"],
            distance_to_even=opening_choice["distance_to_even"],
            opening_weight=opening_choice["weight"],
            weight_exponent=opening_choice["weight_exponent"],
            sampling_mode=opening_choice["sampling_mode"],
            candidate_count=opening_choice["candidate_count"],
            weighted_candidate_count=opening_choice["weighted_candidate_count"],
            sampling_total_weight=opening_choice["sampling_total_weight"],
            model_id=model_id,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            temperature=temperature,
            temperature_end=temperature_end,
            enable_gumbel=enable_gumbel,
            gumbel_max_sims=gumbel_max_sims,
        )
        _attach_inline_heatmap_if_requested(
            result=result,
            inline_heatmap_options=inline_heatmap_options,
            model_id=model_id,
        )
        return jsonify(result), 200
    
    result = make_mcts_move(
        trmph,
        model_id,
        num_simulations,
        exploration_constant,
        temperature,
        temperature_end,
        verbose,
        enable_gumbel=enable_gumbel,
        gumbel_max_sims=gumbel_max_sims
    )

    _attach_inline_heatmap_if_requested(
        result=result,
        inline_heatmap_options=inline_heatmap_options,
        model_id=model_id,
    )
    
    app.logger.info("=== MCTS API RESPONSE ===")
    app.logger.info(f"Result success: {result.get('success', 'MISSING')}")
    if result.get('success'):
        app.logger.info(f"Result keys: {list(result.keys())}")
        app.logger.info(f"Move made: {result.get('move_made', 'MISSING')}")
        app.logger.info(f"Game over: {result.get('game_over', 'MISSING')}")
        app.logger.info(f"Winner: {result.get('winner', 'MISSING')}")
        # Log a few key numeric values that might be causing the toFixed error
        if 'mcts_debug_info' in result:
            debug_info = result['mcts_debug_info']
            if 'profiling_summary' in debug_info:
                profiling = debug_info['profiling_summary']
                app.logger.info(f"Profiling values: {profiling}")
    else:
        app.logger.error(f"Result error: {result.get('error', 'MISSING')}")
    
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code

@app.route("/api/fixed_tree_move", methods=["POST"])
def api_fixed_tree_move():
    """Make a computer move using Fixed Tree Search with diagnostic output."""
    data = request.get_json()
    app.logger.info("=== FIXED TREE API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    validated_data, inline_heatmap_options, error_response = _validate_engine_request(
        data,
        required_fields=["trmph", "search_widths"],
        optional_fields=["model_id", "temperature", "verbose"],
    )
    if error_response:
        return error_response

    trmph = validated_data.get("trmph")
    model_id = validated_data.get("model_id", "best")
    search_widths = validated_data.get("search_widths")
    temperature = validated_data.get("temperature", FIXED_TREE_DEFAULT_TEMPERATURE)
    verbose = validated_data.get("verbose", 0)
    try:
        verbose = _coerce_verbose_level(verbose)
    except ValueError as e:
        return jsonify(
            build_engine_error_payload(
                str(e),
                reason="validation_error",
            )
        ), 400
    
    app.logger.info(f"Parsed parameters: trmph={trmph[:50]}..., model_id={model_id}, search_widths={search_widths}, temp={temperature}, verbose={verbose}")
    
    try:
        search_widths = _validate_fixed_tree_search_widths(search_widths)
    except ValueError as e:
        app.logger.error(f"Error validating search_widths: {e}")
        return jsonify(
            build_engine_error_payload(
                str(e),
                reason="validation_error",
            )
        ), 400
    
    result = make_fixed_tree_move(
        trmph,
        model_id,
        search_widths,
        temperature,
        verbose
    )

    _attach_inline_heatmap_if_requested(
        result=result,
        inline_heatmap_options=inline_heatmap_options,
        model_id=model_id,
    )
    
    app.logger.info("=== FIXED TREE API RESPONSE ===")
    app.logger.info(f"Result success: {result.get('success', 'MISSING')}")
    if result.get('success'):
        app.logger.info(f"Result keys: {list(result.keys())}")
        app.logger.info(f"Move made: {result.get('move_made', 'MISSING')}")
        app.logger.info(f"Game over: {result.get('game_over', 'MISSING')}")
        app.logger.info(f"Winner: {result.get('winner', 'MISSING')}")
        # Log a few key numeric values that might be causing issues
        if 'fixed_tree_debug_info' in result:
            debug_info = result['fixed_tree_debug_info']
            if 'profiling_summary' in debug_info:
                profiling = debug_info['profiling_summary']
                app.logger.info(f"Profiling values: {profiling}")
    else:
        app.logger.error(f"Result error: {result.get('error', 'MISSING')}")
    
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code

@app.route("/api/save_game", methods=["POST"])
def api_save_game():
    """Save a game to the web_games directory in TRMPH format."""
    data = request.get_json()
    app.logger.info("=== SAVE GAME API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=["trmph"],
        optional_fields=["winner", "model_id", "mcts_params"],
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    trmph = validated_data.get("trmph")
    winner = validated_data.get("winner")
    model_id = validated_data.get("model_id", "best")
    mcts_params = validated_data.get("mcts_params", {})
    
    if not trmph:
        return jsonify({"success": False, "error": "TRMPH sequence is required"}), 400
    
    try:
        # Create web_games directory if it doesn't exist
        web_games_dir = os.path.join("data", "web_games")
        os.makedirs(web_games_dir, exist_ok=True)
        
        # Determine winner for TRMPH format
        trmph_winner = None
        if winner:
            if winner.lower() == "blue":
                trmph_winner = TRMPH_BLUE_WIN
            elif winner.lower() == "red":
                trmph_winner = TRMPH_RED_WIN
            else:
                return jsonify({"success": False, "error": f"Invalid winner: {winner}"}), 400
        else:
            # Game not finished, need user input
            return jsonify({
                "success": False, 
                "needs_winner_input": True,
                "message": "Game is not finished. Please specify the winner."
            }), 400
        
        # Save to TRMPH file
        trmph_file = os.path.join(web_games_dir, "web_games.trmph")
        with open(trmph_file, 'a') as f:
            f.write(f"{trmph} {trmph_winner}\n")
        
        # Save metadata to log file
        log_file = os.path.join(web_games_dir, "web_games.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get model info
        model_info = {}
        if model_id in DYNAMIC_MODELS:
            model_info["path"] = DYNAMIC_MODELS[model_id]
        elif is_valid_model_id(model_id):
            model_info["path"] = get_model_path(model_id)
        else:
            model_info["path"] = "unknown"
        
        log_entry = {
            "timestamp": timestamp,
            "trmph": trmph,
            "winner": winner,
            "model_id": model_id,
            "model_path": model_info["path"],
            "mcts_params": mcts_params
        }
        
        with open(log_file, 'a') as f:
            f.write(f"{log_entry}\n")
        
        app.logger.info(f"Game saved successfully to {trmph_file}")
        app.logger.info(f"Metadata logged to {log_file}")
        
        return jsonify({
            "success": True,
            "message": f"Game saved successfully. Winner: {winner}",
            "trmph_file": trmph_file,
            "log_file": log_file
        })
        
    except Exception as e:
        app.logger.error(f"Error saving game: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Failed to save game: {e}"}), 500

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(os.path.dirname(__file__)), "favicon3_cropped.png")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), path)

@app.route("/shared/<path:path>")
def serve_shared(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static_shared"), path)

@app.route("/")
def serve_index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), "index.html")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hex AI Web Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable Flask debug mode')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='Disable Flask debug mode')
    parser.set_defaults(debug=None)
    args = parser.parse_args()

    if args.debug is None:
        debug_enabled = os.getenv("SF25_DEV_WEB_DEBUG", "0").lower() not in ("0", "false", "no")
    else:
        debug_enabled = args.debug

    default_log_level = "DEBUG" if debug_enabled else "INFO"
    log_level_name = os.getenv("SF25_DEV_WEB_LOG_LEVEL", default_log_level).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    logging.basicConfig(level=log_level)
    app.logger.info("=" * 50)
    app.logger.info("Hex AI Web Server Starting...")
    app.logger.info("Debug mode: %s", debug_enabled)
    app.logger.info("=" * 50)
    app.run(
        debug=debug_enabled,
        use_reloader=False,
        use_debugger=debug_enabled,
        threaded=False,
        host=args.host,
        port=args.port,
    ) 
