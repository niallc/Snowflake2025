from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import numpy as np
from flask_cors import CORS
import logging
from typing import Tuple
from datetime import datetime
import time # Added for time.time()


from hex_ai.utils import format_conversion as fc
from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.inference.simple_model_inference import SimpleModelInference

from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, run_mcts_move, create_mcts_config, TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.value_utils import Winner, winner_to_color, get_policy_probs_from_logits, temperature_scaled_softmax, ValuePredictor
from hex_ai.enums import Player
from hex_ai.value_utils import (
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    select_policy_move,
    signed_to_prob,
)
from hex_ai.inference.mcts_utils import compute_win_probability_from_tree_data
from hex_ai.config import BOARD_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.enums import Piece
from hex_ai.inference.game_engine import apply_move_to_state_trmph
from hex_ai.web.model_browser import create_model_browser
from hex_ai.file_utils import add_recent_model
from hex_ai.inference.model_config import get_model_path, get_model_info, get_all_model_info, register_model, is_valid_model_id, get_normalized_path
from hex_ai.inference.model_cache import get_model_cache
from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN

app = Flask(__name__, static_folder="static")
CORS(app)

# API contract for player fields
# - player: UI-friendly color string ("blue"|"red")
# - player_enum: canonical enum name ("BLUE"|"RED")
# - player_index: canonical numeric (0=BLUE, 1=RED)
# - player_raw: DEPRECATED; remove after frontend migrates

# TODO: PERFORMANCE INVESTIGATION - MCTS vs Fixed Tree Search Performance Gap
# Fixed tree search: ~6 games/sec with depth 2, ~100 leaf nodes
# Current MCTS: <1 move/sec despite batching ~64 evaluations
# This represents a ~100x slowdown that needs systematic investigation
# Key areas: state copying, tree traversal overhead, batch utilization, model call efficiency

# NOTE: Value head terminology - We use 'value_signed' as a shorthand for [-1, 1] scores
# returned by the value head (tanh activated) and used by MCTS, as opposed to 'value_logits'
# which were the old sigmoid-based outputs.

# TODO: Refactor duplicated logic across API functions
# The following functions share common patterns that should be extracted into utilities:
# - api_state(), api_apply_move(), api_apply_trmph_sequence(), api_move()
# Common patterns: TRMPH parsing/validation, player color conversion, model inference, response construction

# TODO: Additional refactoring opportunities:
# 1. ✅ Create centralized ModelLoader utility (duplicated in get_model(), simple_model_inference.py, model_wrapper.py) - FIXED
# 2. Consolidate temperature scaling logic (duplicated in batched_mcts.py, mcts.py, value_utils.py)
# 3. Create centralized move selection utility (duplicated in batched_mcts.py, mcts.py, web/app.py)
# 4. Create base MCTS node class (duplicated between BatchedMCTSNode and MCTSNode)
# 5. Break up complex functions: search() in batched_mcts.py, play_single_game() in tournament.py

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

# --- Model Management ---
def get_model(model_id="model1"):
    """Get or create a model instance for the given model_id using centralized cache."""
    app.logger.debug(f"get_model called with model_id: {model_id}")
    
    # Check if it's a dynamic model (user-selected) - these override the central registry
    if model_id in DYNAMIC_MODELS:
        model_path = DYNAMIC_MODELS[model_id]
        app.logger.debug(f"Found dynamic model {model_id} -> {model_path}")
        
        # Handle relative paths by prepending checkpoints directory
        if not os.path.isabs(model_path):
            full_model_path = os.path.join("checkpoints", model_path)
            app.logger.debug(f"Converted relative path to: {full_model_path}")
        else:
            full_model_path = model_path
        
        # Check if file exists before loading
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file does not exist: {full_model_path}")
        
        app.logger.debug(f"Loading dynamic model {model_id} from {full_model_path}")
        return MODEL_CACHE.get_simple_model(full_model_path)
    
    # Use centralized model configuration
    if is_valid_model_id(model_id):
        model_path = get_model_path(model_id)
        app.logger.debug(f"Found model {model_id} -> {model_path}")
        
        # Centralized config provides absolute paths, so use directly
        return MODEL_CACHE.get_simple_model(model_path)
    
    # If model_id is a direct path, try to load it
    if os.path.exists(model_id):
        app.logger.debug(f"Model_id appears to be a direct path: {model_id}")
        return MODEL_CACHE.get_simple_model(model_id)
    
    app.logger.error(f"Unknown model_id: {model_id}")
    app.logger.error(f"Available options: dynamic={list(DYNAMIC_MODELS.keys())}, registered={get_all_model_info()}")
    raise ValueError(f"Unknown model_id: {model_id}")

def register_dynamic_model(model_id: str, model_path: str):
    """Register a dynamically selected model."""
    global DYNAMIC_MODELS
    DYNAMIC_MODELS[model_id] = model_path
    app.logger.info(f"Registered dynamic model {model_id} -> {model_path}")

def get_available_models():
    """Return list of available model configurations."""
    # Get all models from central registry
    all_models = get_all_model_info()
    
    # Filter to only show the main model IDs (model1, model2) to avoid duplicates
    # since current_best/previous_best point to the same files
    main_models = [model for model in all_models if model['id'] in ['model1', 'model2']]
    
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
    
    # Get the model path for this model_id
    if model_id in DYNAMIC_MODELS:
        model_path = DYNAMIC_MODELS[model_id]
        if not os.path.isabs(model_path):
            model_path = os.path.join("checkpoints", model_path)
    elif is_valid_model_id(model_id):
        model_path = get_model_path(model_id)
    elif os.path.exists(model_id):
        model_path = model_id
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    
    app.logger.debug(f"Getting ModelWrapper for path: {model_path}")
    return MODEL_CACHE.get_wrapper_model(model_path)

def clear_model_wrapper_cache():
    """Clear the model cache to free memory."""
    MODEL_CACHE.clear_cache()
    app.logger.info("Model cache cleared")
    return 0  # Return 0 since we don't track individual cache sizes anymore

def generate_debug_info(state, model, policy_logits, value_signed, policy_probs,
                         temperature, verbose, model_move=None, model_id=None):
    """Generate comprehensive debug information based on verbose level."""
    debug_info = {}
    
    # Add algorithm identification
    debug_info["algorithm_info"] = {
        "algorithm": "Policy-Only",
        "early_termination": False,
        "early_termination_reason": "none",
        "parameters": {
            "temperature": temperature
        }
    }
    
    # Level 1: Basic policy and value analysis
    if verbose >= 1:
        # Add defensive programming to catch any issues
        try:
            current_player_enum = state.current_player_enum
            current_player_color = winner_to_color(current_player_enum)
            win_prob = ValuePredictor.get_win_probability(value_signed, current_player_enum)
        except Exception as e:
            # Log the error and provide fallback values
            app.logger.error(f"Error in app.py debug info generation: {e}")
            raise RuntimeError(f"Error in app.py debug info generation: {e}")
        
        debug_info["basic"] = {
            "current_player": current_player_color,
            "current_player_raw": state.current_player,  # Add raw value for debugging
            "game_over": state.game_over,
            "legal_moves_count": len(state.get_legal_moves()),
            "value_signed": float(value_signed),
            "win_probability": float(win_prob),
            "temperature": temperature,
            "model_move": model_move
        }
        
        # Add model information
        if model_id:
            debug_info["model_info"] = {
                "model_id": model_id,
                "model_type": type(model).__name__,
                "model_path": DYNAMIC_MODELS.get(model_id, get_model_path(model_id))
            }
        
        # Use existing utilities to get policy analysis
        legal_moves = state.get_legal_moves()
        
        # Get post-temperature scaling moves using existing utility
        move_probs = []
        for row, col in legal_moves:
            move_trmph = fc.rowcol_to_trmph(row, col)
            prob = float(policy_probs[row * state.board.shape[0] + col])
            move_probs.append({"move": move_trmph, "row": row, "col": col, "probability": prob})
        
        # Sort by probability descending
        move_probs.sort(key=lambda x: x["probability"], reverse=True)
        
        # Get pre-temperature scaling moves using existing utility
        raw_move_probs = []
        raw_policy_logits = policy_logits.flatten()
        for row, col in legal_moves:
            move_trmph = fc.rowcol_to_trmph(row, col)
            raw_prob = float(raw_policy_logits[row * state.board.shape[0] + col])
            raw_move_probs.append({"move": move_trmph, "row": row, "col": col, "raw_logit": raw_prob})
        
        # Sort raw probabilities by logit value descending
        raw_move_probs.sort(key=lambda x: x["raw_logit"], reverse=True)
        
        debug_info["policy_analysis"] = {
            "top_moves": move_probs[:10],  # Top 10 moves (post-temperature)
            "raw_top_moves": raw_move_probs[:10],  # Top 10 moves (pre-temperature)
            "total_legal_moves": len(legal_moves)
        }
    
    # Level 2: Detailed analysis (removed tree search since we only use MCTS now)
    if verbose >= 2:
        # No tree search analysis needed since we only use MCTS
        pass
    
    # Level 3: Full analysis (removed policy-value comparison since we only use MCTS now)
    if verbose >= 3:
        # No policy-value comparison needed since we only use MCTS
        pass
    
    return debug_info



# --- Utility: Convert (row, col) moves to trmph moves ---
def moves_to_trmph(moves):
    return [fc.rowcol_to_trmph(row, col) for row, col in moves]

# TODO: Remove this function
def _build_orchestration_from_dict(cfg: dict | None) -> None:
    """Legacy function - orchestration is now handled internally by BaselineMCTS."""
    return None


def make_mcts_move(trmph, model_id, num_simulations=200, exploration_constant=2.8, 
                   temperature=1.0, temperature_end=0.1, verbose=0, orchestration_overrides=None,
                   enable_gumbel=True, gumbel_max_sims=500):
    """Make one computer move using MCTS and return the new state with diagnostics."""
    try:
        app.logger.info(f"=== MCTS MOVE START ===")
        app.logger.info(f"Input: model_id={model_id}, sims={num_simulations}, temp={temperature}->{temperature_end}, verbose={verbose}, gumbel={enable_gumbel}, gumbel_max_sims={gumbel_max_sims}")
        app.logger.info(f"Input TRMPH: {trmph}")
        
        state = HexGameState.from_trmph(trmph)
        app.logger.info(f"Game state created: game_over={state.game_over}, current_player={state.current_player_enum}")
        
        # If game is over, return current state
        if state.game_over:
            app.logger.info("Game is over, returning current state")
            result = {
                "success": True,
                "new_trmph": trmph,
                "board": state.board.tolist(),
                "player": winner_to_color(state.current_player),
                "legal_moves": moves_to_trmph(state.get_legal_moves()),
                "winner": winner_to_color(state.winner) if state.winner is not None else None,
                "move_made": None,
                "game_over": True,
                "mcts_debug_info": {}
            }
            app.logger.info(f"Returning early result: {result}")
            return result
        
        # Get model
        try:
            model = get_model(model_id)
            app.logger.info(f"Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            app.logger.error(f"Failed to get model {model_id}: {e}")
            return {
                "success": False,
                "error": f"Model loading failed: {e}"
            }
        
        # Create MCTS configuration
        # Ensure temperature_end is <= temperature_start
        if temperature_end > temperature:
            app.logger.info(f"Adjusting temperature_end from {temperature_end} to {temperature/10} (temperature_start/10)")
            temperature_end = temperature / 10
        
        # Warn about very low temperatures that will use deterministic selection
        if temperature < 0.02:
            app.logger.info(f"Temperature {temperature} is very low (< 0.02), will use deterministic selection to avoid numerical issues")
        
        mcts_config = create_mcts_config(
            config_type="tournament",
            sims=num_simulations,
            c_puct=exploration_constant,
            temperature_start=temperature,
            temperature_end=temperature_end,
            enable_gumbel_root_selection=enable_gumbel,
            gumbel_sim_threshold=gumbel_max_sims
        )
        app.logger.info(f"MCTS config created: {mcts_config}")
        
        # Create game engine
        engine = HexGameEngine()
        app.logger.info("Game engine created")
        
        # Get cached model wrapper for MCTS (avoid expensive recreation)
        app.logger.info(f"Getting cached ModelWrapper for model_id={model_id}")
        model_wrapper_start = time.time()
        model_wrapper = get_cached_model_wrapper(model_id)
        model_wrapper_time = time.time() - model_wrapper_start
        app.logger.info(f"ModelWrapper retrieval took {model_wrapper_time:.3f}s")
        
        # Run MCTS search with comprehensive timing
        app.logger.info("Starting MCTS search...")
        total_start_time = time.time()
        
        # Time the actual MCTS run
        mcts_start_time = time.time()
        app.logger.info("About to call run_mcts_move...")
        try:
            move, stats, tree_data, algorithm_termination_info = run_mcts_move(engine, model_wrapper, state, mcts_config)
            app.logger.info("run_mcts_move completed successfully")
        except Exception as e:
            app.logger.error(f"run_mcts_move failed with exception: {e}")
            import traceback
            app.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        mcts_search_time = time.time() - mcts_start_time
        
        # Time the rest of the processing
        post_mcts_start = time.time()
        
        # Log detailed timing breakdown
        app.logger.info(f"=== DETAILED TIMING BREAKDOWN ===")
        app.logger.info(f"MCTS search completed in {mcts_search_time:.3f}s")
        app.logger.info(f"Total wall time so far: {time.time() - total_start_time:.3f}s")
        app.logger.info(f"MCTS selected move: {move}")
        
        app.logger.info(f"Simulations per second: {stats.get('simulations_per_second', 0):.2f}")
        app.logger.info(f"Forward pass (total): {stats.get('forward_ms', 0):.1f}ms ({stats.get('forward_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"  - Pure neural network: {stats.get('pure_forward_ms', 0):.1f}ms ({stats.get('pure_forward_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"  - Device sync: {stats.get('sync_ms', 0):.1f}ms ({stats.get('sync_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Selection: {stats.get('select_ms', 0):.1f}ms ({stats.get('select_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"  - Terminal move detection: {stats.get('terminal_detect_ms', 0):.1f}ms ({stats.get('terminal_detect_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"  - PUCT calculation: {stats.get('puct_calc_ms', 0):.1f}ms ({stats.get('puct_calc_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"State creation: {stats.get('state_creation_ms', 0):.1f}ms ({stats.get('state_creation_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Cache lookup: {stats.get('cache_lookup_ms', 0):.1f}ms ({stats.get('cache_lookup_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Encoding: {stats.get('encode_ms', 0):.1f}ms ({stats.get('encode_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Stacking: {stats.get('stack_ms', 0):.1f}ms ({stats.get('stack_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Host-to-device: {stats.get('h2d_ms', 0):.1f}ms ({stats.get('h2d_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Device-to-host: {stats.get('d2h_ms', 0):.1f}ms ({stats.get('d2h_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Expansion: {stats.get('expand_ms', 0):.1f}ms ({stats.get('expand_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Backpropagation: {stats.get('backprop_ms', 0):.1f}ms ({stats.get('backprop_ms', 0)/mcts_search_time/10:.1f}%)")
        app.logger.info(f"Cache hits: {stats.get('cache_hits', 0)}, misses: {stats.get('cache_misses', 0)}")
        app.logger.info(f"Batch count: {stats.get('batch_count', 0)}, avg batch size: {sum(stats.get('batch_sizes', [0]))/max(1, len(stats.get('batch_sizes', []))):.1f}")
        app.logger.info(f"Median forward time: {stats.get('median_forward_ms_ex_warm', 0):.1f}ms")
        app.logger.info(f"Median select time: {stats.get('median_select_ms', 0):.1f}ms")
        app.logger.info(f"Median terminal detect time: {stats.get('median_terminal_detect_ms', 0):.1f}ms")
        app.logger.info(f"Median PUCT calc time: {stats.get('median_puct_calc_ms', 0):.1f}ms")
        app.logger.info(f"=== END TIMING BREAKDOWN ===")
        
        # Add performance summary
        forward_percentage = (stats.get('forward_ms', 0) / mcts_search_time / 10) if mcts_search_time > 0 else 0
        app.logger.info(f"=== PERFORMANCE SUMMARY ===")
        app.logger.info(f"Forward pass dominates: {forward_percentage:.1f}% of total time")
        app.logger.info(f"Cache efficiency: {stats.get('cache_hits', 0)} hits, {stats.get('cache_misses', 0)} misses")
        app.logger.info(f"Batch efficiency: {stats.get('batch_count', 0)} batches, avg size {sum(stats.get('batch_sizes', [0]))/max(1, len(stats.get('batch_sizes', []))):.1f}")
        app.logger.info(f"Simulations per second: {stats.get('simulations_per_second', 0):.1f}")
        app.logger.info(f"=== END PERFORMANCE SUMMARY ===")
        
        selected_move_trmph = fc.rowcol_to_trmph(*move)
        app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")
        
        # Get direct policy comparison
        app.logger.info("Getting direct policy comparison...")
        policy_start = time.time()
        policy_logits, value_output = model.simple_infer(trmph)
        policy_time = time.time() - policy_start
        app.logger.info(f"Direct policy inference took {policy_time:.3f}s")
        
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        app.logger.info(f"Policy logits shape: {policy_logits.shape}, value_output: {value_output}")
        
        # Get legal moves count for summary
        legal_moves = state.get_legal_moves()
        original_legal_moves_count = len(legal_moves)
        app.logger.info(f"Legal moves count: {original_legal_moves_count}")
        
        # Get direct policy probabilities for comparison with MCTS
        legal_move_probs = {}
        for move in legal_moves:
            move_trmph = fc.rowcol_to_trmph(*move)
            tensor_idx = fc.rowcol_to_tensor(*move)
            if 0 <= tensor_idx < len(policy_probs):
                legal_move_probs[move_trmph] = float(policy_probs[tensor_idx])
        
        # Log top 10 legal move probabilities
        sorted_moves = sorted(legal_move_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_moves_str = {move: f"{prob:.3f}" for move, prob in sorted_moves}
        app.logger.info(f"Top 10 legal move probabilities: {top_moves_str}")
        
        # Apply the move
        app.logger.info(f"Applying move: {selected_move_trmph}")
        state = apply_move_to_state_trmph(state, selected_move_trmph)
        app.logger.info(f"Move applied. New state game_over: {state.game_over}")
        
        # Generate MCTS diagnostic info
        # Determine algorithm type based on algorithm termination info and Gumbel usage
        if algorithm_termination_info:
            if algorithm_termination_info.reason == "terminal_move":
                algorithm = "Terminal Move Detection"
            elif algorithm_termination_info.reason == "neural_network_confidence":
                algorithm = "Confidence-Based Termination"
            else:
                algorithm = "MCTS (Algorithm Termination)"
        else:
            # Check if Gumbel was used based on MCTS stats
            gumbel_used = stats.get('gumbel_candidates_m', 0) > 0
            if gumbel_used:
                algorithm = "MCTS (Gumbel on)"
            else:
                algorithm = "MCTS (Gumbel off)"
        
        # Use centralized utilities for value conversions and tree data
        
        # Get win probabilities using centralized utility
        root_win_prob = compute_win_probability_from_tree_data(tree_data)
        best_child_signed_value = tree_data.get("v_ptm_ref_signed_best_child", 0.0)
        best_child_win_prob = signed_to_prob(best_child_signed_value)
        
        # Handle algorithm termination win probability
        if algorithm_termination_info and algorithm_termination_info.win_prob is not None:
            if algorithm_termination_info.reason == "terminal_move":
                algorithm_win_prob = 1.0  # Already correct
            else:
                algorithm_win_prob = signed_to_prob(algorithm_termination_info.win_prob)
        else:
            algorithm_win_prob = None
        
        # Log the conversion for debugging
        app.logger.debug(f"Value conversion: root_prob={root_win_prob:.3f}, "
                        f"best_child_signed={best_child_signed_value:.3f} -> best_child_prob={best_child_win_prob:.3f}")
        if algorithm_termination_info:
            app.logger.debug(f"Algorithm termination: reason={algorithm_termination_info.reason}, "
                           f"original_win_prob={algorithm_termination_info.win_prob}, converted={algorithm_win_prob}")
        
        # Get temperature-scaled MCTS probabilities from tree data (calculated by MCTS core)
        temperature_scaled_mcts_probs = tree_data.get("temperature_scaled_probabilities", {})
        
        mcts_debug_info = {
            "algorithm_info": {
                "algorithm": algorithm,
                "early_termination": algorithm_termination_info is not None,
                "early_termination_reason": algorithm_termination_info.reason if algorithm_termination_info else "none",
                "early_termination_details": {
                    "reason": algorithm_termination_info.reason if algorithm_termination_info else "none",
                    "win_probability": algorithm_win_prob,  # Use converted probability
                    "move": algorithm_termination_info.move if algorithm_termination_info else None
                },
                "parameters": {
                    "simulations": num_simulations,
                    "exploration_constant": exploration_constant,
                    "temperature": temperature,
                    "temperature_end": temperature_end,
                    "gumbel_enabled": enable_gumbel,
                    "gumbel_max_sims": gumbel_max_sims
                }
            },
            "search_stats": {
                "num_simulations": num_simulations if not algorithm_termination_info else 0,
                "search_time": mcts_search_time,
                "exploration_constant": exploration_constant,
                "temperature": temperature,
                "mcts_stats": stats,
                "inferences": tree_data.get("inferences", 0) if not algorithm_termination_info else 0,
                "algorithm_used": algorithm
            },
            "tree_statistics": {
                "total_visits": tree_data["total_visits"] if not algorithm_termination_info else 0,
                "total_nodes": tree_data["total_nodes"] if not algorithm_termination_info else 0,
                "max_depth": tree_data["max_depth"] if not algorithm_termination_info else 0,
                "inferences": tree_data["inferences"] if not algorithm_termination_info else 0,
                "algorithm_note": "No tree search performed" if algorithm_termination_info else "Full MCTS tree search"
            },
            "move_selection": {
                "selected_move": selected_move_trmph,
                "selected_move_coords": move
            },
            "move_probabilities": {
                "direct_policy": legal_move_probs,
                "mcts_visits": tree_data["visit_counts"],
                "mcts_probabilities": tree_data["mcts_probabilities"],
                "mcts_temperature_scaled_probabilities": temperature_scaled_mcts_probs
            },
            "comparison": {
                "mcts_vs_direct": {}
            },
            "win_rate_analysis": {
                "root_value": tree_data["v_ptm_ref_signed_root"],
                "best_child_value": tree_data["v_ptm_ref_signed_best_child"],
                "win_probability": root_win_prob,
                "best_child_win_probability": best_child_win_prob
            },
            "move_sequence_analysis": {
                "principal_variation": [fc.rowcol_to_trmph(*move) for move in tree_data["principal_variation"]],
                "alternative_lines": [],  # Placeholder - would need more complex tree analysis
                "pv_length": len(tree_data["principal_variation"])
            },
            "gumbel_analysis": {
                "gumbel_used": stats.get('gumbel_candidates_m', 0) > 0,
                "gumbel_candidates_m": stats.get('gumbel_candidates_m', 0),
                "gumbel_rounds_R": stats.get('gumbel_rounds_R', 0),
                "gumbel_nn_calls_per_move": stats.get('gumbel_nn_calls_per_move', 0),
                "gumbel_total_leaves_evaluated": stats.get('gumbel_total_leaves_evaluated', 0),
                "gumbel_distinct_leaves_evaluated": stats.get('gumbel_distinct_leaves_evaluated', 0),
                "gumbel_avg_nn_batch_size": stats.get('gumbel_avg_nn_batch_size', 0.0),
                "gumbel_leaves_distinct_ratio": stats.get('gumbel_leaves_distinct_ratio', 0.0)
            },
            "summary": {
                "top_direct_move": max(legal_move_probs.items(), key=lambda x: x[1])[0] if legal_move_probs else None,
                "top_mcts_move": max(tree_data["mcts_probabilities"].items(), key=lambda x: x[1])[0] if tree_data["mcts_probabilities"] and len(tree_data["mcts_probabilities"]) > 0 else None,
                "top_mcts_move_temperature_scaled": max(temperature_scaled_mcts_probs.items(), key=lambda x: x[1])[0] if temperature_scaled_mcts_probs and len(temperature_scaled_mcts_probs) > 0 else None,
                "top_mcts_move_raw_visits": max(tree_data["mcts_probabilities"].items(), key=lambda x: x[1])[0] if tree_data["mcts_probabilities"] and len(tree_data["mcts_probabilities"]) > 0 else None,
                "total_legal_moves": original_legal_moves_count,
                "moves_explored": f"{tree_data['total_visits']}/{original_legal_moves_count}" if not algorithm_termination_info else "N/A (Algorithm Termination)",
                "search_efficiency": tree_data["inferences"] / max(1, tree_data["total_visits"]) if not algorithm_termination_info else 0.0,
                "algorithm_summary": algorithm,
                "gumbel_summary": f"Gumbel: {'ON' if stats.get('gumbel_candidates_m', 0) > 0 else 'OFF'} (candidates: {stats.get('gumbel_candidates_m', 0)}, rounds: {stats.get('gumbel_rounds_R', 0)})",
                "move_selection_explanation": "Top MCTS Move shows raw visit counts. Top MCTS Move (Temperature Scaled) shows what's actually used for move selection."
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
                # New MCTS performance metrics
                "unique_evals_total": stats.get("unique_evals_total", 0),
                "effective_sims_total": stats.get("effective_sims_total", 0),
                "unique_evals_per_sec": stats.get("unique_evals_per_sec", 0),
                "effective_sims_per_sec": stats.get("effective_sims_per_sec", 0),
                "deduplication_ratio": stats.get("unique_evals_total", 0) / max(1, stats.get("effective_sims_total", 1)),
                "efficiency_gain_percent": (1.0 - (stats.get("unique_evals_total", 0) / max(1, stats.get("effective_sims_total", 1)))) * 100,
            },
        }
        
        # Add comparison data
        mcts_probs = tree_data["mcts_probabilities"]
        for move_trmph in legal_move_probs:
            direct_prob = legal_move_probs.get(move_trmph, 0)
            mcts_prob = mcts_probs.get(move_trmph, 0.0)
            temp_scaled_prob = temperature_scaled_mcts_probs.get(move_trmph, 0.0)
            mcts_debug_info["comparison"]["mcts_vs_direct"][move_trmph] = {
                "direct_probability": direct_prob,
                "mcts_probability": mcts_prob,
                "mcts_temperature_scaled_probability": temp_scaled_prob,
                "difference": mcts_prob - direct_prob
            }
        
        result = {
            "success": True,
            "new_trmph": state.to_trmph(),
            "board": state.board.tolist(),
            "player": winner_to_color(state.current_player),
            "legal_moves": moves_to_trmph(state.get_legal_moves()),
            "winner": winner_to_color(state.winner) if state.winner is not None else None,
            "move_made": selected_move_trmph,
            "game_over": state.game_over,
            "mcts_debug_info": mcts_debug_info,
            "tree_data": tree_data
        }
        
        # Validate that no None values exist in numeric fields that frontend expects
        def validate_numeric_fields(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key in ['search_time', 'total_compute_ms', 'encode_ms', 'forward_ms', 'expand_ms', 'backprop_ms', 
                              'batch_count', 'cache_hits', 'cache_misses', 'root_value', 'best_child_value', 
                              'win_probability', 'best_child_win_probability', 'pv_length', 'search_efficiency',
                              'mcts_probability', 'direct_probability', 'difference', 'total_visits', 'total_nodes', 
                              'max_depth', 'inferences', 'unique_evals_total', 'effective_sims_total', 
                              'unique_evals_per_sec', 'effective_sims_per_sec', 'deduplication_ratio', 
                              'efficiency_gain_percent']:
                        if value is None:
                            app.logger.warning(f"Found None value in numeric field {current_path}, replacing with 0.0")
                            obj[key] = 0.0
                        elif isinstance(value, str):
                            app.logger.warning(f"Found string value '{value}' in numeric field {current_path}, replacing with 0.0")
                            obj[key] = 0.0
                    validate_numeric_fields(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    validate_numeric_fields(item, f"{path}[{i}]")
        
        validate_numeric_fields(result)
        
        # Time JSON serialization and response preparation
        json_start_time = time.time()
        
        # Calculate total wall time before JSON serialization
        total_wall_time = time.time() - total_start_time
        post_mcts_time = total_wall_time - mcts_search_time
        
        app.logger.info(f"=== MCTS MOVE COMPLETE ===")
        app.logger.info(f"=== WALL TIME BREAKDOWN ===")
        app.logger.info(f"ModelWrapper retrieval: {model_wrapper_time:.3f}s")
        app.logger.info(f"MCTS search time: {mcts_search_time:.3f}s")
        app.logger.info(f"Post-MCTS processing: {post_mcts_time:.3f}s")
        app.logger.info(f"TOTAL WALL TIME: {total_wall_time:.3f}s")
        app.logger.info(f"=== END WALL TIME BREAKDOWN ===")
        app.logger.info(f"Final result keys: {list(result.keys())}")
        app.logger.info(f"Move made: {result['move_made']}")
        app.logger.info(f"Game over: {result['game_over']}")
        app.logger.info(f"Winner: {result['winner']}")
        
        # Log detailed exploration info if available
        if 'tree_data' in result and 'detailed_exploration' in result['tree_data']:
            de = result['tree_data']['detailed_exploration']
            app.logger.info(f"Detailed exploration: enabled={de.get('enabled')}, "
                           f"simulations={de.get('total_simulations')}, "
                           f"trace_length={len(de.get('trace', []))}")
        else:
            app.logger.info("No detailed exploration data found in tree_data")
        
        # Log response size for debugging
        import json
        try:
            response_json = json.dumps(result)
            response_size = len(response_json)
            app.logger.info(f"Response JSON size: {response_size:,} bytes ({response_size/1024:.1f} KB)")
        except Exception as e:
            app.logger.warning(f"Could not serialize response for size measurement: {e}")
        
        json_time = time.time() - json_start_time
        app.logger.info(f"JSON serialization timing took {json_time:.3f}s")
        
        return result
    except Exception as e:
        app.logger.error(f"=== MCTS MOVE ERROR ===")
        app.logger.error(f"Error in make_mcts_move: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"MCTS move generation failed: {e}"
        }




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
        # Frontend currently expects numeric player codes; expose Enum .value at boundary
        # TODO: Figure out: should we change the frontend to use the Player enum instead?
        #       If not, we should at least use constants rather than literals here.
        "PLAYER_VALUES": {
        	"BLUE": 0,
        	"RED": 1
        },
        "WINNER_VALUES": {
            "BLUE": TRMPH_BLUE_WIN,
            "RED": TRMPH_RED_WIN
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
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")  # Default to model1
    temperature = data.get("temperature", 1.0)  # Default temperature
    verbose = data.get("verbose", 0)  # Get verbose level
    
    try:
        state = HexGameState.from_trmph(trmph)
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400

    board = state.board.tolist()
    player_enum = state.current_player_enum  # Use enum directly
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner

    # Debug logging for board data (only if verbose >= 4)
    if verbose >= 4:
        app.logger.debug(f"Board data being sent to frontend: {board}")
        app.logger.debug(f"Board type: {type(board)}, Board shape: {len(board)}x{len(board[0]) if board else 0}")
        if board and len(board) > 0 and len(board[0]) > 0:
            app.logger.debug(f"Sample board values: [0,0]='{board[0][0]}', [0,1]='{board[0][1]}', [1,0]='{board[1][0]}'")

    # Use enum-based color conversion - much safer
    player_color = winner_to_color(player_enum)

    # Model inference - fail fast if this fails
    model = get_model(model_id)
    policy_logits, value_signed = model.simple_infer(trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    # Map policy to trmph moves
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    # Win probability for current player - use enum directly
    win_prob = ValuePredictor.get_win_probability(value_signed, player_enum)

    # Consistent enum-based player representation
    player_enum_name = player_enum.name
    player_index = int(player_enum.value)
    return jsonify({
        "board": board,
        "player": player_color,
        "player_enum": player_enum_name,  # Canonical enum name
        "player_index": player_index,     # Canonical numeric index (0=BLUE,1=RED)
        "legal_moves": legal_moves,
        "winner": winner,
        "policy": policy_dict,
        "value_signed": float(value_signed) if 'value_signed' in locals() else 0.0,
        "win_prob": win_prob,
        "trmph": trmph,
    })

@app.route("/api/apply_move", methods=["POST"])
def api_apply_move():
    """Apply only a human move without making a computer move."""
    data = request.get_json()
    trmph = data.get("trmph")
    move = data.get("move")
    model_id = data.get("model_id", "model1")
    temperature = data.get("temperature", 1.0)  # Default temperature
    verbose = data.get("verbose", 0)  # Get verbose level
    
    try:
        state = HexGameState.from_trmph(trmph)
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400
    
    try:
        state = apply_move_to_state_trmph(state, move)
    except Exception as e:
        return jsonify({"error": f"Invalid move: {e}"}), 400

    new_trmph = state.to_trmph()
    board = state.board.tolist()
    player_enum = state.current_player_enum  # Use enum directly
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner

    # Debug logging for board data (only if verbose >= 4)
    if verbose >= 4:
        app.logger.debug(f"Apply move - Board data being sent to frontend: {board}")
        app.logger.debug(f"Apply move - Board data type: {type(board)}, Board shape: {len(board[0]) if board else 0}")
        if board and len(board) > 0 and len(board[0]) > 0:
            app.logger.debug(f"Apply move - Sample board values: [0,0]='{board[0][0]}', [0,1]='{board[0][1]}', [1,0]='{board[1][0]}'")

    # Use enum-based color conversion - much safer
    player_color = winner_to_color(player_enum)
    winner_color = winner_to_color(winner) if winner is not None else None

    # Recompute policy/value for the new state - fail fast if this fails
    model = get_model(model_id)
    policy_logits, value_signed = model.simple_infer(new_trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = ValuePredictor.get_win_probability(value_signed, player_enum)

    # Consistent enum-based player representation
    player_enum_name = player_enum.name
    player_index = int(player_enum.value)
    return jsonify({
        "new_trmph": new_trmph,
        "board": board,
        "player": player_color,
        "player_enum": player_enum_name,
        "player_index": player_index,
        "legal_moves": legal_moves,
        "winner": winner_color,
        "model_move": None,  # No computer move made
        "policy": policy_dict,
        "value_signed": float(value_signed) if 'value_signed' in locals() else 0.0,
        "win_prob": win_prob,
    })

@app.route("/api/apply_trmph_sequence", methods=["POST"])
def api_apply_trmph_sequence():
    """Apply a sequence of TRMPH moves to the board state."""
    data = request.get_json()
    trmph = data.get("trmph")
    trmph_sequence = data.get("trmph_sequence", "")
    model_id = data.get("model_id", "model1")
    temperature = data.get("temperature", 1.0)
    verbose = data.get("verbose", 0)
    
    try:
        # Start with the current state
        state = HexGameState.from_trmph(trmph)
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400
    
    # Parse and apply the sequence of moves
    try:
        # Strip any whitespace and split by commas or spaces if needed
        trmph_sequence = trmph_sequence.strip()
        if not trmph_sequence:
            return jsonify({"error": "No TRMPH sequence provided"}), 400
        
        # Parse the moves using the existing utility
        moves = fc.split_trmph_moves(trmph_sequence)
        
        # Apply each move
        for move in moves:
            if state.game_over:
                break  # Stop if game is already over
            state = apply_move_to_state_trmph(state, move)
        
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH sequence: {e}"}), 400

    new_trmph = state.to_trmph()
    board = state.board.tolist()
    player_enum = state.current_player_enum  # Use enum directly
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner

    # Use enum-based color conversion - much safer
    player_color = winner_to_color(player_enum)
    winner_color = winner_to_color(winner) if winner is not None else None

    # Recompute policy/value for the new state - fail fast if this fails
    model = get_model(model_id)
    policy_logits, value_signed = model.simple_infer(new_trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = ValuePredictor.get_win_probability(value_signed, player_enum)

    # Consistent enum-based player representation
    player_enum_name = player_enum.name
    player_index = int(player_enum.value)
    return jsonify({
        "new_trmph": new_trmph,
        "board": board,
        "player": player_color,
        "player_enum": player_enum_name,
        "player_index": player_index,
        "legal_moves": legal_moves,
        "winner": winner_color,
        "policy": policy_dict,
        "value_signed": float(value_signed) if 'value_signed' in locals() else 0.0,
        "win_prob": win_prob,
        "moves_applied": len(moves),
        "game_over": state.game_over,
    })



@app.route("/api/mcts_move", methods=["POST"])
def api_mcts_move():
    """Make a computer move using MCTS with diagnostic output."""
    data = request.get_json()
    app.logger.info(f"=== MCTS API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")
    num_simulations = data.get("num_simulations", 200)
    exploration_constant = data.get("exploration_constant", 2.8)
    temperature = data.get("temperature", 1.0)
    temperature_end = data.get("temperature_end", 0.1)  # Default final temperature
    verbose = data.get("verbose", 0)
    enable_gumbel = data.get("enable_gumbel", True)
    gumbel_max_sims = data.get("gumbel_max_sims", 500)
    
    app.logger.info(f"Parsed parameters: trmph={trmph[:50]}..., model_id={model_id}, sims={num_simulations}, temp={temperature}->{temperature_end}, verbose={verbose}, gumbel={enable_gumbel}, gumbel_max_sims={gumbel_max_sims}")
    
    # Optional orchestration overrides from request
    orchestration_cfg = data.get("orchestration", None)
    orchestration = _build_orchestration_from_dict(orchestration_cfg)
    
    result = make_mcts_move(
        trmph,
        model_id,
        num_simulations,
        exploration_constant,
        temperature,
        temperature_end,
        verbose,
        orchestration_overrides=orchestration,
        enable_gumbel=enable_gumbel,
        gumbel_max_sims=gumbel_max_sims
    )
    
    app.logger.info(f"=== MCTS API RESPONSE ===")
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
    
    return jsonify(result)

@app.route("/api/save_game", methods=["POST"])
def api_save_game():
    """Save a game to the web_games directory in TRMPH format."""
    data = request.get_json()
    app.logger.info(f"=== SAVE GAME API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    trmph = data.get("trmph")
    winner = data.get("winner")  # "blue", "red", or None if game not finished
    model_id = data.get("model_id", "model1")
    mcts_params = data.get("mcts_params", {})
    
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

@app.route("/")
def serve_index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), "index.html")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hex AI Web Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, use_reloader=False, use_debugger=True, threaded=False, host=args.host, port=args.port) 