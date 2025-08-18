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
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, run_mcts_move
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.value_utils import Winner, winner_to_color, get_policy_probs_from_logits, get_win_prob_from_model_output, temperature_scaled_softmax
from hex_ai.enums import Player
from hex_ai.value_utils import (
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    select_policy_move,
)
from hex_ai.config import BOARD_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.enums import Piece
from hex_ai.inference.game_engine import apply_move_to_state_trmph
from hex_ai.web.model_browser import create_model_browser
from hex_ai.file_utils import add_recent_model
from hex_ai.inference.model_config import get_default_model_paths

# Model checkpoint defaults from central configuration
DEFAULT_MODEL_PATHS = get_default_model_paths()
DEFAULT_CHKPT_PATH1 = DEFAULT_MODEL_PATHS["model1"]
DEFAULT_CHKPT_PATH2 = DEFAULT_MODEL_PATHS["model2"]

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

# TODO: Refactor duplicated logic across API functions
# The following functions share common patterns that should be extracted into utilities:
# - api_state(), api_apply_move(), api_apply_trmph_sequence(), api_move()
# Common patterns: TRMPH parsing/validation, player color conversion, model inference, response construction

# TODO: Additional refactoring opportunities:
# 1. Create centralized ModelLoader utility (duplicated in get_model(), simple_model_inference.py, model_wrapper.py)
# 2. Consolidate temperature scaling logic (duplicated in batched_mcts.py, mcts.py, value_utils.py)
# 3. Create centralized move selection utility (duplicated in batched_mcts.py, mcts.py, web/app.py)
# 4. Create base MCTS node class (duplicated between BatchedMCTSNode and MCTSNode)
# 5. Break up complex functions: search() in batched_mcts.py, play_single_game() in tournament.py

# Add debug logging for incoming requests
@app.before_request
def log_request_info():
    app.logger.debug(f"Request: {request.method} {request.path}")

# Global model instances
MODELS = {}
MODEL_PATHS = {
    "model1": os.environ.get("HEX_MODEL_PATH1", DEFAULT_CHKPT_PATH1),
    "model2": os.environ.get("HEX_MODEL_PATH2", DEFAULT_CHKPT_PATH2),
}

# Global model browser instance
MODEL_BROWSER = create_model_browser()

# Dynamic model registry for user-selected models
DYNAMIC_MODELS = {}

# --- Model Management ---
def get_model(model_id="model1"):
    """Get or create a model instance for the given model_id."""
    global MODELS, DYNAMIC_MODELS
    
    app.logger.debug(f"get_model called with model_id: {model_id}")
    app.logger.debug(f"Available dynamic models: {list(DYNAMIC_MODELS.keys())}")
    app.logger.debug(f"Available predefined models: {list(MODEL_PATHS.keys())}")
    app.logger.debug(f"Currently loaded models: {list(MODELS.keys())}")
    
    # Check if it's a dynamic model (user-selected)
    if model_id in DYNAMIC_MODELS:
        model_path = DYNAMIC_MODELS[model_id]
        app.logger.debug(f"Found dynamic model {model_id} -> {model_path}")
        
        if model_id not in MODELS:
            try:
                app.logger.debug(f"Loading dynamic model {model_id} from {model_path}")
                
                # Handle relative paths by prepending checkpoints directory
                if not os.path.isabs(model_path):
                    full_model_path = os.path.join("checkpoints", model_path)
                    app.logger.debug(f"Converted relative path to: {full_model_path}")
                else:
                    full_model_path = model_path
                
                # Check if file exists before loading
                if not os.path.exists(full_model_path):
                    raise FileNotFoundError(f"Model file does not exist: {full_model_path}")
                
                app.logger.debug(f"File exists, attempting to load with SimpleModelInference")
                MODELS[model_id] = SimpleModelInference(full_model_path)
                app.logger.info(f"Successfully loaded dynamic model {model_id} from {full_model_path}")
            except Exception as e:
                app.logger.error(f"Failed to load dynamic model {model_id} from {full_model_path}: {e}")
                app.logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                app.logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        else:
            app.logger.debug(f"Model {model_id} already loaded")
        return MODELS[model_id]
    
    # Check if it's a predefined model
    if model_id in MODEL_PATHS:
        app.logger.debug(f"Found predefined model {model_id} -> {MODEL_PATHS[model_id]}")
        if model_id not in MODELS:
            try:
                app.logger.debug(f"Loading predefined model {model_id} from {MODEL_PATHS[model_id]}")
                MODELS[model_id] = SimpleModelInference(MODEL_PATHS[model_id])
                app.logger.info(f"Successfully loaded model {model_id} from {MODEL_PATHS[model_id]}")
            except Exception as e:
                app.logger.error(f"Failed to load model {model_id}: {e}")
                raise
        return MODELS[model_id]
    
    # If model_id is a direct path, try to load it
    if os.path.exists(model_id):
        app.logger.debug(f"Model_id appears to be a direct path: {model_id}")
        if model_id not in MODELS:
            try:
                app.logger.debug(f"Loading model from direct path {model_id}")
                MODELS[model_id] = SimpleModelInference(model_id)
                app.logger.info(f"Successfully loaded model from path {model_id}")
            except Exception as e:
                app.logger.error(f"Failed to load model from path {model_id}: {e}")
                raise
        return MODELS[model_id]
    
    app.logger.error(f"Unknown model_id: {model_id}")
    app.logger.error(f"Available options: dynamic={list(DYNAMIC_MODELS.keys())}, predefined={list(MODEL_PATHS.keys())}")
    raise ValueError(f"Unknown model_id: {model_id}")

def register_dynamic_model(model_id: str, model_path: str):
    """Register a dynamically selected model."""
    global DYNAMIC_MODELS
    DYNAMIC_MODELS[model_id] = model_path
    app.logger.info(f"Registered dynamic model {model_id} -> {model_path}")

def get_available_models():
    """Return list of available model configurations."""
    # Extract filenames from paths for display
    model1_filename = os.path.basename(MODEL_PATHS["model1"])
    model2_filename = os.path.basename(MODEL_PATHS["model2"])
    
    return [
        {"id": "model1", "name": f"Model 1 ({model1_filename})", "path": MODEL_PATHS["model1"]},
        {"id": "model2", "name": f"Model 2 ({model2_filename})", "path": MODEL_PATHS["model2"]},
    ]

def generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                       search_widths, temperature, verbose, model_move=None, search_tree=None, model_id=None):
    """Generate comprehensive debug information based on verbose level."""
    debug_info = {}
    
    # Level 1: Basic policy and value analysis
    if verbose >= 1:
        # Add defensive programming to catch any issues
        try:
            current_player_enum = state.current_player_enum
            current_player_color = winner_to_color(current_player_enum)
            win_prob = get_win_prob_from_model_output(value_logit, current_player_enum)
        except Exception as e:
            # Log the error and provide fallback values
            app.logger.error(f"Error in app.py debug info generation: {e}")
            raise RuntimeError(f"Error in app.py debug info generation: {e}")
        
        debug_info["basic"] = {
            "current_player": current_player_color,
            "current_player_raw": state.current_player,  # Add raw value for debugging
            "game_over": state.game_over,
            "legal_moves_count": len(state.get_legal_moves()),
            "value_logit": float(value_logit),
            "win_probability": float(win_prob),
            "temperature": temperature,
            "search_widths": search_widths,
            "model_move": model_move
        }
        
        # Add model information
        if model_id:
            debug_info["model_info"] = {
                "model_id": model_id,
                "model_type": type(model).__name__,
                "model_path": DYNAMIC_MODELS.get(model_id, MODEL_PATHS.get(model_id, "unknown"))
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
    
    # Level 2: Detailed analysis including tree search
    if verbose >= 2 and search_tree is not None:
        try:
            # Use the actual search tree from the main flow
            debug_info["tree_search"] = {
                "search_widths": search_widths,
                "tree_depth": len(search_widths),
                "final_value": float(search_tree.value) if search_tree.value is not None else None,
                "best_move": fc.rowcol_to_trmph(*search_tree.best_move) if search_tree.best_move else None,
                "tree_size": count_tree_nodes(search_tree)
            }
            
            # Add terminal node analysis
            if verbose >= 3:
                debug_info["tree_search"]["terminal_nodes"] = collect_terminal_nodes(search_tree)
                
        except Exception as e:
            debug_info["tree_search"] = {"error": str(e)}
    
    # Level 3: Full analysis including policy-value conflicts
    if verbose >= 3:
        # Compare policy top move vs tree search best move
        if debug_info.get("policy_analysis") and debug_info.get("tree_search"):
            policy_top_move = debug_info["policy_analysis"]["top_moves"][0]["move"]
            tree_best_move = debug_info["tree_search"]["best_move"]
            
            debug_info["policy_value_comparison"] = {
                "policy_top_move": policy_top_move,
                "tree_best_move": tree_best_move,
                "moves_match": policy_top_move == tree_best_move,
                "policy_top_prob": debug_info["policy_analysis"]["top_moves"][0]["probability"]
            }
    
    return debug_info

def count_tree_nodes(node):
    """Count total nodes in search tree."""
    count = 1
    for child in node.children.values():
        count += count_tree_nodes(child)
    return count

def collect_terminal_nodes(root):
    """Collect all terminal nodes with their values."""
    terminals = []
    
    def collect_terminals(node):
        if not node.children:  # Terminal node
            terminals.append({
                "path": [fc.rowcol_to_trmph(*move) for move in node.path],
                "value": float(node.value) if node.value is not None else None,
                "depth": node.depth
            })
        else:
            for child in node.children.values():
                collect_terminals(child)
    
    collect_terminals(root)
    return terminals

# --- Utility: Convert (row, col) moves to trmph moves ---
def moves_to_trmph(moves):
    return [fc.rowcol_to_trmph(row, col) for row, col in moves]

# --- Computer move functionality ---
def make_computer_move(trmph, model_id, search_widths=None, temperature=1.0, verbose=0):
    """Make one computer move and return the new state."""
    try:
        app.logger.debug(f"make_computer_move called with model_id: {model_id}")
        app.logger.debug(f"Current dynamic models: {DYNAMIC_MODELS}")
        
        state = HexGameState.from_trmph(trmph)
        app.logger.debug(f"Game state created, game_over: {state.game_over}")
        
        # If game is over, return current state
        if state.game_over:
            app.logger.debug("Game is over, returning current state")
            return {
                "success": True,
                "new_trmph": trmph,
                "board": state.board.tolist(),
                "player": winner_to_color(state.current_player),
                "legal_moves": moves_to_trmph(state.get_legal_moves()),
                "winner": state.winner,
                "move_made": None,
                "game_over": True
            }
        
        app.logger.debug(f"Attempting to get model with model_id: {model_id}")
        try:
            model = get_model(model_id)
            app.logger.debug(f"Successfully got model: {type(model)}")
        except Exception as e:
            app.logger.error(f"Failed to get model {model_id}: {e}")
            return {
                "success": False,
                "error": f"Model loading failed: {e}"
            }
        
        debug_info = {}
        
        # Use tree search if search_widths provided, otherwise use simple policy
        search_tree = None
        if search_widths and len(search_widths) > 0:
            try:
                # Capture debug information during the actual search
                if verbose >= 1:
                    # Get policy and value for the current state before search
                    policy_logits, value_logit = model.simple_infer(trmph)
                    policy_probs = policy_logits_to_probs(policy_logits, temperature)
                    
                    # Generate basic debug info from the original state
                    debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                                  search_widths, temperature, verbose, None, None, model_id)  # No search tree yet
                
                best_move, _, search_tree = minimax_policy_value_search(
                    state, model, search_widths, batch_size=1000, temperature=temperature,
                    return_tree=(verbose >= 2)  # Only return tree if we need debug info
                )
                if best_move is not None:
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
                else:
                    app.logger.error("Tree search returned None for best_move")
                    raise RuntimeError("Tree search returned None for best_move")
                    
                # Update debug info with search tree if available
                if verbose >= 2 and search_tree is not None:
                    debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                                  search_widths, temperature, verbose, None, search_tree, model_id)
            except Exception as e:
                app.logger.error(f"Tree search failed: {e}")
                raise RuntimeError(f"Tree search failed: {e}")
        else:
            # Simple policy-based move using centralized utilities
            if verbose >= 1:
                # Get policy and value for the current state before move selection
                policy_logits, value_logit = model.simple_infer(trmph)
                policy_probs = policy_logits_to_probs(policy_logits, temperature)
                
                # Generate basic debug info from the original state
                debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                              search_widths, temperature, verbose, None, None, model_id)
            
            best_move = select_policy_move(state, model, temperature)
            best_move_trmph = fc.rowcol_to_trmph(*best_move)
        
        # Update debug info with the actual move made
        if verbose >= 1 and debug_info:
            debug_info["basic"]["model_move"] = best_move_trmph
        
        # Apply the move using centralized utility
        state = apply_move_to_state_trmph(state, best_move_trmph)
        
        return {
            "success": True,
            "new_trmph": state.to_trmph(),
            "board": state.board.tolist(),
            "player": winner_to_color(state.current_player),
            "legal_moves": moves_to_trmph(state.get_legal_moves()),
            "winner": state.winner,
            "move_made": best_move_trmph,
            "game_over": state.game_over,
            "debug_info": debug_info if verbose >= 1 else None
        }
        
    except Exception as e:
        app.logger.error(f"Computer move failed: {e}")
        return {"success": False, "error": str(e)}


# TODO: Remove this function
def _build_orchestration_from_dict(cfg: dict | None) -> None:
    """Legacy function - orchestration is now handled internally by BaselineMCTS."""
    return None


def make_mcts_move(trmph, model_id, num_simulations=200, exploration_constant=1.4, 
                   temperature=1.0, verbose=0, orchestration_overrides=None):
    """Make one computer move using MCTS and return the new state with diagnostics."""
    try:
        app.logger.info(f"=== MCTS MOVE START ===")
        app.logger.info(f"Input: model_id={model_id}, sims={num_simulations}, temp={temperature}, verbose={verbose}")
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
                "winner": state.winner,
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
        mcts_config = BaselineMCTSConfig(
            sims=num_simulations,
            c_puct=exploration_constant,
            temperature=temperature
        )
        app.logger.info(f"MCTS config created: {mcts_config}")
        
        # Create game engine
        engine = HexGameEngine()
        app.logger.info("Game engine created")
        
        # Create model wrapper for MCTS
        app.logger.info(f"Creating ModelWrapper with checkpoint_path={model.checkpoint_path}, model_type={model.model_type}")
        model_wrapper = ModelWrapper(model.checkpoint_path, device=None, model_type=model.model_type)
        app.logger.info("ModelWrapper created successfully")
        
        # Run MCTS search
        app.logger.info("Starting MCTS search...")
        search_start_time = time.time()
        move, stats = run_mcts_move(engine, model_wrapper, state, mcts_config)
        search_time = time.time() - search_start_time
        app.logger.info(f"MCTS search completed in {search_time:.3f}s")
        app.logger.info(f"MCTS selected move: {move}")
        app.logger.info(f"MCTS stats: {stats}")
        
        selected_move_trmph = fc.rowcol_to_trmph(*move)
        app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")
        
        # Get direct policy comparison
        app.logger.info("Getting direct policy comparison...")
        policy_logits, value_logit = model.simple_infer(trmph)
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        app.logger.info(f"Policy logits shape: {policy_logits.shape}, value_logit: {value_logit}")
        
        # Get legal moves and their probabilities
        legal_moves = state.get_legal_moves()
        app.logger.info(f"Legal moves count: {len(legal_moves)}")
        legal_move_probs = {}
        for move in legal_moves:
            move_trmph = fc.rowcol_to_trmph(*move)
            # Convert (row, col) to tensor index to access policy_probs array
            tensor_idx = fc.rowcol_to_tensor(*move)
            if 0 <= tensor_idx < len(policy_probs):
                legal_move_probs[move_trmph] = float(policy_probs[tensor_idx])
        
        app.logger.info(f"Legal move probabilities: {legal_move_probs}")
        
        # Apply the move
        app.logger.info(f"Applying move: {selected_move_trmph}")
        state = apply_move_to_state_trmph(state, selected_move_trmph)
        app.logger.info(f"Move applied. New state game_over: {state.game_over}")
        
        # Generate MCTS diagnostic info
        mcts_debug_info = {
            "search_stats": {
                "num_simulations": num_simulations,
                "search_time": search_time,
                "exploration_constant": exploration_constant,
                "temperature": temperature,
                "mcts_stats": stats
            },
            "move_selection": {
                "selected_move": selected_move_trmph,
                "selected_move_coords": move
            },
            "move_probabilities": {
                "direct_policy": legal_move_probs,
                "mcts_visits": {},  # Placeholder - would need MCTS tree access
                "mcts_probabilities": {}  # Placeholder - would need MCTS tree access
            },
            "comparison": {
                "mcts_vs_direct": {}
            },
            "win_rate_analysis": {
                "root_value": 0.0,  # Placeholder - would need MCTS tree access
                "best_child_value": 0.0,  # Placeholder - would need MCTS tree access
                "win_probability": 0.5,  # Placeholder
                "best_child_win_probability": 0.5  # Placeholder
            },
            "move_sequence_analysis": {
                "principal_variation": [],  # Placeholder - would need MCTS tree access
                "alternative_lines": [],  # Placeholder - would need MCTS tree access
                "pv_length": 0  # Placeholder
            },
            "summary": {
                "top_direct_move": max(legal_move_probs.items(), key=lambda x: x[1])[0] if legal_move_probs else None,
                "total_legal_moves": len(legal_moves),
                "search_efficiency": 0.0  # Placeholder - would need inference count
            },
            "profiling_summary": {
                "total_compute_ms": int(search_time * 1000.0),
                "encode_ms": stats.get("encode_ms", 0),
                "forward_ms": stats.get("forward_ms", 0),
                "expand_ms": stats.get("expand_ms", 0),
                "backprop_ms": stats.get("backprop_ms", 0),
                "batch_count": stats.get("batch_count", 0),
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0),
            },
        }
        
        # Add comparison data
        for move_trmph in legal_move_probs:
            direct_prob = legal_move_probs.get(move_trmph, 0)
            mcts_debug_info["comparison"]["mcts_vs_direct"][move_trmph] = {
                "direct_probability": direct_prob,
                "mcts_probability": 0.0,  # Placeholder - would need MCTS tree access
                "difference": 0.0  # Placeholder
            }
        
        result = {
            "success": True,
            "new_trmph": state.to_trmph(),
            "board": state.board.tolist(),
            "player": winner_to_color(state.current_player),
            "legal_moves": moves_to_trmph(state.get_legal_moves()),
            "winner": state.winner,
            "move_made": selected_move_trmph,
            "game_over": state.game_over,
            "mcts_debug_info": mcts_debug_info
        }
        
        # Validate that no None values exist in numeric fields that frontend expects
        def validate_numeric_fields(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key in ['search_time', 'total_compute_ms', 'encode_ms', 'forward_ms', 'expand_ms', 'backprop_ms', 
                              'batch_count', 'cache_hits', 'cache_misses', 'root_value', 'best_child_value', 
                              'win_probability', 'best_child_win_probability', 'pv_length', 'search_efficiency',
                              'mcts_probability', 'direct_probability', 'difference']:
                        if value is None:
                            app.logger.warning(f"Found None value in numeric field {current_path}, replacing with 0.0")
                            obj[key] = 0.0
                    validate_numeric_fields(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    validate_numeric_fields(item, f"{path}[{i}]")
        
        validate_numeric_fields(result)
        
        app.logger.info(f"=== MCTS MOVE COMPLETE ===")
        app.logger.info(f"Final result keys: {list(result.keys())}")
        app.logger.info(f"Move made: {result['move_made']}")
        app.logger.info(f"Game over: {result['game_over']}")
        app.logger.info(f"Winner: {result['winner']}")
        
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
    policy_logits, value_logit = model.simple_infer(trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    # Map policy to trmph moves
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    # Win probability for current player - use enum directly
    win_prob = get_win_prob_from_model_output(value_logit, player_enum)

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
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
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
        app.logger.debug(f"Apply move - Board type: {type(board)}, Board shape: {len(board)}x{len(board[0]) if board else 0}")
        if board and len(board) > 0 and len(board[0]) > 0:
            app.logger.debug(f"Apply move - Sample board values: [0,0]='{board[0][0]}', [0,1]='{board[0][1]}', [1,0]='{board[1][0]}'")

    # Use enum-based color conversion - much safer
    player_color = winner_to_color(player_enum)

    # Recompute policy/value for the new state - fail fast if this fails
    model = get_model(model_id)
    policy_logits, value_logit = model.simple_infer(new_trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = get_win_prob_from_model_output(value_logit, player_enum)

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
        "winner": winner,
        "model_move": None,  # No computer move made
        "policy": policy_dict,
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
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

    # Recompute policy/value for the new state - fail fast if this fails
    model = get_model(model_id)
    policy_logits, value_logit = model.simple_infer(new_trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = get_win_prob_from_model_output(value_logit, player_enum)

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
        "winner": winner,
        "policy": policy_dict,
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
        "win_prob": win_prob,
        "moves_applied": len(moves),
        "game_over": state.game_over,
    })

@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json()
    trmph = data.get("trmph")
    move = data.get("move")
    model_id = data.get("model_id", "model1")
    search_widths = data.get("search_widths", None)
    temperature = data.get("temperature", 0.15)  # Default temperature
    verbose = data.get("verbose", 1)  # Verbose level: 0=none, 1=basic, 2=detailed, 3=full
    
    # MCTS parameters
    use_mcts = data.get("use_mcts", True)
    num_simulations = data.get("num_simulations", 200)
    exploration_constant = data.get("exploration_constant", 1.4)
    
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

    model_move = None
    debug_info = {}
    # If game not over and it's model's turn, have model pick a move
    app.logger.info(f"Game state after human move: game_over={state.game_over}, current_player={player_enum}")
    if not state.game_over:
        try:
            # Determine which player's settings to use for the computer move
            # The current player after the human move determines whose settings to use
            current_player_enum = state.current_player_enum
            current_player_color = winner_to_color(current_player_enum)
            
            if current_player_color == 'blue':
                # Use blue's settings for blue's computer move
                computer_model_id = data.get("blue_model_id", model_id)
                computer_search_widths = data.get("blue_search_widths", search_widths)
                computer_temperature = data.get("blue_temperature", temperature)
                computer_use_mcts = data.get("blue_use_mcts", use_mcts)
                computer_num_simulations = data.get("blue_num_simulations", num_simulations)
                computer_exploration_constant = data.get("blue_exploration_constant", exploration_constant)
            else:  # current_player_color == 'red'
                # Use red's settings for red's computer move
                computer_model_id = data.get("red_model_id", model_id)
                computer_search_widths = data.get("red_search_widths", search_widths)
                computer_temperature = data.get("red_temperature", temperature)
                computer_use_mcts = data.get("red_use_mcts", use_mcts)
                computer_num_simulations = data.get("red_num_simulations", num_simulations)
                computer_exploration_constant = data.get("red_exploration_constant", exploration_constant)
            
            model = get_model(computer_model_id)
            
            # Capture the state before the computer move for debug info
            state_before_computer_move = state
            trmph_before_computer_move = new_trmph
            
            # Choose between MCTS and fixed-tree search
            search_tree = None
            mcts_debug_info = {}
            
            if computer_use_mcts:
                # Use MCTS for move selection
                try:
                    app.logger.debug(f"Using MCTS with {computer_num_simulations} simulations")
                    mcts_result = make_mcts_move(
                        trmph_before_computer_move, 
                        computer_model_id, 
                        computer_num_simulations, 
                        computer_exploration_constant, 
                        computer_temperature, 
                        verbose
                    )
                    
                    if mcts_result["success"]:
                        best_move_trmph = mcts_result["move_made"]
                        mcts_debug_info = mcts_result.get("mcts_debug_info", {})
                        # Convert TRMPH move back to coordinates for consistency
                        best_move = fc.trmph_move_to_rowcol(best_move_trmph)
                    else:
                        app.logger.error(f"MCTS failed: {mcts_result.get('error', 'Unknown error')}")
                        raise RuntimeError(f"MCTS failed: {mcts_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    app.logger.error(f"MCTS failed: {e}")
                    raise RuntimeError(f"MCTS failed: {e}")
                    
            elif computer_search_widths and len(computer_search_widths) > 0:
                # Use fixed-tree search if search_widths provided
                try:
                    best_move, _, search_tree = minimax_policy_value_search(
                        state, model, computer_search_widths, batch_size=1000, temperature=computer_temperature,
                        return_tree=(verbose >= 2)  # Only return tree if we need debug info
                    )
                    if best_move is not None:
                        best_move_trmph = fc.rowcol_to_trmph(*best_move)
                    else:
                        app.logger.error("Tree search returned None for best_move")
                        raise RuntimeError("Tree search returned None for best_move")
                except Exception as e:
                    app.logger.error(f"Tree search failed: {e}")
                    raise RuntimeError(f"Tree search failed: {e}")
            else:
                # Simple policy-based move using centralized utilities
                best_move = select_policy_move(state, model, computer_temperature)
                best_move_trmph = fc.rowcol_to_trmph(*best_move)
            
            # Apply model move using centralized utility
            state = apply_move_to_state_trmph(state, best_move_trmph)
            model_move = best_move_trmph
            new_trmph = state.to_trmph()
            board = state.board.tolist()
            legal_moves = moves_to_trmph(state.get_legal_moves())
            winner = state.winner
            
            # Generate debug information after the move is made (when all data is available)
            if verbose >= 1:
                # Get policy and value for the state BEFORE the computer move (the state the computer was thinking about)
                policy_logits, value_logit = model.simple_infer(trmph_before_computer_move)
                policy_probs = policy_logits_to_probs(policy_logits, computer_temperature)
                
                if computer_use_mcts:
                    # Use MCTS debug info
                    debug_info = mcts_debug_info
                else:
                    # Generate debug info using the actual search tree from the move selection
                    debug_info = generate_debug_info(state_before_computer_move, model, policy_logits, value_logit, policy_probs, 
                                                  computer_search_widths, computer_temperature, verbose, model_move, search_tree, computer_model_id)
            
        except Exception as e:
            app.logger.error(f"Model move failed: {e}")
            # Continue without model move if there's an error
    
    # Use enum-based color conversion - much safer
    player_color = winner_to_color(player_enum)

    # Recompute policy/value for final state using centralized utilities
    model = get_model(model_id)
    policy_logits, value_logit = model.simple_infer(new_trmph)
    # Apply temperature scaling to policy using centralized utility
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = get_win_prob_from_model_output(value_logit, player_enum)

    # Consistent enum-based player representation
    player_enum_name = player_enum.name
    player_index = int(player_enum.value)

    response = {
        "new_trmph": new_trmph,
        "board": board,
        "player": player_color,
        "player_enum": player_enum_name,
        "player_index": player_index,
        "legal_moves": legal_moves,
        "winner": winner,
        "model_move": model_move,
        "policy": policy_dict,
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
        "win_prob": win_prob,
    }
    
    if verbose >= 1:
        if computer_use_mcts:
            response["mcts_debug_info"] = debug_info
        else:
            response["debug_info"] = debug_info
    
    return jsonify(response)

@app.route("/api/computer_move", methods=["POST"])
def api_computer_move():
    data = request.get_json()
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")
    search_widths = data.get("search_widths", None)
    temperature = data.get("temperature", 1.0)
    verbose = data.get("verbose", 0)
    
    result = make_computer_move(trmph, model_id, search_widths, temperature, verbose)
    return jsonify(result)


@app.route("/api/mcts_move", methods=["POST"])
def api_mcts_move():
    """Make a computer move using MCTS with diagnostic output."""
    data = request.get_json()
    app.logger.info(f"=== MCTS API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")
    num_simulations = data.get("num_simulations", 200)
    exploration_constant = data.get("exploration_constant", 1.4)
    temperature = data.get("temperature", 1.0)
    verbose = data.get("verbose", 0)
    
    app.logger.info(f"Parsed parameters: trmph={trmph[:50]}..., model_id={model_id}, sims={num_simulations}, temp={temperature}, verbose={verbose}")
    
    # Optional orchestration overrides from request
    orchestration_cfg = data.get("orchestration", None)
    orchestration = _build_orchestration_from_dict(orchestration_cfg)
    
    result = make_mcts_move(
        trmph,
        model_id,
        num_simulations,
        exploration_constant,
        temperature,
        verbose,
        orchestration_overrides=orchestration,
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
    app.run(debug=True, host=args.host, port=args.port) 