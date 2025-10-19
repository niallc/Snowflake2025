from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import logging
import time
from datetime import datetime

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import HexGameState, HexGameEngine, apply_move_to_state_trmph, make_empty_hex_state
from hex_ai.inference.simple_model_inference import SimpleModelInference
import re

from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, run_mcts_move, create_mcts_config, TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.value_utils import (
    Winner, 
    winner_to_color, 
    temperature_scaled_softmax, 
    ValuePredictor,
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    select_policy_move,
    signed_to_prob,
)
from hex_ai.enums import Player, Piece
from hex_ai.inference.mcts_utils import compute_win_probability_from_tree_data
from hex_ai.config import BOARD_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.inference.model_config import get_model_path, get_model_info, get_all_model_info, register_model, is_valid_model_id, get_normalized_path
from hex_ai.inference.model_cache import get_model_cache

app = Flask(__name__, static_folder="static_public")
CORS(app)

# Get centralized model cache
MODEL_CACHE = get_model_cache()

# Preload default model on startup
def preload_default_model():
    """Preload the default model to avoid loading delays on first move."""
    try:
        app.logger.info("Preloading default model...")
        model_path = get_model_path("model1")
        app.logger.info(f"Preloading model1 from {model_path}")
        MODEL_CACHE.get_simple_model(model_path)
        MODEL_CACHE.get_wrapper_model(model_path)
        app.logger.info("Successfully preloaded model1")
    except Exception as e:
        app.logger.error(f"Error during model preloading: {e}")

# Preload model on startup
preload_default_model()

# =============================================================================
# SECURITY VALIDATION
# =============================================================================

def validate_trmph_input(trmph_string):
    """
    Validate TRMPH input format for security.
    
    Only allows single letters a-m followed by numbers 1-13.
    Pattern: ([a-m]([0-9]|1[0-3]))+
    
    Args:
        trmph_string (str): The TRMPH string to validate
        
    Returns:
        tuple: (is_valid, error_message) where is_valid is bool and error_message is str or None
    """
    if not trmph_string or not isinstance(trmph_string, str):
        return False, "TRMPH input must be a non-empty string"
    
    # Remove any whitespace
    trmph_string = trmph_string.strip()
    
    if not trmph_string:
        return False, "TRMPH input cannot be empty"
    
    # Validate format: ([a-m]([0-9]|1[0-3]))+
    trmph_pattern = re.compile(r'^([a-m]([0-9]|1[0-3]))+$')
    
    if not trmph_pattern.match(trmph_string):
        return False, "Invalid TRMPH format. Only letters a-m followed by numbers 1-13 are allowed (e.g., a1b2c3)"
    
    # Use existing utility function to properly count moves
    try:
        moves = fc.split_trmph_moves(trmph_string)
        max_moves = BOARD_SIZE * BOARD_SIZE  # 13^2 = 169
        if len(moves) > max_moves:
            return False, f"Too many moves (maximum {max_moves} moves for a complete game)"
    except ValueError as e:
        return False, f"Invalid TRMPH format: {str(e)}"
    
    return True, None

def validate_api_input(data, required_fields=None, optional_fields=None):
    """
    Centralized validation for API endpoints.
    
    Args:
        data (dict): The request data to validate
        required_fields (list): List of required field names
        optional_fields (list): List of optional field names that should be validated if present
        
    Returns:
        tuple: (is_valid, error_message, validated_data) where validated_data is the cleaned data
    """
    if not isinstance(data, dict):
        return False, "Request data must be a JSON object", None
    
    validated_data = {}
    
    # Check required fields
    if required_fields:
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}", None
            validated_data[field] = data[field]
    
    # Validate optional fields if present
    if optional_fields:
        for field in optional_fields:
            if field in data:
                if field in ['trmph', 'move', 'trmph_sequence']:
                    # These are TRMPH fields that need special validation
                    is_valid, error_msg = validate_trmph_input(data[field])
                    if not is_valid:
                        return False, f"Invalid {field}: {error_msg}", None
                validated_data[field] = data[field]
    
    return True, None, validated_data

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def get_model(model_id="model1"):
    """Get or create a model instance for the given model_id using centralized cache."""
    app.logger.debug(f"get_model called with model_id: {model_id}")
    
    # Use centralized model configuration
    if is_valid_model_id(model_id):
        model_path = get_model_path(model_id)
        app.logger.debug(f"Found model {model_id} -> {model_path}")
        return MODEL_CACHE.get_simple_model(model_path)
    
    app.logger.error(f"Unknown model_id: {model_id}")
    raise ValueError(f"Unknown model_id: {model_id}")

def get_cached_model_wrapper(model_id: str):
    """Get or create a cached ModelWrapper instance for the given model_id using centralized cache."""
    app.logger.debug(f"get_cached_model_wrapper called with model_id: {model_id}")
    
    # Get the model path for this model_id
    if is_valid_model_id(model_id):
        model_path = get_model_path(model_id)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    
    app.logger.debug(f"Getting ModelWrapper for path: {model_path}")
    return MODEL_CACHE.get_wrapper_model(model_path)

# =============================================================================
# DIFFICULTY LEVEL MAPPING
# =============================================================================

def get_difficulty_parameters(elo_rating):
    """Convert ELO rating to appropriate algorithm parameters."""
    if elo_rating < 1:
        elo_rating = 1
    elif elo_rating > 2350:
        elo_rating = 2350
    
    if elo_rating <= 300:
        # ELO 1-300: Policy play with exponential temperature decay
        # temp = 10000 * (0.0003)^(elo/300)
        temp = 10000 * (0.0003 ** (elo_rating / 300))
        return {
            "algorithm": "policy",
            "temperature": temp,
            "num_simulations": 0,
            "exploration_constant": 0,
            "enable_gumbel": False
        }
    elif elo_rating <= 2100:
        # ELO 301-2100: Policy play with refined temperature
        # temp = 3.0 * (0.0333)^((elo-300)/1800)
        temp = 3.0 * (0.0333 ** ((elo_rating - 300) / 1800))
        return {
            "algorithm": "policy",
            "temperature": temp,
            "num_simulations": 0,
            "exploration_constant": 0,
            "enable_gumbel": False
        }
    else:
        # ELO 2101-2350: Gumbel MCTS with increasing simulations
        # sims = 8 + (elo - 2100) * (39 - 8) / 250
        sims = 8 + (elo_rating - 2100) * (39 - 8) / 250
        sims = int(round(sims))
        return {
            "algorithm": "mcts",
            "temperature": 1.0,
            "temperature_end": 0.1,
            "num_simulations": sims,
            "exploration_constant": 2.8,
            "enable_gumbel": True,
            "gumbel_max_sims": 500
        }

def get_difficulty_levels():
    """Return predefined difficulty levels for dropdown selection."""
    return [
        {"name": "Mindless", "elo": 1, "description": "Extremely random play"},
        {"name": "Beginner", "elo": 300, "description": "Very random, makes many mistakes"},
        {"name": "Novice", "elo": 500, "description": "Somewhat random, basic play"},
        {"name": "Medium", "elo": 1000, "description": "Reasonable play with some mistakes"},
        {"name": "Hard", "elo": 1500, "description": "Good play, occasional mistakes"},
        {"name": "Very Hard", "elo": 1800, "description": "Strong play, few mistakes"},
        {"name": "Expert", "elo": 2100, "description": "Very strong play"},
        {"name": "Extra Hard", "elo": 2150, "description": "Expert level with tree search"},
        {"name": "Ultra Hard", "elo": 2250, "description": "Master level play"},
        {"name": "Ultra Difficult", "elo": 2350, "description": "Maximum difficulty"}
    ]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_game_state_from_trmph(trmph, context=""):
    """
    Create a game state from TRMPH string, handling empty TRMPH case.
    
    Args:
        trmph (str): TRMPH string (can be empty for initial state)
        context (str): Context for logging (e.g., "for MCTS move")
        
    Returns:
        HexGameState: The game state
        
    Raises:
        Exception: If TRMPH is invalid
    """
    if not trmph or trmph.strip() == "":
        state = make_empty_hex_state()
        app.logger.info(f"Created initial game state {context}")
    else:
        state = HexGameState.from_trmph(trmph)
        app.logger.info(f"Loaded game state from TRMPH {context}: {trmph[:50]}...")
    
    return state

def build_game_response(state, elo_rating, trmph_for_inference=None, additional_fields=None):
    """
    Build a standardized game response with board, player info, and model predictions.
    
    Args:
        state (HexGameState): The current game state
        elo_rating (int): ELO rating for difficulty parameters
        trmph_for_inference (str, optional): TRMPH string for model inference (defaults to state.to_trmph())
        additional_fields (dict, optional): Additional fields to include in response
        
    Returns:
        dict: Standardized game response
    """
    # Extract basic state information
    board = state.board.tolist()
    player_enum = state.current_player_enum
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner
    
    # Use enum-based color conversion
    player_color = winner_to_color(player_enum)
    winner_color = winner_to_color(winner) if winner is not None else None
    
    # Model inference
    model = get_model("model1")
    if trmph_for_inference is None:
        trmph_for_inference = state.to_trmph()
    
    policy_logits, value_signed = model.simple_infer(trmph_for_inference)
    
    # Get difficulty parameters and apply temperature scaling
    difficulty_params = get_difficulty_parameters(elo_rating)
    temperature = difficulty_params["temperature"]
    
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = ValuePredictor.get_win_probability(value_signed, player_enum)
    
    # Consistent enum-based player representation
    player_enum_name = player_enum.name
    player_index = int(player_enum.value)
    
    # Build base response
    response = {
        "board": board,
        "player": player_color,
        "player_enum": player_enum_name,
        "player_index": player_index,
        "legal_moves": legal_moves,
        "winner": winner_color,
        "policy": policy_dict,
        "value_signed": float(value_signed),
        "win_prob": win_prob,
    }
    
    # Add any additional fields
    if additional_fields:
        response.update(additional_fields)
    
    return response

def moves_to_trmph(moves):
    return [fc.rowcol_to_trmph(row, col) for row, col in moves]

def make_mcts_move(trmph, model_id, num_simulations, exploration_constant, 
                   temperature, temperature_end, verbose, enable_gumbel, gumbel_max_sims):
    """Make one computer move using MCTS and return the new state with diagnostics."""
    try:
        app.logger.info(f"=== MCTS MOVE START ===")
        app.logger.info(f"Input: model_id={model_id}, sims={num_simulations}, temp={temperature}->{temperature_end}, verbose={verbose}, gumbel={enable_gumbel}, gumbel_max_sims={gumbel_max_sims}")
        app.logger.info(f"Input TRMPH: '{trmph}'")
        
        # Note: TRMPH validation is handled by the calling API endpoint
        
        # Create game state from TRMPH
        state = create_game_state_from_trmph(trmph, "for MCTS move")
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
                "game_over": True
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
        if temperature_end > temperature:
            app.logger.info(f"Adjusting temperature_end from {temperature_end} to {temperature/10} (temperature_start/10)")
            temperature_end = temperature / 10
        
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
        
        # Get cached model wrapper for MCTS
        app.logger.info(f"Getting cached ModelWrapper for model_id={model_id}")
        model_wrapper_start = time.time()
        model_wrapper = get_cached_model_wrapper(model_id)
        model_wrapper_time = time.time() - model_wrapper_start
        app.logger.info(f"ModelWrapper retrieval took {model_wrapper_time:.3f}s")
        
        # Run MCTS search
        app.logger.info("Starting MCTS search...")
        total_start_time = time.time()
        
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
        
        selected_move_trmph = fc.rowcol_to_trmph(*move)
        app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")
        
        # Apply the move
        app.logger.info(f"Applying move: {selected_move_trmph}")
        state = apply_move_to_state_trmph(state, selected_move_trmph)
        app.logger.info(f"Move applied. New state game_over: {state.game_over}")
        
        result = {
            "success": True,
            "new_trmph": state.to_trmph(),
            "board": state.board.tolist(),
            "player": winner_to_color(state.current_player),
            "legal_moves": moves_to_trmph(state.get_legal_moves()),
            "winner": winner_to_color(state.winner) if state.winner is not None else None,
            "move_made": selected_move_trmph,
            "game_over": state.game_over
        }
        
        total_wall_time = time.time() - total_start_time
        app.logger.debug(f"=== MCTS MOVE COMPLETE ===")
        app.logger.debug(f"Total wall time: {total_wall_time:.3f}s")
        app.logger.debug(f"Move made: {result['move_made']}")
        app.logger.debug(f"Game over: {result['game_over']}")
        app.logger.debug(f"Winner: {result['winner']}")
        
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

# =============================================================================
# API ROUTES
# =============================================================================

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
            "BLUE": 0,
            "RED": 1
        },
        "WINNER_VALUES": {
            "BLUE": TRMPH_BLUE_WIN,
            "RED": TRMPH_RED_WIN
        }
    })

@app.route("/api/difficulty_levels", methods=["GET"])
def api_difficulty_levels():
    """Get available difficulty levels."""
    return jsonify({"difficulty_levels": get_difficulty_levels()})

@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json()
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=None,  # No required fields
        optional_fields=['trmph', 'elo_rating']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", 1000)  # Default to Medium difficulty
    
    app.logger.info(f"api_state called with trmph='{trmph}', elo_rating={elo_rating}")
    
    try:
        # Create game state from TRMPH
        state = create_game_state_from_trmph(trmph)
    except Exception as e:
        app.logger.error(f"Failed to create game state from TRMPH '{trmph}': {e}")
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400

    # For empty TRMPH, we need to use the state's TRMPH representation
    if not trmph or trmph.strip() == "":
        trmph_for_inference = state.to_trmph()
        app.logger.info(f"Using state TRMPH for inference: {trmph_for_inference[:50]}...")
    else:
        trmph_for_inference = trmph
    
    # Build response using helper function
    response = build_game_response(state, elo_rating, trmph_for_inference, {"trmph": trmph})
    return jsonify(response)

@app.route("/api/apply_move", methods=["POST"])
def api_apply_move():
    """Apply only a human move without making a computer move."""
    data = request.get_json()
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=['move'],  # Move is required
        optional_fields=['trmph', 'elo_rating']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    move = validated_data.get("move")
    elo_rating = validated_data.get("elo_rating", 1000)
    
    app.logger.info(f"api_apply_move called with trmph='{trmph}', move='{move}', elo_rating={elo_rating}")
    
    try:
        # Create game state from TRMPH
        state = create_game_state_from_trmph(trmph, "for move")
    except Exception as e:
        app.logger.error(f"Failed to create game state from TRMPH '{trmph}': {e}")
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400
    
    try:
        state = apply_move_to_state_trmph(state, move)
    except Exception as e:
        return jsonify({"error": f"Invalid move: {e}"}), 400

    new_trmph = state.to_trmph()
    
    # Build response using helper function
    response = build_game_response(state, elo_rating, new_trmph, {
        "new_trmph": new_trmph,
        "model_move": None  # No computer move made
    })
    return jsonify(response)

@app.route("/api/policy_move", methods=["POST"])
def api_policy_move():
    """Make a computer move using policy sampling."""
    data = request.get_json()
    app.logger.info(f"=== POLICY API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=None,  # No required fields
        optional_fields=['trmph', 'elo_rating']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", 1000)
    
    app.logger.info(f"Parsed parameters: trmph='{trmph}', elo_rating={elo_rating}")
    
    try:
        # Create game state from TRMPH
        state = create_game_state_from_trmph(trmph, "for policy move")
        
        # Get difficulty parameters
        difficulty_params = get_difficulty_parameters(elo_rating)
        temperature = difficulty_params["temperature"]
        
        # Get model and make policy move
        model = get_model("model1")
        move = select_policy_move(state, model, temperature)
        
        if move is None:
            return jsonify({"success": False, "error": "No valid moves available"}), 400
        
        # Apply the move
        move_trmph = fc.rowcol_to_trmph(move[0], move[1])
        new_state = apply_move_to_state_trmph(state, move_trmph)
        new_trmph = new_state.to_trmph()
        
        result = {
            "success": True,
            "new_trmph": new_trmph,
            "board": new_state.board.tolist(),
            "player": winner_to_color(new_state.current_player_enum),
            "legal_moves": [fc.rowcol_to_trmph(r, c) for r, c in new_state.get_legal_moves()],
            "winner": winner_to_color(new_state.winner) if new_state.winner is not None else None,
            "move_made": move_trmph,
            "game_over": new_state.game_over
        }
        
        app.logger.info(f"=== POLICY API RESPONSE ===")
        app.logger.info(f"Selected move: {move_trmph}")
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Policy move error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/mcts_move", methods=["POST"])
def api_mcts_move():
    """Make a computer move using MCTS with diagnostic output."""
    data = request.get_json()
    app.logger.info(f"=== MCTS API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=None,  # No required fields
        optional_fields=['trmph', 'elo_rating']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", 1000)
    
    app.logger.info(f"Parsed parameters: trmph='{trmph}', elo_rating={elo_rating}")
    
    # Get difficulty parameters
    difficulty_params = get_difficulty_parameters(elo_rating)
    
    if difficulty_params["algorithm"] == "policy":
        # Use policy move for lower difficulties
        return api_policy_move()
    
    # Use MCTS for higher difficulties
    result = make_mcts_move(
        trmph,
        "model1",
        difficulty_params["num_simulations"],
        difficulty_params["exploration_constant"],
        difficulty_params["temperature"],
        difficulty_params["temperature_end"],
        0,  # verbose
        difficulty_params["enable_gumbel"],
        difficulty_params["gumbel_max_sims"]
    )
    
    app.logger.info(f"=== MCTS API RESPONSE ===")
    app.logger.info(f"Result success: {result.get('success', 'MISSING')}")
    if result.get('success'):
        app.logger.info(f"Move made: {result.get('move_made', 'MISSING')}")
        app.logger.info(f"Game over: {result.get('game_over', 'MISSING')}")
        app.logger.info(f"Winner: {result.get('winner', 'MISSING')}")
    else:
        app.logger.error(f"Result error: {result.get('error', 'MISSING')}")
    
    return jsonify(result)

@app.route("/api/apply_trmph_sequence", methods=["POST"])
def api_apply_trmph_sequence():
    """Apply a sequence of TRMPH moves to the current game state."""
    data = request.get_json()
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=['trmph_sequence'],  # Sequence is required
        optional_fields=['trmph', 'elo_rating']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    trmph_sequence = validated_data.get("trmph_sequence", "")
    elo_rating = validated_data.get("elo_rating", 1000)
    
    app.logger.info(f"api_apply_trmph_sequence called with trmph='{trmph}', sequence='{trmph_sequence}', elo_rating={elo_rating}")
    
    try:
        # Create game state from TRMPH
        state = create_game_state_from_trmph(trmph, "for TRMPH sequence")
        
        # Apply the TRMPH sequence
        moves_applied = 0
        if trmph_sequence and trmph_sequence.strip():
            # Parse individual moves from the sequence
            moves = []
            for i in range(0, len(trmph_sequence), 2):
                if i + 1 < len(trmph_sequence):
                    move = trmph_sequence[i:i+2]
                    moves.append(move)
            
            app.logger.info(f"Applying {len(moves)} moves from sequence: {moves}")
            
            # Apply each move
            for move in moves:
                if not state.game_over:
                    state = apply_move_to_state_trmph(state, move)
                    moves_applied += 1
                    app.logger.info(f"Applied move {move}, game_over: {state.game_over}")
                else:
                    app.logger.info(f"Game is over, skipping remaining moves")
                    break
        
        new_trmph = state.to_trmph()
        
        # Build response using helper function
        response = build_game_response(state, elo_rating, new_trmph, {
            "new_trmph": new_trmph,
            "moves_applied": moves_applied
        })
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"TRMPH sequence application error: {e}")
        return jsonify({"error": f"Failed to apply TRMPH sequence: {e}"}), 500


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(os.path.dirname(__file__)), "favicon3_cropped.png")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static_public"), path)

@app.route("/")
def serve_index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static_public"), "index.html")

@app.route("/rules.html")
def serve_rules():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static_public"), "rules.html")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hex AI Public Web Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    app.logger.info("=" * 50)
    app.logger.info("Hex AI Public Web Server Starting...")
    app.logger.info("=" * 50)
    app.run(debug=True, use_reloader=False, use_debugger=True, threaded=False, host=args.host, port=args.port)