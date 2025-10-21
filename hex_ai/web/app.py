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
import string

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

# =============================================================================
# FLASK APP CONFIGURATION
# =============================================================================

# Set maximum content length to prevent large payloads (2 KB limit)
# Rationale: Maximum legitimate payload = 1,088 bytes (169 moves × 3 chars + JSON overhead)
# 2KB provides tight security with minimal buffer for any edge cases
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024  # 2 KB

# JSON error handler for malformed JSON
@app.errorhandler(413)
def too_large(e):
    """Handle request too large errors."""
    app.logger.warning(f"Request too large: {e}")
    return jsonify({"error": "Request too large. Maximum size is 2 KB."}), 413

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request errors (including malformed JSON)."""
    app.logger.warning(f"Bad request: {e}")
    return jsonify({"error": "Invalid request format. Please check your input."}), 400

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
    Also allows empty strings for initial game state.
    
    Args:
        trmph_string (str): The TRMPH string to validate
        
    Returns:
        tuple: (is_valid, error_message) where is_valid is bool and error_message is str or None
    """
    # Handle None values (from frontend null/undefined) by treating as empty string
    if trmph_string is None:
        trmph_string = ""
    
    if not isinstance(trmph_string, str):
        return False, "TRMPH input must be a string"
    
    # Remove any whitespace
    trmph_string = trmph_string.strip()
    
    # Allow empty strings for initial game state
    if not trmph_string:
        return True, None
    
    # Build pattern based on board size
    max_col = string.ascii_lowercase[BOARD_SIZE - 1]  # 'm' for 13x13
    if BOARD_SIZE <= 9:
        # For boards 9x9 or smaller, only single digits
        number_pattern = f'[1-{BOARD_SIZE}]'
    else:
        # For boards 10x10 or larger, handle 10-13 range
        number_pattern = f'(1[0-{BOARD_SIZE%10}]|[1-9])'
    
    trmph_pattern_with_prefix = re.compile(f'^#{BOARD_SIZE},([a-{max_col}]{number_pattern})+$')
    trmph_pattern_without_prefix = re.compile(f'^([a-{max_col}]{number_pattern})+$')
    
    if not (trmph_pattern_with_prefix.match(trmph_string) or trmph_pattern_without_prefix.match(trmph_string)):
        return False, f"Invalid TRMPH format. Only letters a-{max_col} followed by numbers 1-{BOARD_SIZE} are allowed (e.g., a1b2c3 or #{BOARD_SIZE},a1b2c3)"
    
    # Use existing utility function to properly count moves
    try:
        # Strip the #{BOARD_SIZE}, prefix if present before parsing moves
        prefix = f'#{BOARD_SIZE},'
        if trmph_string.startswith(prefix):
            bare_moves = trmph_string[len(prefix):]  # Remove prefix
        else:
            bare_moves = trmph_string
        
        moves = fc.split_trmph_moves(bare_moves)
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
    
    # Reject unexpected fields (defense-in-depth)
    all_allowed = set(required_fields or []) | set(optional_fields or [])
    unexpected = set(data.keys()) - all_allowed
    if unexpected:
        return False, f"Unexpected fields: {list(unexpected)}", None
    
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
                elif field == 'elo_rating':
                    # ELO rating needs special validation
                    try:
                        validated_data[field] = validate_elo_rating(data[field])
                    except ValueError as e:
                        return False, f"Invalid {field}: {e}", None
                else:
                    validated_data[field] = data[field]
    
    return True, None, validated_data

def validate_elo_rating(value):
    """
    Validate and convert ELO rating to integer.
    
    Args:
        value: ELO rating value (int, float, str, or None)
        
    Returns:
        int: Validated ELO rating in range [1, 2350]
        
    Raises:
        ValueError: If value is None, cannot be converted, or is out of range
    """
    if value is None:
        raise ValueError("ELO rating is required and cannot be null")
    
    try:
        # Convert to float first to handle string inputs like "1000.0"
        elo_float = float(value)
        elo_int = int(round(elo_float))
    except (ValueError, TypeError):
        raise ValueError("ELO rating must be a number")
    
    if not (1 <= elo_int <= 2350):
        raise ValueError("ELO rating must be between 1 and 2350")
    
    return elo_int

def sanitize_exception_message(exception):
    """
    Sanitize exception messages to remove sensitive implementation details.
    
    Args:
        exception: Exception object
        
    Returns:
        str: Sanitized error message safe for user display
    """
    # Get the base error message
    error_msg = str(exception)
    
    # Remove file paths and line numbers
    import re
    # Remove patterns like "/path/to/file.py:123:"
    error_msg = re.sub(r'/[^\s]*\.py:\d+:', '', error_msg)
    # Remove patterns like "line 123"
    error_msg = re.sub(r'line \d+', 'line', error_msg)
    # Remove patterns like "at 0x12345678"
    error_msg = re.sub(r'at 0x[0-9a-fA-F]+', 'at memory location', error_msg)
    
    # Remove common Python internal details
    error_msg = error_msg.replace('Traceback (most recent call last):', '')
    error_msg = error_msg.replace('File "<', 'File "')
    
    # Clean up whitespace
    error_msg = ' '.join(error_msg.split())
    
    # If we've stripped too much, provide a generic message
    if not error_msg or len(error_msg) < 3:
        return "Invalid input format"
    
    return error_msg

# =============================================================================
# TOKEN BUCKET RATE LIMITING
# =============================================================================

import time
from functools import wraps
from collections import defaultdict

class TokenBucket:
    """
    Token bucket rate limiter for fair resource allocation.
    
    Each IP address gets a bucket with a maximum capacity of tokens.
    Tokens refill at a steady rate. Requests consume tokens based on
    their computational cost. This allows bursts while preventing
    sustained abuse.
    
    Algorithm:
    - Bucket capacity: 20 tokens (allows bursts)
    - Refill rate: 1 token/second (60/minute sustained)
    - Request cost: varies by endpoint (0.5 to 5.0 tokens)
    """
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()
    
    def _refill(self):
        """Refill tokens based on time elapsed since last update."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now
    
    def consume(self, tokens: float) -> bool:
        """
        Attempt to consume tokens. Returns True if successful, False otherwise.
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_remaining(self) -> float:
        """Get remaining tokens (after refilling)."""
        self._refill()
        return self.tokens

# In-memory token bucket storage
# Format: {ip_address: {endpoint: TokenBucket}}
_token_buckets = defaultdict(dict)

# Global request tracking for abuse prevention (max 1000 requests per 3 minutes)
_request_history = defaultdict(list)
REQUEST_WINDOW = 180  # 3 minutes in seconds
MAX_REQUESTS_PER_WINDOW = 1000

def check_global_rate_limit(client_ip):
    """
    Check if client has exceeded global rate limit (1000 requests per 3 minutes).
    Returns (allowed: bool, wait_time: float)
    """
    now = time.time()
    cutoff = now - REQUEST_WINDOW
    
    # Clean up old requests
    _request_history[client_ip] = [
        req_time for req_time in _request_history[client_ip] 
        if req_time > cutoff
    ]
    
    # Check if over limit
    if len(_request_history[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        oldest_request = min(_request_history[client_ip])
        wait_time = REQUEST_WINDOW - (now - oldest_request)
        return False, wait_time
    
    # Record this request
    _request_history[client_ip].append(now)
    
    # Cleanup memory if too many IPs (keep last 10000 IPs)
    if len(_request_history) > 10000:
        # Remove IPs with no recent requests
        for ip in list(_request_history.keys()):
            if not _request_history[ip]:
                del _request_history[ip]
    
    return True, 0

# Token bucket configuration
BUCKET_CAPACITY = 60.0  # Maximum tokens in bucket (allows ~120 rapid undo/redo)
REFILL_RATE = 1.0       # Tokens per second (60/minute)

# Token costs by endpoint (adjusted for user requirements)
# MCTS: every 2 seconds (5 tokens cost, 1 token/sec refill = 2 sec wait)
# Policy: every 1 second (2 tokens cost, 1 token/sec refill = 1 sec wait)
ENDPOINT_COSTS = {
    'api_state': 0.5,              # Read-only, cheap (40 burst calls)
    'api_apply_move': 1.0,        # Simple move application (20 burst calls)
    'api_policy_move': 2.0,       # Model inference (10 burst calls, 1/sec sustained)
    'api_apply_trmph_sequence': 2.0,  # Batch operation (10 burst calls)
    'api_mcts_move': 5.0,         # Expensive MCTS (4 burst calls, 2/sec sustained)
}

def rate_limit(cost: float = 1.0):
    """
    Token bucket rate limiting decorator with weighted costs.
    
    Args:
        cost (float): Number of tokens this request costs (default 1.0)
        
    Returns:
        decorator: Flask route decorator
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client IP address
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            if client_ip:
                client_ip = client_ip.split(',')[0].strip()  # Handle multiple proxies
            
            # Check global rate limit first
            allowed, wait_time = check_global_rate_limit(client_ip)
            if not allowed:
                app.logger.warning(
                    f"Global rate limit exceeded for IP {client_ip}. "
                    f"Over 1000 requests in 3 minutes. Wait: {wait_time:.1f}s"
                )
                return jsonify({
                    "error": f"Too many requests. Please wait {wait_time:.1f} seconds.",
                    "retry_after": wait_time
                }), 429
            
            # Get endpoint name
            endpoint = f.__name__
            
            # Get or create token bucket for this IP/endpoint
            if endpoint not in _token_buckets[client_ip]:
                _token_buckets[client_ip][endpoint] = TokenBucket(
                    capacity=BUCKET_CAPACITY,
                    refill_rate=REFILL_RATE
                )
            
            bucket = _token_buckets[client_ip][endpoint]
            
            # Try to consume tokens
            if not bucket.consume(cost):
                remaining = bucket.get_remaining()
                wait_time = (cost - remaining) / REFILL_RATE
                app.logger.warning(
                    f"Rate limit exceeded for IP {client_ip} on endpoint {endpoint}. "
                    f"Cost: {cost}, Remaining: {remaining:.2f}, Wait: {wait_time:.1f}s"
                )
                return jsonify({
                    "error": f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.",
                    "retry_after": wait_time
                }), 429
            
            # Clean up old buckets (older than 1 hour) to prevent memory leaks
            if len(_token_buckets) > 1000:
                cutoff_time = time.time() - 3600
                for ip in list(_token_buckets.keys()):
                    for ep in list(_token_buckets[ip].keys()):
                        if _token_buckets[ip][ep].last_update < cutoff_time:
                            del _token_buckets[ip][ep]
                    if not _token_buckets[ip]:
                        del _token_buckets[ip]
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

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
    
    # Define difficulty breakpoints for linear interpolation
    # Format: (elo, temperature, num_simulations, algorithm)
    difficulty_points = [
        (1, 2.5, 0, "policy"),       # Mindless
        (300, 1.5, 0, "policy"),     # Beginner  
        (500, 1.1, 0, "policy"),     # Novice
        (1000, 0.75, 0, "policy"),   # Medium
        (1500, 0.50, 0, "policy"),   # Hard
        (1800, 0.25, 0, "policy"),   # Very Hard
        (2100, 0.08, 0, "policy"),   # Expert
        (2150, 0.1, 8, "mcts"),      # Extra Hard - Gumbel MCTS
        (2250, 0.1, 20, "mcts"),     # Ultra Hard - Gumbel MCTS
        (2350, 0.1, 39, "mcts"),     # Ultra Difficult - Gumbel MCTS
    ]
    
    # Find the appropriate segment for linear interpolation
    for i in range(len(difficulty_points) - 1):
        elo_low, temp_low, sims_low, algo_low = difficulty_points[i]
        elo_high, temp_high, sims_high, algo_high = difficulty_points[i + 1]
        
        if elo_low <= elo_rating <= elo_high:
            # Linear interpolation
            if elo_high == elo_low:
                # Avoid division by zero
                ratio = 0
            else:
                ratio = (elo_rating - elo_low) / (elo_high - elo_low)
            
            # Interpolate temperature and simulations
            temperature = temp_low + ratio * (temp_high - temp_low)
            num_simulations = int(round(sims_low + ratio * (sims_high - sims_low)))
            
            # Determine algorithm (use higher algorithm if we're in MCTS range)
            algorithm = algo_high if algo_high == "mcts" else algo_low
            
            if algorithm == "policy":
                return {
                    "algorithm": "policy",
                    "temperature": temperature,
                    "num_simulations": 0,
                    "exploration_constant": 0,
                    "enable_gumbel": False
                }
            else:  # mcts
                return {
                    "algorithm": "mcts",
                    "temperature": 1.0,
                    "temperature_end": 0.1,
                    "num_simulations": num_simulations,
                    "exploration_constant": 2.8,
                    "enable_gumbel": True,
                    "gumbel_max_sims": 500
                }
    
    # Fallback (should not reach here with proper bounds checking)
    return {
        "algorithm": "policy",
        "temperature": 0.5,
        "num_simulations": 0,
        "exploration_constant": 0,
        "enable_gumbel": False
    }


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

def build_move_response(state, move_made=None, success=True, error=None):
    """
    Build a standardized move response with game state information.
    
    Args:
        state (HexGameState): The current game state
        move_made (str, optional): TRMPH representation of the move made
        success (bool): Whether the operation was successful
        error (str, optional): Error message if operation failed
        
    Returns:
        dict: Standardized move response
    """
    response = {
        "success": success,
        "new_trmph": state.to_trmph(),
        "board": state.board.tolist(),
        "player": winner_to_color(state.current_player_enum),
        "legal_moves": moves_to_trmph(state.get_legal_moves()),
        "winner": winner_to_color(state.winner) if state.winner is not None else None,
        "move_made": move_made,
        "game_over": state.game_over
    }
    
    if error:
        response["error"] = error
        
    return response

def _check_game_over_early_return(state, trmph):
    """Check if game is over and return early response if so."""
    if state.game_over:
        app.logger.info("Game is over, returning current state")
        result = build_move_response(state, move_made=None)
        app.logger.info(f"Returning early result: {result}")
        return result
    return None

def _load_model_safely(model_id):
    """Load model with error handling."""
    try:
        model = get_model(model_id)
        app.logger.info(f"Model loaded successfully: {type(model).__name__}")
        return model, None
    except Exception as e:
        app.logger.error(f"Failed to get model {model_id}: {e}")
        return None, f"Model loading failed: {e}"

def _create_mcts_configuration(num_simulations, exploration_constant, temperature, temperature_end, enable_gumbel, gumbel_max_sims):
    """Create MCTS configuration with temperature adjustments."""
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
    return mcts_config

def _execute_mcts_search(state, model_id, mcts_config):
    """Execute MCTS search and return the selected move."""
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
    
    return move

def _apply_move_and_build_response(state, move):
    """Apply the selected move and build the response."""
    selected_move_trmph = fc.rowcol_to_trmph(*move)
    app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")
    
    # Apply the move
    app.logger.info(f"Applying move: {selected_move_trmph}")
    state = apply_move_to_state_trmph(state, selected_move_trmph)
    app.logger.info(f"Move applied. New state game_over: {state.game_over}")
    
    return build_move_response(state, move_made=selected_move_trmph)

def _prepare_mcts_parameters(num_simulations, exploration_constant, temperature, temperature_end, enable_gumbel, gumbel_max_sims):
    """Prepare and validate MCTS parameters."""
    return {
        "num_simulations": num_simulations,
        "exploration_constant": exploration_constant,
        "temperature": temperature,
        "temperature_end": temperature_end,
        "enable_gumbel": enable_gumbel,
        "gumbel_max_sims": gumbel_max_sims
    }

def _execute_mcts_move_workflow(state, model_id, mcts_params):
    """Execute the core MCTS move workflow."""
    # Load model
    model, model_error = _load_model_safely(model_id)
    if model_error:
        return {"success": False, "error": model_error}
    
    # Create MCTS configuration
    mcts_config = _create_mcts_configuration(**mcts_params)
    
    # Execute MCTS search
    move = _execute_mcts_search(state, model_id, mcts_config)
    
    # Apply move and build response
    return _apply_move_and_build_response(state, move)

def make_mcts_move(trmph, model_id, num_simulations, exploration_constant, 
                   temperature, temperature_end, verbose, enable_gumbel, gumbel_max_sims):
    """Make one computer move using MCTS and return the new state with diagnostics."""
    try:
        app.logger.info(f"=== MCTS MOVE START ===")
        app.logger.info(f"Input: model_id={model_id}, sims={num_simulations}, temp={temperature}->{temperature_end}, verbose={verbose}, gumbel={enable_gumbel}, gumbel_max_sims={gumbel_max_sims}")
        app.logger.info(f"Input TRMPH: '{trmph}'")
        
        # Create game state from TRMPH (validation already done by calling API endpoint)
        state = create_game_state_from_trmph(trmph, "for MCTS move")
        app.logger.info(f"Game state created: game_over={state.game_over}, current_player={state.current_player_enum}")
        
        # Check if game is over (early return)
        early_result = _check_game_over_early_return(state, trmph)
        if early_result:
            return early_result
        
        # Prepare MCTS parameters
        mcts_params = _prepare_mcts_parameters(
            num_simulations, exploration_constant, temperature, temperature_end, 
            enable_gumbel, gumbel_max_sims
        )
        
        # Execute MCTS workflow
        result = _execute_mcts_move_workflow(state, model_id, mcts_params)
        
        app.logger.debug(f"=== MCTS MOVE COMPLETE ===")
        app.logger.debug(f"Move made: {result.get('move_made', 'N/A')}")
        app.logger.debug(f"Game over: {result.get('game_over', 'N/A')}")
        app.logger.debug(f"Winner: {result.get('winner', 'N/A')}")
        
        return result
    except Exception as e:
        app.logger.error(f"=== MCTS MOVE ERROR ===")
        app.logger.error(f"Error in make_mcts_move: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": "MCTS move generation failed. Please try again."
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


@app.route("/api/state", methods=["POST"])
@rate_limit(cost=0.5)
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
    
    # Create game state from TRMPH (validation already done by validate_api_input)
    state = create_game_state_from_trmph(trmph)

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
@rate_limit(cost=1.0)
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
    
    # Create game state from TRMPH (validation already done by validate_api_input)
    state = create_game_state_from_trmph(trmph, "for move")
    
    try:
        state = apply_move_to_state_trmph(state, move)
    except Exception as e:
        # Silently ignore invalid moves (e.g., clicking on already filled hex)
        app.logger.debug(f"Invalid move ignored: {e}")
        # Return current state without error - user will learn not to click filled hexes
        response = build_game_response(state, elo_rating, trmph, {
            "new_trmph": trmph,
            "model_move": None  # No computer move made
        })
        return jsonify(response)

    new_trmph = state.to_trmph()
    
    # Build response using helper function
    response = build_game_response(state, elo_rating, new_trmph, {
        "new_trmph": new_trmph,
        "model_move": None  # No computer move made
    })
    return jsonify(response)

@app.route("/api/policy_move", methods=["POST"])
@rate_limit(cost=2.0)
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
        # Create game state from TRMPH (validation already done by validate_api_input)
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
        
        result = build_move_response(new_state, move_made=move_trmph)
        
        app.logger.info(f"=== POLICY API RESPONSE ===")
        app.logger.info(f"Selected move: {move_trmph}")
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Policy move error: {e}")
        return jsonify({"success": False, "error": "Policy move generation failed. Please try again."}), 500

@app.route("/api/mcts_move", methods=["POST"])
@rate_limit(cost=5.0)
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
@rate_limit(cost=2.0)
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
        # Create game state from TRMPH (validation already done by validate_api_input)
        state = create_game_state_from_trmph(trmph, "for TRMPH sequence")
        
        # Apply the TRMPH sequence
        moves_applied = 0
        if trmph_sequence and trmph_sequence.strip():
            # Use the proper TRMPH parsing utility instead of naive string slicing
            try:
                moves = fc.split_trmph_moves(trmph_sequence.strip())
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
            except ValueError as e:
                app.logger.error(f"Invalid TRMPH sequence format: {e}")
                return jsonify({"error": "Invalid TRMPH sequence format. Please check the format and try again."}), 400
        
        new_trmph = state.to_trmph()
        
        # Build response using helper function
        response = build_game_response(state, elo_rating, new_trmph, {
            "new_trmph": new_trmph,
            "moves_applied": moves_applied
        })
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"TRMPH sequence application error: {e}")
        return jsonify({"error": "Failed to apply TRMPH sequence. Please check the sequence and try again."}), 500


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