from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import numpy as np
from flask_cors import CORS
import logging
from typing import Tuple


from hex_ai.utils import format_conversion as fc
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.value_utils import Winner, winner_to_color, get_policy_probs_from_logits, get_win_prob_from_model_output, temperature_scaled_softmax
from hex_ai.value_utils import (
    # Add new utilities
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    select_policy_move,  # Add the new public function
    apply_move_to_state_trmph,  # Add move application utilities
)

# Model checkpoint defaults
ALL_RESULTS_DIR = "checkpoints/hyperparameter_tuning/"
THIS_MODEL_DIR = "loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
CHECKPOINT_FILE1 = "epoch2_mini4.pt"
CHECKPOINT_FILE2 = "epoch2_mini26.pt"
DEFAULT_CHKPT_PATH1 = f"{ALL_RESULTS_DIR}/{THIS_MODEL_DIR}/{CHECKPOINT_FILE1}"
DEFAULT_CHKPT_PATH2 = f"{ALL_RESULTS_DIR}/{THIS_MODEL_DIR}/{CHECKPOINT_FILE2}"

app = Flask(__name__, static_folder="static")
CORS(app)

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

# --- Model Management ---
def get_model(model_id="model1"):
    """Get or create a model instance for the given model_id."""
    global MODELS
    if model_id not in MODELS:
        try:
            MODELS[model_id] = SimpleModelInference(MODEL_PATHS[model_id])
            app.logger.info(f"Loaded model {model_id} from {MODEL_PATHS[model_id]}")
        except Exception as e:
            app.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    return MODELS[model_id]

def get_available_models():
    """Return list of available model configurations."""
    return [
        {"id": "model1", "name": f"Model 1 ({CHECKPOINT_FILE1})", "path": MODEL_PATHS["model1"]},
        {"id": "model2", "name": f"Model 2 ({CHECKPOINT_FILE2})", "path": MODEL_PATHS["model2"]},
    ]

def generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                       search_widths, temperature, verbose, model_move=None, search_tree=None):
    """Generate comprehensive debug information based on verbose level."""
    debug_info = {}
    
    # Level 1: Basic policy and value analysis
    if verbose >= 1:
        # Add defensive programming to catch any issues
        try:
            current_player_color = winner_to_color(state.current_player)
            win_prob = get_win_prob_from_model_output(value_logit, current_player_color)
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
        state = HexGameState.from_trmph(trmph)
        
        # If game is over, return current state
        if state.game_over:
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
        
        model = get_model(model_id)
        debug_info = {}
        
        # Use tree search if search_widths provided, otherwise use simple policy
        search_tree = None
        if search_widths and len(search_widths) > 0:
            try:
                # Capture debug information during the actual search
                if verbose >= 1:
                    # Get policy and value for the current state before search
                    policy_logits, value_logit = model.infer(trmph)
                    policy_probs = policy_logits_to_probs(policy_logits, temperature)
                    
                    # Generate basic debug info from the original state
                    debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                                  search_widths, temperature, verbose, None, None)  # No search tree yet
                
                best_move, _, search_tree = minimax_policy_value_search(
                    state, model, search_widths, batch_size=1000, temperature=temperature,
                    return_tree=(verbose >= 2)  # Only return tree if we need debug info
                )
                if best_move is not None:
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
                else:
                    # Fallback to policy-based move using centralized utilities
                    best_move = select_policy_move(state, model, temperature)
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
                    
                # Update debug info with search tree if available
                if verbose >= 2 and search_tree is not None:
                    debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                                  search_widths, temperature, verbose, None, search_tree)
            except Exception as e:
                app.logger.warning(f"Tree search failed, falling back to policy: {e}")
                # Fallback to policy-based move using centralized utilities
                best_move = select_policy_move(state, model, temperature)
                best_move_trmph = fc.rowcol_to_trmph(*best_move)
        else:
            # Simple policy-based move using centralized utilities
            if verbose >= 1:
                # Get policy and value for the current state before move selection
                policy_logits, value_logit = model.infer(trmph)
                policy_probs = policy_logits_to_probs(policy_logits, temperature)
                
                # Generate basic debug info from the original state
                debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                              search_widths, temperature, verbose, None)
            
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

@app.route("/api/models", methods=["GET"])
def api_models():
    """Get available models."""
    return jsonify({"models": get_available_models()})

@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json()
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")  # Default to model1
    temperature = data.get("temperature", 1.0)  # Default temperature
    
    try:
        state = HexGameState.from_trmph(trmph)
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400

    board = state.board.tolist()
    player = state.current_player
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner

    # Add defensive programming to catch any issues
    try:
        player_color = winner_to_color(player)
    except Exception as e:
        app.logger.error(f"Error converting player to color: {e}, player={player}")
        player_color = 'unknown'

    # Model inference
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(trmph)
        # Apply temperature scaling to policy using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        # Map policy to trmph moves
        policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
        # Win probability for current player
        win_prob = get_win_prob_from_model_output(value_logit, player_color)
    except Exception as e:
        app.logger.error(f"Model inference failed: {e}")
        policy_dict = {}
        win_prob = 0.5

    return jsonify({
        "board": board,
        "player": player_color,
        "player_raw": player,  # Add raw value for debugging
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
    player = state.current_player
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner

    # Add defensive programming to catch any issues
    try:
        player_color = winner_to_color(player)
    except Exception as e:
        app.logger.error(f"Error converting player to color: {e}, player={player}")
        player_color = 'unknown'

    # Recompute policy/value for the new state
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(new_trmph)
        # Apply temperature scaling to policy using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
        win_prob = get_win_prob_from_model_output(value_logit, player_color)
    except Exception as e:
        app.logger.error(f"Final inference failed: {e}")
        policy_dict = {}
        win_prob = 0.5

    return jsonify({
        "new_trmph": new_trmph,
        "board": board,
        "player": player_color,
        "player_raw": player,  # Add raw value for debugging
        "legal_moves": legal_moves,
        "winner": winner,
        "model_move": None,  # No computer move made
        "policy": policy_dict,
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
        "win_prob": win_prob,
    })

@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json()
    trmph = data.get("trmph")
    move = data.get("move")
    model_id = data.get("model_id", "model1")
    search_widths = data.get("search_widths", None)
    temperature = data.get("temperature", 0.15)  # Default temperature
    verbose = data.get("verbose", 3)  # Verbose level: 0=none, 1=basic, 2=detailed, 3=full
    
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
    player = state.current_player
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner

    model_move = None
    debug_info = {}
    # If game not over and it's model's turn, have model pick a move
    app.logger.info(f"Game state after human move: game_over={state.game_over}, current_player={state.current_player}")
    if not state.game_over:
        try:
            # Determine which player's settings to use for the computer move
            # The current player after the human move determines whose settings to use
            try:
                current_player_color = winner_to_color(player)
            except Exception as e:
                app.logger.error(f"Error converting player to color for computer move: {e}, player={player}")
                current_player_color = 'unknown'  # Avoids fallbacks as they can cause silent errors
            
            if current_player_color == 'blue':
                # Use blue's settings for blue's computer move
                computer_model_id = data.get("blue_model_id", model_id)
                computer_search_widths = data.get("blue_search_widths", search_widths)
                computer_temperature = data.get("blue_temperature", temperature)
            else:  # current_player_color == 'red'
                # Use red's settings for red's computer move
                computer_model_id = data.get("red_model_id", model_id)
                computer_search_widths = data.get("red_search_widths", search_widths)
                computer_temperature = data.get("red_temperature", temperature)
            
            model = get_model(computer_model_id)
            
            # Capture the state before the computer move for debug info
            state_before_computer_move = state
            trmph_before_computer_move = new_trmph
            
            # Use tree search if search_widths provided
            search_tree = None
            if computer_search_widths and len(computer_search_widths) > 0:
                try:
                    best_move, _, search_tree = minimax_policy_value_search(
                        state, model, computer_search_widths, batch_size=1000, temperature=computer_temperature,
                        return_tree=(verbose >= 2)  # Only return tree if we need debug info
                    )
                    if best_move is not None:
                        best_move_trmph = fc.rowcol_to_trmph(*best_move)
                    else:
                        # Fallback to policy-based move using centralized utilities
                        best_move = select_policy_move(state, model, computer_temperature)
                        best_move_trmph = fc.rowcol_to_trmph(*best_move)
                except Exception as e:
                    app.logger.warning(f"Tree search failed, falling back to policy: {e}")
                    # Fallback to policy-based move using centralized utilities
                    best_move = select_policy_move(state, model, computer_temperature)
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
            else:
                # Simple policy-based move using centralized utilities
                best_move = select_policy_move(state, model, computer_temperature)
                best_move_trmph = fc.rowcol_to_trmph(*best_move)
            
            # Apply model move using centralized utility
            state = apply_move_to_state_trmph(state, best_move_trmph)
            model_move = best_move_trmph
            new_trmph = state.to_trmph()
            board = state.board.tolist()
            player = state.current_player
            legal_moves = moves_to_trmph(state.get_legal_moves())
            winner = state.winner
            
            # Generate debug information after the move is made (when all data is available)
            if verbose >= 1:
                # Get policy and value for the state BEFORE the computer move (the state the computer was thinking about)
                policy_logits, value_logit = model.infer(trmph_before_computer_move)
                policy_probs = policy_logits_to_probs(policy_logits, computer_temperature)
                
                # Generate debug info using the actual search tree from the move selection
                debug_info = generate_debug_info(state_before_computer_move, model, policy_logits, value_logit, policy_probs, 
                                              computer_search_widths, computer_temperature, verbose, model_move, search_tree)
            
        except Exception as e:
            app.logger.error(f"Model move failed: {e}")
            # Continue without model move if there's an error
    
    # Add defensive programming to catch any issues
    try:
        player_color = winner_to_color(player)
    except Exception as e:
        app.logger.error(f"Error converting player to color: {e}, player={player}")
        player_color = 'unknown'

    # Recompute policy/value for final state using centralized utilities
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(new_trmph)
        # Apply temperature scaling to policy using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
        win_prob = get_win_prob_from_model_output(value_logit, player_color)
    except Exception as e:
        app.logger.error(f"Final inference failed: {e}")
        policy_dict = {}
        win_prob = 0.5

    response = {
        "new_trmph": new_trmph,
        "board": board,
        "player": player_color,
        "player_raw": player,  # Add raw value for debugging
        "legal_moves": legal_moves,
        "winner": winner,
        "model_move": model_move,
        "policy": policy_dict,
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
        "win_prob": win_prob,
    }
    
    if verbose >= 1:
        response["debug_info"] = debug_info
    
    return jsonify(response)

@app.route("/api/computer_move", methods=["POST"])
def api_computer_move():
    """Make one computer move and return the new state."""
    data = request.get_json()
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")
    search_widths = data.get("search_widths", None)
    temperature = data.get("temperature", 1.0)  # Default temperature
    verbose = data.get("verbose", 0)  # Verbose level: 0=none, 1=basic, 2=detailed, 3=full
    
    if not trmph:
        return jsonify({"error": "TRMPH required"}), 400
    
    result = make_computer_move(trmph, model_id, search_widths, temperature, verbose)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 500

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