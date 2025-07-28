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
CHECKPOINT_FILE1 = "epoch1_mini30.pt"
CHECKPOINT_FILE2 = "epoch2_mini20.pt"
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
                       search_widths, temperature, verbose, model_move=None):
    """Generate comprehensive debug information based on verbose level."""
    debug_info = {}
    
    # Level 1: Basic policy and value analysis
    if verbose >= 1:
        debug_info["basic"] = {
            "current_player": winner_to_color(state.current_player),
            "game_over": state.game_over,
            "legal_moves_count": len(state.get_legal_moves()),
            "value_logit": float(value_logit),
            "win_probability": float(get_win_prob_from_model_output(value_logit, winner_to_color(state.current_player))),
            "temperature": temperature,
            "search_widths": search_widths,
            "model_move": model_move
        }
        
        # Top-k policy moves
        legal_moves = state.get_legal_moves()
        move_probs = []
        for row, col in legal_moves:
            move_trmph = fc.rowcol_to_trmph(row, col)
            prob = float(policy_probs[row * state.board.shape[0] + col])
            move_probs.append({"move": move_trmph, "row": row, "col": col, "probability": prob})
        
        # Sort by probability descending
        move_probs.sort(key=lambda x: x["probability"], reverse=True)
        debug_info["policy_analysis"] = {
            "top_moves": move_probs[:10],  # Top 10 moves
            "total_legal_moves": len(legal_moves)
        }
    
    # Level 2: Detailed analysis including tree search
    if verbose >= 2 and search_widths and len(search_widths) > 0:
        try:
            from hex_ai.inference.fixed_tree_search import minimax_policy_value_search, build_search_tree, evaluate_leaf_nodes, minimax_backup
            
            # Build search tree for analysis
            root = build_search_tree(state, model, search_widths, temperature)
            evaluate_leaf_nodes([root], model, batch_size=1000, root_player=state.current_player)
            final_value = minimax_backup(root)
            
            debug_info["tree_search"] = {
                "search_widths": search_widths,
                "tree_depth": len(search_widths),
                "final_value": float(final_value),
                "best_move": fc.rowcol_to_trmph(*root.best_move) if root.best_move else None,
                "tree_size": count_tree_nodes(root)
            }
            
            # Add terminal node analysis
            if verbose >= 3:
                debug_info["tree_search"]["terminal_nodes"] = collect_terminal_nodes(root)
                
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
def make_computer_move(trmph, model_id, search_widths=None, temperature=1.0):
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
        
        # Use tree search if search_widths provided, otherwise use simple policy
        if search_widths and len(search_widths) > 0:
            try:
                best_move, _ = minimax_policy_value_search(
                    state, model, search_widths, batch_size=1000, temperature=temperature
                )
                if best_move is not None:
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
                else:
                    # Fallback to policy-based move using centralized utilities
                    best_move = select_policy_move(state, model, temperature)
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
            except Exception as e:
                app.logger.warning(f"Tree search failed, falling back to policy: {e}")
                # Fallback to policy-based move using centralized utilities
                best_move = select_policy_move(state, model, temperature)
                best_move_trmph = fc.rowcol_to_trmph(*best_move)
        else:
            # Simple policy-based move using centralized utilities
            best_move = select_policy_move(state, model, temperature)
            best_move_trmph = fc.rowcol_to_trmph(*best_move)
        
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
            "game_over": state.game_over
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

    # Model inference
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(trmph)
        # Apply temperature scaling to policy using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        # Map policy to trmph moves
        policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
        # Win probability for current player
        win_prob = get_win_prob_from_model_output(value_logit, winner_to_color(player))
    except Exception as e:
        app.logger.error(f"Model inference failed: {e}")
        policy_dict = {}
        win_prob = 0.5

    return jsonify({
        "board": board,
        "player": winner_to_color(player),
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

    # Recompute policy/value for the new state
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(new_trmph)
        # Apply temperature scaling to policy using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
        win_prob = get_win_prob_from_model_output(value_logit, winner_to_color(player))
    except Exception as e:
        app.logger.error(f"Final inference failed: {e}")
        policy_dict = {}
        win_prob = 0.5

    return jsonify({
        "new_trmph": new_trmph,
        "board": board,
        "player": winner_to_color(player),
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
    verbose = data.get("verbose", 0)  # Verbose level: 0=none, 1=basic, 2=detailed, 3=full
    
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
    # If game not over and it's model's turn, have model pick a move
    if not state.game_over:
        try:
            # Determine which player's settings to use for the computer move
            # The current player after the human move determines whose settings to use
            current_player_color = winner_to_color(player)
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
            
            # Use tree search if search_widths provided
            if computer_search_widths and len(computer_search_widths) > 0:
                try:
                    best_move, _ = minimax_policy_value_search(
                        state, model, computer_search_widths, batch_size=1000, temperature=computer_temperature
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
            
        except Exception as e:
            app.logger.error(f"Model move failed: {e}")
            # Continue without model move if there's an error
    
    # Recompute policy/value for final state using centralized utilities
    debug_info = {}
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(new_trmph)
        # Apply temperature scaling to policy using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
        win_prob = get_win_prob_from_model_output(value_logit, winner_to_color(player))
        
        # Add verbose debug information
        if verbose >= 1:
            debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs, 
                                          search_widths, temperature, verbose, model_move)
    except Exception as e:
        app.logger.error(f"Final inference failed: {e}")
        policy_dict = {}
        win_prob = 0.5

    response = {
        "new_trmph": new_trmph,
        "board": board,
        "player": winner_to_color(player),
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
    
    result = make_computer_move(trmph, model_id, search_widths, temperature)
    
    # Add verbose debug information if requested
    if verbose >= 1 and result["success"]:
        try:
            state = HexGameState.from_trmph(result["new_trmph"])
            model = get_model(model_id)
            policy_logits, value_logit = model.infer(result["new_trmph"])
            policy_probs = policy_logits_to_probs(policy_logits, temperature)
            
            debug_info = generate_debug_info(state, model, policy_logits, value_logit, policy_probs,
                                          search_widths, temperature, verbose, result.get("move_made"))
            result["debug_info"] = debug_info
        except Exception as e:
            app.logger.error(f"Debug info generation failed: {e}")
            result["debug_info"] = {"error": str(e)}
    
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
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, port=5001) 