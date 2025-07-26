from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import numpy as np
from flask_cors import CORS
import logging

from hex_ai.utils import format_conversion as fc
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.value_utils import Winner, winner_to_color, get_policy_probs_from_logits, get_win_prob_from_model_output, temperature_scaled_softmax

# Model checkpoint defaults
ALL_RESULTS_DIR = "checkpoints/hyperparameter_tuning/"
THIS_MODEL_DIR = "loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
CHECKPOINT_FILE = "epoch1_mini35.pt"
CHECKPOINT_FILE1 = "epoch1_mini1.pt"
CHECKPOINT_FILE2 = "epoch2_mini10.pt"
DEFAULT_CHKPT_PATH = f"{ALL_RESULTS_DIR}/{THIS_MODEL_DIR}/{CHECKPOINT_FILE}"
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
        {"id": "model1", "name": "Model 1 (epoch1_mini1)", "path": MODEL_PATHS["model1"]},
        {"id": "model2", "name": "Model 2 (epoch2_mini10)", "path": MODEL_PATHS["model2"]},
    ]

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
                    state, model, search_widths, batch_size=1000
                )
                if best_move is not None:
                    best_move_trmph = fc.rowcol_to_trmph(*best_move)
                else:
                    # Fallback to policy-based move
                    policy_logits, _ = model.infer(trmph)
                    # Apply temperature scaling to policy
                    policy_probs = temperature_scaled_softmax(policy_logits, temperature)
                    legal_moves = state.get_legal_moves()
                    legal_indices = [fc.trmph_to_tensor(fc.rowcol_to_trmph(row, col)) for row, col in legal_moves]
                    best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
                    best_move_trmph = fc.tensor_to_trmph(best_idx)
            except Exception as e:
                app.logger.warning(f"Tree search failed, falling back to policy: {e}")
                # Fallback to policy-based move
                policy_logits, _ = model.infer(trmph)
                # Apply temperature scaling to policy
                policy_probs = temperature_scaled_softmax(policy_logits, temperature)
                legal_moves = state.get_legal_moves()
                legal_indices = [fc.trmph_to_tensor(fc.rowcol_to_trmph(row, col)) for row, col in legal_moves]
                best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
                best_move_trmph = fc.tensor_to_trmph(best_idx)
        else:
            # Simple policy-based move
            policy_logits, _ = model.infer(trmph)
            # Apply temperature scaling to policy
            policy_probs = temperature_scaled_softmax(policy_logits, temperature)
            legal_moves = state.get_legal_moves()
            legal_indices = [fc.trmph_to_tensor(fc.rowcol_to_trmph(row, col)) for row, col in legal_moves]
            best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
            best_move_trmph = fc.tensor_to_trmph(best_idx)
        
        # Apply the move
        row, col = fc.trmph_move_to_rowcol(best_move_trmph)
        state = state.make_move(row, col)
        
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
        # Apply temperature scaling to policy
        policy_probs = temperature_scaled_softmax(policy_logits, temperature)
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

@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json()
    trmph = data.get("trmph")
    move = data.get("move")
    model_id = data.get("model_id", "model1")
    search_widths = data.get("search_widths", None)
    temperature = data.get("temperature", 1.0)  # Default temperature
    
    try:
        state = HexGameState.from_trmph(trmph)
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400
    
    try:
        row, col = fc.trmph_move_to_rowcol(move)
        state = state.make_move(row, col)
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
            model = get_model(model_id)
            
            # Use tree search if search_widths provided
            if search_widths and len(search_widths) > 0:
                try:
                    best_move, _ = minimax_policy_value_search(
                        state, model, search_widths, batch_size=1000
                    )
                    if best_move is not None:
                        best_move_trmph = fc.rowcol_to_trmph(*best_move)
                    else:
                        # Fallback to policy-based move
                        policy_logits, _ = model.infer(new_trmph)
                        # Apply temperature scaling to policy
                        policy_probs = temperature_scaled_softmax(policy_logits, temperature)
                        legal_indices = [fc.trmph_to_tensor(m) for m in legal_moves]
                        best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
                        best_move_trmph = fc.tensor_to_trmph(best_idx)
                except Exception as e:
                    app.logger.warning(f"Tree search failed, falling back to policy: {e}")
                    # Fallback to policy-based move
                    policy_logits, _ = model.infer(new_trmph)
                    # Apply temperature scaling to policy
                    policy_probs = temperature_scaled_softmax(policy_logits, temperature)
                    legal_indices = [fc.trmph_to_tensor(m) for m in legal_moves]
                    best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
                    best_move_trmph = fc.tensor_to_trmph(best_idx)
            else:
                # Simple policy-based move
                policy_logits, _ = model.infer(new_trmph)
                # Apply temperature scaling to policy
                policy_probs = temperature_scaled_softmax(policy_logits, temperature)
                legal_indices = [fc.trmph_to_tensor(m) for m in legal_moves]
                best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
                best_move_trmph = fc.tensor_to_trmph(best_idx)
            
            # Apply model move
            row, col = fc.trmph_move_to_rowcol(best_move_trmph)
            state = state.make_move(row, col)
            model_move = best_move_trmph
            new_trmph = state.to_trmph()
            board = state.board.tolist()
            player = state.current_player
            legal_moves = moves_to_trmph(state.get_legal_moves())
            winner = state.winner
            
        except Exception as e:
            app.logger.error(f"Model move failed: {e}")
            # Continue without model move if there's an error
    
    # Recompute policy/value for final state
    try:
        model = get_model(model_id)
        policy_logits, value_logit = model.infer(new_trmph)
        # Apply temperature scaling to policy
        policy_probs = temperature_scaled_softmax(policy_logits, temperature)
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
        "model_move": model_move,
        "policy": policy_dict,
        "value": float(value_logit) if 'value_logit' in locals() else 0.0,
        "win_prob": win_prob,
    })

@app.route("/api/computer_move", methods=["POST"])
def api_computer_move():
    """Make one computer move and return the new state."""
    data = request.get_json()
    trmph = data.get("trmph")
    model_id = data.get("model_id", "model1")
    search_widths = data.get("search_widths", None)
    temperature = data.get("temperature", 1.0)  # Default temperature
    
    if not trmph:
        return jsonify({"error": "TRMPH required"}), 400
    
    result = make_computer_move(trmph, model_id, search_widths, temperature)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 500

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(os.path.dirname(__file__)), "favicon2_transp_cropped.png")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), path)

@app.route("/")
def serve_index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), "index.html")

if __name__ == "__main__" or __name__ == "hex_ai.web.app":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, port=5001) 