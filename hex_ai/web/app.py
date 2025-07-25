from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import numpy as np
from flask_cors import CORS
import logging

from hex_ai.utils import format_conversion as fc
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.value_utils import Winner, winner_to_color, get_policy_probs_from_logits, get_win_prob_from_model_output

# Model checkpoint defaults (from scripts/play_vs_model_cli.py)
ALL_RESULTS_DIR = "checkpoints/hyperparameter_tuning/"
THIS_MODEL_DIR = "loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
CHECKPOINT_FILE = "epoch1_mini35.pt"
DEFAULT_CHKPT_PATH = f"{ALL_RESULTS_DIR}/{THIS_MODEL_DIR}/{CHECKPOINT_FILE}"

app = Flask(__name__, static_folder="static")
CORS(app)

# Add debug logging for incoming requests
@app.before_request
def log_request_info():
    app.logger.debug(f"Request: {request.method} {request.path}")

MODEL = None
MODEL_PATH = os.environ.get("HEX_MODEL_PATH", DEFAULT_CHKPT_PATH)
MODEL_INFER = None

# --- Model loading ---
def get_model():
    global MODEL_INFER
    if MODEL_INFER is None:
        MODEL_INFER = SimpleModelInference(MODEL_PATH)
    return MODEL_INFER

# --- Utility: Convert (row, col) moves to trmph moves ---
def moves_to_trmph(moves):
    return [fc.rowcol_to_trmph(row, col) for row, col in moves]

@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json()
    trmph = data.get("trmph")
    try:
        state = HexGameState.from_trmph(trmph)
    except Exception as e:
        return jsonify({"error": f"Invalid TRMPH: {e}"}), 400

    board = state.board.tolist()
    player = state.current_player
    legal_moves = moves_to_trmph(state.get_legal_moves())
    winner = state.winner  # "blue", "red", or None

    # Model inference
    model = get_model()
    policy_logits, value_logit = model.infer(trmph)
    policy_probs = get_policy_probs_from_logits(policy_logits)
    # Map policy to trmph moves
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    # Win probability for current player
    win_prob = get_win_prob_from_model_output(value_logit, winner_to_color(player))

    return jsonify({
        "board": board,
        "player": winner_to_color(player),
        "legal_moves": legal_moves,
        "winner": winner,  # already a string or None
        "policy": policy_dict,
        "value": float(value_logit),
        "win_prob": win_prob,
        "trmph": trmph,
    })

@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json()
    trmph = data.get("trmph")
    move = data.get("move")
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
        model = get_model()
        policy_logits, value_logit = model.infer(new_trmph)
        policy_probs = get_policy_probs_from_logits(policy_logits)
        # Pick the move with highest probability among legal moves
        legal_indices = [fc.trmph_to_tensor(m) for m in legal_moves]
        best_idx = max(legal_indices, key=lambda idx: policy_probs[idx])
        best_move = fc.tensor_to_trmph(best_idx)
        # Apply model move
        row, col = fc.trmph_move_to_rowcol(best_move)
        state = state.make_move(row, col)
        model_move = best_move
        new_trmph = state.to_trmph()
        board = state.board.tolist()
        player = state.current_player
        legal_moves = moves_to_trmph(state.get_legal_moves())
        winner = state.winner
        # Recompute policy/value for new state
        policy_logits, value_logit = model.infer(new_trmph)
        policy_probs = get_policy_probs_from_logits(policy_logits)
    else:
        # If game is over, still run inference for display
        model = get_model()
        policy_logits, value_logit = model.infer(new_trmph)
        policy_probs = get_policy_probs_from_logits(policy_logits)

    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_prob = get_win_prob_from_model_output(value_logit, winner_to_color(player))

    return jsonify({
        "new_trmph": new_trmph,
        "board": board,
        "player": winner_to_color(player),
        "legal_moves": legal_moves,
        "winner": winner,
        "model_move": model_move,
        "policy": policy_dict,
        "value": float(value_logit),
        "win_prob": win_prob,
    })

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