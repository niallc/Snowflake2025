from flask import Flask, request, jsonify, send_from_directory, g
import os
from flask_cors import CORS
import json
import logging
import time
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import string
import uuid
import random
import threading
import numpy as np

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import HexGameState, HexGameEngine, apply_move_to_state_trmph
from hex_ai.inference.simple_model_inference import SimpleModelInference

from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, run_mcts_move, create_mcts_config
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
from hex_ai.inference.model_config import get_model_path, get_model_info, get_all_model_info, register_model, is_valid_model_id, get_normalized_path, get_model_path_with_fallback, get_available_model_with_fallback
from hex_ai.inference.model_cache import get_model_cache
from hex_ai.web.web_config import INTERACTIVE_CONFIDENCE_TERMINATION_THRESHOLD
from hex_ai.web.move_heatmap import build_policy_value_heatmap

app = Flask(__name__, static_folder="static_public")
CORS(app)

# =============================================================================
# ANALYTICS CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ANALYTICS_LOG_PATH = BASE_DIR / "logs" / "web_usage.jsonl"

ANALYTICS_ENABLED = os.getenv("SF25_ANALYTICS_ENABLED", "1").lower() not in ("0", "false", "no")
ANALYTICS_LOG_PATH = os.getenv("SF25_ANALYTICS_LOG_PATH", str(DEFAULT_ANALYTICS_LOG_PATH))
ANALYTICS_SALT = os.getenv("SF25_ANALYTICS_SALT", "")
ANALYTICS_HASH_IP = os.getenv("SF25_ANALYTICS_HASH_IP", "1").lower() not in ("0", "false", "no")
ANALYTICS_COOKIE_NAME = os.getenv("SF25_ANALYTICS_COOKIE_NAME", "sf25_cid")
ANALYTICS_COOKIE_DAYS = int(os.getenv("SF25_ANALYTICS_COOKIE_DAYS", "365"))
ANALYTICS_SEQUENCE_TTL_SECONDS = int(os.getenv("SF25_ANALYTICS_SEQUENCE_TTL_SECONDS", "3600"))
ANALYTICS_SEQUENCE_MAX_CLIENTS = int(os.getenv("SF25_ANALYTICS_SEQUENCE_MAX_CLIENTS", "10000"))

class MonthlyJsonlHandler(logging.Handler):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = Path(base_path)
        self._current_month = None
        self._stream = None
        self._open_for_datetime(datetime.now(timezone.utc))

    def _path_for_datetime(self, dt: datetime) -> Path:
        stem = self.base_path.stem
        suffix = self.base_path.suffix or ".jsonl"
        return self.base_path.with_name(f"{stem}_{dt:%Y_%m}{suffix}")

    def _open_for_datetime(self, dt: datetime) -> None:
        path = self._path_for_datetime(dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = open(path, "a", encoding="utf-8")
        self._current_month = (dt.year, dt.month)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            now = datetime.now(timezone.utc)
            if self._current_month != (now.year, now.month):
                self._open_for_datetime(now)
            msg = self.format(record)
            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
        super().close()

analytics_logger = logging.getLogger("sf25.analytics")
if ANALYTICS_ENABLED and not analytics_logger.handlers:
    try:
        handler = MonthlyJsonlHandler(ANALYTICS_LOG_PATH)
        handler.setFormatter(logging.Formatter("%(message)s"))
        analytics_logger.setLevel(logging.INFO)
        analytics_logger.addHandler(handler)
        analytics_logger.propagate = False
        if not ANALYTICS_SALT:
            app.logger.warning("SF25_ANALYTICS_SALT not set; IP hashes are less private.")
        app.logger.info(f"Web analytics enabled (monthly files): {ANALYTICS_LOG_PATH}")
    except Exception as e:
        ANALYTICS_ENABLED = False
        app.logger.warning(f"Web analytics disabled (init failed): {e}")

# =============================================================================
# DIFFICULTY CONFIGURATION CONSTANTS
# =============================================================================

# Difficulty configuration constants
MIN_ELO = 1
MAX_ELO = 2350
DEFAULT_ELO = 500

# Native virtual-board support: play on top-left KxK while model still uses BOARD_SIZE.
MIN_DISPLAY_BOARD_SIZE = 2
DEFAULT_DISPLAY_BOARD_SIZE = BOARD_SIZE
DISPLAY_BOARD_SIZE_OPTIONS = list(range(BOARD_SIZE, MIN_DISPLAY_BOARD_SIZE - 1, -1))

# Derived from legacy_code/FileConversion.py and existing rules.html guidance.
# Values are bare TRMPH move strings (no "#13," prefix).
VIRTUAL_BOARD_PREFILL_MOVES = {
    13: "",
    12: "a13m1b13m2c13m3d13m4e13m5f13m6g13m7h13m8i13m9j13m10k13m11l13m12",
    11: "a12l1b12l2c12l3d12l4e12l5f12l6g12l7h12l8i12l9j12l10k12l11k13m11",
    10: "a11k1b11k2c11k3d11k4e11k5f11k6g11k7h11k8i11k9j11k10j12l10j13m10",
    9: "a10j1b10j2c10j3d10j4e10j5f10j6g10j7h10j8i10j9i11k9i12l9i13m9",
    8: "a9i1b9i2c9i3d9i4e9i5f9i6g9i7h9i8h10j8h11k8h12l8h13m8",
    7: "a8h1b8h2c8h3d8h4e8h5f8h6g8h7g9i7g10j7g11k7g12l7g13m7",
    6: "a7g1b7g2c7g3d7g4e7g5f7g6f8h6f9i6f10j6f11k6f12l6f13m6",
    5: "a6f1b6f2c6f3d6f4e6f5e7g5e8h5e9i5e10j5e11k5e12l5e13m5",
    4: "a5e1b5e2c5e3d5e4d6f4d7g4d8h4d9i4d10j4d11k4d12l4d13m4",
    3: "a4d1b4d2c4d3c5e3c6f3c7g3c8h3c9i3c10j3c11k3c12l3c13m3",
    2: "a3c1b3c2b4d2b5e2b6f2b7g2b8h2b9i2b10j2b11k2b12l2b13m2",
}

_DISPLAY_MASK_CACHE = {}

def validate_display_board_size(value) -> int:
    """Validate and normalize requested display-board size."""
    if value is None:
        return DEFAULT_DISPLAY_BOARD_SIZE

    try:
        size = int(value)
    except (TypeError, ValueError):
        raise ValueError("display_board_size must be an integer")

    if not (MIN_DISPLAY_BOARD_SIZE <= size <= BOARD_SIZE):
        raise ValueError(
            f"display_board_size must be between {MIN_DISPLAY_BOARD_SIZE} and {BOARD_SIZE}"
        )
    return size

def get_virtual_prefill_moves(display_board_size: int) -> str:
    """Get prefill move sequence that makes KxK play equivalent on BOARD_SIZE board."""
    try:
        return VIRTUAL_BOARD_PREFILL_MOVES[display_board_size]
    except KeyError as e:
        raise ValueError(f"Unsupported display_board_size: {display_board_size}") from e

def _strip_virtual_prefill_prefix_if_present(bare_moves: str, display_board_size: int) -> str:
    """
    Strip the known virtual-board prefill prefix if present.

    This allows old copy-paste workflows to continue working.
    """
    prefill = get_virtual_prefill_moves(display_board_size)
    if prefill and bare_moves.startswith(prefill):
        return bare_moves[len(prefill):]
    return bare_moves

def validate_moves_within_display_board(bare_moves: str, display_board_size: int, field_name: str) -> None:
    """Ensure all moves are inside the top-left display board."""
    if not bare_moves:
        return

    try:
        moves = fc.split_trmph_moves(bare_moves)
    except ValueError as e:
        raise ValueError(f"Invalid {field_name} sequence: {e}") from e

    max_moves = display_board_size * display_board_size
    if len(moves) > max_moves:
        raise ValueError(
            f"{field_name} has too many moves for {display_board_size}x{display_board_size} "
            f"(maximum {max_moves})"
        )

    for move in moves:
        row, col = fc.trmph_move_to_rowcol(move, board_size=BOARD_SIZE)
        if row >= display_board_size or col >= display_board_size:
            raise ValueError(
                f"Move '{move}' is outside top-left {display_board_size}x{display_board_size} display board"
            )

def normalize_user_trmph_for_display(trmph_text: str, display_board_size: int, field_name: str = "trmph") -> str:
    """
    Normalize a TRMPH string into user-move-only bare moves for the chosen display size.

    Returns bare moves without preamble and without virtual-board prefill.
    """
    bare_moves = fc.strip_trmph_preamble((trmph_text or "").strip())
    bare_moves = _strip_virtual_prefill_prefix_if_present(bare_moves, display_board_size)
    validate_moves_within_display_board(bare_moves, display_board_size, field_name)
    return bare_moves

def compose_full_trmph_from_user_trmph(user_bare_moves: str, display_board_size: int) -> str:
    """Compose full BOARD_SIZE TRMPH by prepending the configured virtual-board prefill."""
    prefill = get_virtual_prefill_moves(display_board_size)
    return f"#{BOARD_SIZE},{prefill}{user_bare_moves}"

def state_to_user_trmph(state: HexGameState, display_board_size: int) -> str:
    """Convert full state TRMPH into user-visible move sequence (without prefill)."""
    bare_moves = fc.strip_trmph_preamble(state.to_trmph())
    prefill = get_virtual_prefill_moves(display_board_size)
    if prefill and bare_moves.startswith(prefill):
        return bare_moves[len(prefill):]
    return bare_moves

def get_display_legal_move_mask(display_board_size: int) -> np.ndarray:
    """Return cached legal-move mask for top-left display board."""
    cached = _DISPLAY_MASK_CACHE.get(display_board_size)
    if cached is not None:
        return cached

    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    mask[:display_board_size, :display_board_size] = True
    _DISPLAY_MASK_CACHE[display_board_size] = mask
    return mask

def apply_display_mask_to_state(state: HexGameState, display_board_size: int) -> HexGameState:
    """Apply top-left display-board legal mask to a game state."""
    state.set_legal_move_mask(get_display_legal_move_mask(display_board_size))
    return state

# =============================================================================
# PIE RULE CONFIGURATION
# =============================================================================

DEFAULT_PIE_RULE_ENABLED = True
PIE_RULE_SWAP_ALPHA = 4.0
PIE_RULE_FORCE_NO_SWAP_AT = 0.01
PIE_RULE_FORCE_SWAP_AT = 0.99

_PIE_RULE_OPENING_CACHE = {}
_PIE_RULE_OPENING_CACHE_LOCK = threading.Lock()


def _extract_model_epoch_mini(model_path: str):
    """Extract epoch/mini identifiers from checkpoint filename when present."""
    match = re.search(r"epoch(\d+)_mini(\d+)", os.path.basename(model_path))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _build_pie_rule_model_identity(model_id: str):
    """Build stable cache identity for a model ID."""
    if not is_valid_model_id(model_id):
        raise ValueError(f"Unknown model_id for pie rule cache: {model_id}")
    model_path = get_model_path_with_fallback(model_id)
    normalized_path = get_normalized_path(model_path)
    try:
        model_mtime_ns = os.stat(normalized_path).st_mtime_ns
    except OSError:
        model_mtime_ns = None
    epoch, mini = _extract_model_epoch_mini(normalized_path)
    identity_key = f"{normalized_path}|mtime_ns={model_mtime_ns}"
    return {
        "model_id": model_id,
        "model_path": normalized_path,
        "model_mtime_ns": model_mtime_ns,
        "epoch": epoch,
        "mini": mini,
        "identity_key": identity_key,
    }


def _compute_pie_rule_opening_scores(model_id: str, display_board_size: int):
    """Compute Blue win probability for every legal first move on the display board."""
    state = create_game_state_from_trmph(
        "",
        display_board_size=display_board_size,
        context="for pie-rule opening cache",
    )
    model = get_model(model_id)
    heatmap = build_policy_value_heatmap(
        state=state,
        model=model,
        selection_mode="all_legal",
        top_k=None,
        policy_temperature=1.0,
    )
    return heatmap.scores


def _get_pie_rule_opening_scores(model_id: str, display_board_size: int):
    """
    Get cached opening first-move scores, computing once per model identity and board size.

    Cache is process-local (in-memory) for MVP.
    """
    model_identity = _build_pie_rule_model_identity(model_id)
    cache_key = (
        f"{model_identity['identity_key']}|display_board_size={display_board_size}"
    )

    with _PIE_RULE_OPENING_CACHE_LOCK:
        cached = _PIE_RULE_OPENING_CACHE.get(cache_key)
    if cached is not None:
        cache_meta = dict(cached)
        cache_meta["cache_hit"] = True
        return cached["scores"], cache_meta

    scores = _compute_pie_rule_opening_scores(model_id, display_board_size)
    new_entry = {
        "scores": scores,
        "model_id": model_identity["model_id"],
        "model_path": model_identity["model_path"],
        "model_mtime_ns": model_identity["model_mtime_ns"],
        "epoch": model_identity["epoch"],
        "mini": model_identity["mini"],
        "computed_at_unix": time.time(),
    }

    with _PIE_RULE_OPENING_CACHE_LOCK:
        existing = _PIE_RULE_OPENING_CACHE.get(cache_key)
        if existing is None:
            _PIE_RULE_OPENING_CACHE[cache_key] = new_entry
            cache_meta = dict(new_entry)
            cache_meta["cache_hit"] = False
            return scores, cache_meta

    # Another thread won the race and filled cache while we were computing.
    cache_meta = dict(existing)
    cache_meta["cache_hit"] = True
    return existing["scores"], cache_meta


def _pie_rule_swap_probability_from_opening_prob(opening_win_prob: float) -> float:
    """
    Convert opening move Blue win probability to swap probability.

    Formula:
      p_swap = p^alpha / (p^alpha + (1-p)^alpha)
    where p is Blue win probability after the opening move.
    """
    p = max(0.0, min(1.0, float(opening_win_prob)))
    if p <= PIE_RULE_FORCE_NO_SWAP_AT:
        return 0.0
    if p >= PIE_RULE_FORCE_SWAP_AT:
        return 1.0

    p_alpha = p ** PIE_RULE_SWAP_ALPHA
    q_alpha = (1.0 - p) ** PIE_RULE_SWAP_ALPHA
    denom = p_alpha + q_alpha
    if denom <= 0:
        return 0.5
    return p_alpha / denom


def _get_first_user_move_from_trmph(trmph: str):
    """Return first user-visible move from a user TRMPH string, or None."""
    bare = fc.strip_trmph_preamble((trmph or "").strip())
    if not bare:
        return None
    try:
        moves = fc.split_trmph_moves(bare)
    except ValueError:
        return None
    if not moves:
        return None
    return moves[0]


def _is_pie_rule_swap_window(state: HexGameState, trmph: str, pie_rule_enabled: bool) -> bool:
    """Check whether pie-rule swap decision should be evaluated for this position."""
    if not pie_rule_enabled:
        return False
    if state.game_over:
        return False
    move_count = _safe_count_trmph_moves(trmph)
    if move_count != 1:
        return False
    # With virtual-board prefill, one visible move still maps to Red to move.
    if state.current_player_enum != Player.RED:
        return False
    return True


def evaluate_pie_rule_swap_decision(
    trmph: str,
    display_board_size: int,
    model_id: str,
    pie_rule_enabled: bool,
    state: HexGameState = None,
):
    """Evaluate probabilistic pie-rule swap decision for the first visible move."""
    if state is None:
        move_count = _safe_count_trmph_moves(trmph)
        if move_count != 1:
            return None
        state = create_game_state_from_trmph(
            trmph,
            display_board_size=display_board_size,
            context="for pie-rule decision",
        )

    if not _is_pie_rule_swap_window(state, trmph, pie_rule_enabled):
        return None

    first_move = _get_first_user_move_from_trmph(trmph)
    if not first_move:
        return None

    scores, cache_meta = _get_pie_rule_opening_scores(model_id, display_board_size)
    opening_win_prob = scores.get(first_move)
    if opening_win_prob is None:
        app.logger.warning(
            "Pie-rule decision: first move %s missing in opening-score cache for model=%s display=%s",
            first_move,
            model_id,
            display_board_size,
        )
        return None

    swap_probability = _pie_rule_swap_probability_from_opening_prob(opening_win_prob)
    swap_roll = random.random()
    should_swap = swap_roll < swap_probability

    return {
        "first_move": first_move,
        "opening_win_probability": float(opening_win_prob),
        "swap_probability": float(swap_probability),
        "swap_roll": float(swap_roll),
        "should_swap": bool(should_swap),
        "model_id": model_id,
        "cache_hit": bool(cache_meta.get("cache_hit", False)),
        "cache_model_path": cache_meta.get("model_path"),
        "cache_model_epoch": cache_meta.get("epoch"),
        "cache_model_mini": cache_meta.get("mini"),
    }


def build_pie_rule_response_fields(
    trmph: str,
    state: HexGameState,
    pie_rule_enabled: bool,
    pie_decision: dict = None,
    pie_rule_action: str = "none",
):
    """Build consistent pie-rule status fields for API responses."""
    response = {
        "pie_rule_enabled": bool(pie_rule_enabled),
        "pie_rule_action": pie_rule_action,
    }

    if pie_decision is None:
        response["pie_rule_can_swap"] = _is_pie_rule_swap_window(
            state, trmph, pie_rule_enabled
        )
        return response

    response.update(
        {
            # Decision has already been made for this response payload.
            "pie_rule_can_swap": False,
            "pie_rule_first_move": pie_decision["first_move"],
            "pie_rule_opening_win_probability": pie_decision["opening_win_probability"],
            "pie_rule_swap_probability": pie_decision["swap_probability"],
            "pie_rule_model_id": pie_decision["model_id"],
            "pie_rule_cache_hit": pie_decision["cache_hit"],
            "pie_rule_cache_model_epoch": pie_decision["cache_model_epoch"],
            "pie_rule_cache_model_mini": pie_decision["cache_model_mini"],
        }
    )
    return response


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

# =============================================================================
# ANALYTICS HELPERS
# =============================================================================

_last_trmph_by_client = {}

def _is_request_secure():
    if request.is_secure:
        return True
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    if forwarded_proto:
        return forwarded_proto.split(",")[0].strip().lower() == "https"
    return False

def _get_client_ip():
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr

def _hash_identifier(value):
    if not value:
        return None
    salt = ANALYTICS_SALT or "sf25"
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
    return digest[:16]

def _safe_count_trmph_moves(trmph_text):
    try:
        return fc.count_trmph_moves(trmph_text)
    except Exception:
        return None

def _update_sequence_info(client_id, trmph):
    if not client_id:
        return {"sequence_continuation": None, "prev_trmph_len": None, "prev_age_ms": None}

    now = time.time()
    prev_entry = _last_trmph_by_client.get(client_id)
    sequence_continuation = None
    prev_trmph_len = None
    prev_age_ms = None

    if prev_entry:
        prev_trmph, prev_ts = prev_entry
        if now - prev_ts <= ANALYTICS_SEQUENCE_TTL_SECONDS:
            prev_bare = fc.strip_trmph_preamble((prev_trmph or "").strip())
            curr_bare = fc.strip_trmph_preamble((trmph or "").strip())
            prev_trmph_len = len(prev_bare)
            prev_age_ms = int((now - prev_ts) * 1000)
            if not prev_bare:
                sequence_continuation = True
            else:
                sequence_continuation = curr_bare.startswith(prev_bare)

    _last_trmph_by_client[client_id] = (trmph, now)

    if len(_last_trmph_by_client) > ANALYTICS_SEQUENCE_MAX_CLIENTS:
        cutoff = now - ANALYTICS_SEQUENCE_TTL_SECONDS
        for key in list(_last_trmph_by_client.keys()):
            _, ts = _last_trmph_by_client[key]
            if ts < cutoff:
                del _last_trmph_by_client[key]
        if len(_last_trmph_by_client) > ANALYTICS_SEQUENCE_MAX_CLIENTS:
            overflow = len(_last_trmph_by_client) - ANALYTICS_SEQUENCE_MAX_CLIENTS
            for key in list(_last_trmph_by_client.keys())[:overflow]:
                del _last_trmph_by_client[key]

    return {
        "sequence_continuation": sequence_continuation,
        "prev_trmph_len": prev_trmph_len,
        "prev_age_ms": prev_age_ms
    }

def _build_trmph_stats(trmph):
    bare = fc.strip_trmph_preamble((trmph or "").strip())
    return {
        "trmph_len": len(bare),
        "trmph_moves": _safe_count_trmph_moves(trmph)
    }

def _prune_none_values(payload):
    return {k: v for k, v in payload.items() if v is not None}

def log_usage_event(event, **fields):
    if not ANALYTICS_ENABLED:
        return
    try:
        status = fields.pop("status", None)
        duration_ms = fields.pop("duration_ms", None)
        if duration_ms is None and hasattr(g, "analytics_start"):
            duration_ms = int((time.time() - g.analytics_start) * 1000)

        client_ip = _get_client_ip()
        ip_value = _hash_identifier(client_ip) if ANALYTICS_HASH_IP else client_ip
        user_agent = request.headers.get("User-Agent", "")
        ua_hash = _hash_identifier(user_agent) if user_agent else None

        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "path": request.path,
            "endpoint": request.endpoint,
            "method": request.method,
            "status": status,
            "duration_ms": duration_ms,
            "client_id": getattr(g, "analytics_client_id", None),
            "ip_hash": ip_value if ANALYTICS_HASH_IP else None,
            "ip": ip_value if not ANALYTICS_HASH_IP else None,
            "ua_hash": ua_hash,
        }
        payload.update(fields)

        analytics_logger.info(json.dumps(_prune_none_values(payload), separators=(",", ":"), default=str))
    except Exception as e:
        app.logger.warning(f"Analytics logging failed: {e}")

@app.before_request
def _analytics_before_request():
    if not ANALYTICS_ENABLED:
        return
    g.analytics_start = time.time()
    client_id = request.cookies.get(ANALYTICS_COOKIE_NAME)
    if not client_id:
        client_id = uuid.uuid4().hex
        g.analytics_set_cookie = True
    else:
        g.analytics_set_cookie = False
    g.analytics_client_id = client_id

@app.after_request
def _analytics_after_request(response):
    if not ANALYTICS_ENABLED:
        return response
    if getattr(g, "analytics_set_cookie", False):
        max_age = ANALYTICS_COOKIE_DAYS * 24 * 3600
        response.set_cookie(
            ANALYTICS_COOKIE_NAME,
            g.analytics_client_id,
            max_age=max_age,
            httponly=True,
            samesite="Lax",
            secure=_is_request_secure()
        )
    return response

# Get centralized model cache
MODEL_CACHE = get_model_cache()

# Preload default models on startup
def preload_default_model():
    """Preload commonly used models to avoid loading delays when switching difficulty."""
    try:
        app.logger.info("Preloading models...")

        # Preload strongest and lower-difficulty models used in DIFFICULTY_POINTS.
        # Keep "simple" for tiers that still reference it.
        for model_id in ["best", "beginner", "simple"]:
            model_path = get_model_path_with_fallback(model_id)
            app.logger.info(f"Preloading {model_id} model from {model_path}")
            MODEL_CACHE.get_simple_model(model_path)
            MODEL_CACHE.get_wrapper_model(model_path)
            app.logger.info(f"Successfully preloaded {model_id} model")
        
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

    display_board_size = DEFAULT_DISPLAY_BOARD_SIZE
    if "display_board_size" in all_allowed:
        try:
            display_board_size = validate_display_board_size(
                data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
            )
            validated_data["display_board_size"] = display_board_size
        except ValueError as e:
            app.logger.warning(f"Validation failed for display_board_size: {e}")
            return False, f"Invalid display_board_size: {e}", None

    # Process all allowed fields present in data
    for field in all_allowed:
        if field not in data or field == "display_board_size":
            continue

        if field in ['trmph', 'move', 'trmph_sequence']:
            # Normalize input first (handle LittleGolem format, swap, etc.)
            try:
                normalized_input = fc.normalize_game_input(data[field])
            except ValueError as e:
                app.logger.warning(f"Normalization failed for {field}: {e}")
                return False, f"Invalid format for {field}: {str(e)}", None
            except Exception as e:
                app.logger.error(f"Input normalization failed for {field}: {e}")
                return False, f"Normalization error for {field}: {str(e)}", None

            # Validate TRMPH syntax first.
            is_valid, error_msg = validate_trmph_input(normalized_input)
            if not is_valid:
                return False, f"Invalid {field} [DEBUG-CHECK]: {error_msg}", None

            # Enforce display-board geometry constraints.
            try:
                if field in {"trmph", "trmph_sequence"}:
                    validated_data[field] = normalize_user_trmph_for_display(
                        normalized_input, display_board_size, field_name=field
                    )
                else:
                    row, col = fc.trmph_move_to_rowcol(normalized_input, board_size=BOARD_SIZE)
                    if row >= display_board_size or col >= display_board_size:
                        return (
                            False,
                            f"Move '{normalized_input}' is outside top-left "
                            f"{display_board_size}x{display_board_size} display board",
                            None,
                        )
                    validated_data[field] = normalized_input
            except ValueError as e:
                app.logger.warning(f"Display-board validation failed for {field}: {e}")
                return False, f"Invalid {field}: {e}", None
        elif field == 'elo_rating':
            # ELO rating needs special validation
            try:
                validated_data[field] = validate_elo_rating(data[field])
            except ValueError as e:
                app.logger.warning(f"Validation failed for {field}: {e}")
                return False, f"Invalid {field}: {e}", None
        elif field == 'pie_rule_enabled':
            # Pie-rule toggle accepts booleans and common string/int forms.
            try:
                validated_data[field] = validate_boolean_flag(data[field], field)
            except ValueError as e:
                app.logger.warning(f"Validation failed for {field}: {e}")
                return False, f"Invalid {field}: {e}", None
        else:
            # Other fields just copy over
            validated_data[field] = data[field]

    # Check required fields are present
    if required_fields:
        for field in required_fields:
            if field not in validated_data:
                return False, f"Missing required field: {field}", None
    
    return True, None, validated_data

def validate_elo_rating(value):
    """
    Validate and convert ELO rating to integer.
    
    Args:
        value: ELO rating value (int, float, str, or None)
        
    Returns:
        int: Validated ELO rating in range [MIN_ELO, MAX_ELO]
        
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
    
    if not (MIN_ELO <= elo_int <= MAX_ELO):
        raise ValueError(f"ELO rating must be between {MIN_ELO} and {MAX_ELO}")
    
    return elo_int


def validate_boolean_flag(value, field_name):
    """Validate and normalize a boolean feature flag from JSON input."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(
        f"{field_name} must be a boolean (or one of true/false, 1/0)"
    )

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
    'api_state': 0.5,                # Read-only, cheap (40 burst calls)
    'api_apply_move': 0.5,           # Simple move application (20 burst calls)
    'api_policy_move': 1.0,          # Model inference (20 burst calls, 2/sec sustained)
    'api_apply_trmph_sequence': 5.0, # Batch operation (10 burst calls)
    'api_mcts_move': 2,            # Expensive MCTS (4 burst calls, 2/sec sustained)
    'api_constants': 0.1,            # Constants endpoint (very cheap)
    'api_move_heatmap': 8.0,         # Heavy batch of value inferences
}

def rate_limit(cost: float):
    """
    Token bucket rate limiting decorator with weighted costs.
    
    Args:
        cost (float): Number of tokens this request costs.
        
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

def get_model(model_id="best"):
    """Get or create a model instance for the given model_id using centralized cache with fallback support."""
    app.logger.debug(f"get_model called with model_id: {model_id}")
    
    # Use centralized model configuration with fallback support
    if is_valid_model_id(model_id):
        model_path = get_model_path_with_fallback(model_id)
        app.logger.debug(f"Found model {model_id} -> {model_path}")
        return MODEL_CACHE.get_simple_model(model_path)
    
    app.logger.error(f"Unknown model_id: {model_id}")
    raise ValueError(f"Unknown model_id: {model_id}")

def get_cached_model_wrapper(model_id: str):
    """Get or create a cached ModelWrapper instance for the given model_id using centralized cache with fallback support."""
    app.logger.debug(f"get_cached_model_wrapper called with model_id: {model_id}")
    
    # Get the model path for this model_id with fallback support
    if is_valid_model_id(model_id):
        model_path = get_model_path_with_fallback(model_id)
        app.logger.debug(f"Getting ModelWrapper for path: {model_path}")
        return MODEL_CACHE.get_wrapper_model(model_path)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

# =============================================================================
# DIFFICULTY LEVEL MAPPING
# =============================================================================

# Define difficulty breakpoints for linear interpolation
# Format: (elo, temperature, num_simulations, algorithm, model, label)
DIFFICULTY_POINTS = [
    (MIN_ELO, 1.2,  0 , "policy", "beginner", "Very Easy"),
    (300    , 0.7,  0 , "policy", "beginner", "Beginner"),  
    (500    , 0.15, 0 , "policy", "beginner", "Novice 1"),
    (800    , 1.00, 0 , "policy", "simple",   "Novice 2"),
    (1199   , 0.50, 0 , "policy", "simple",   "Medium"),
    (1200   , 0.15, 0 , "policy", "simple",   "Spicy"),     
    (1500   , 0.75, 0 , "policy", "best",     "Hard"),
    (1800   , 0.55, 0 , "policy", "best",     "Very Hard"),
    (2100   , 0.12, 0 , "policy", "best",     "Expert 1"),
    (2149   , 0.08, 0 , "policy", "best",     "Expert 2"),
    (2150   , 1.0,  8 , "mcts",   "best",     "Extra Hard"), # Gumbel MCTS
    (2250   , 1.0,  20, "mcts",   "best",     "Ultra Hard"), # Gumbel MCTS
    (MAX_ELO, 1.0,  39, "mcts",   "best",     "Master"),     # Gumbel MCTS
]

def get_difficulty_levels():
    """Extract difficulty levels from DIFFICULTY_POINTS for frontend use."""
    return [{"elo": elo, "label": label} for elo, _, _, _, _, label in DIFFICULTY_POINTS]

def get_difficulty_parameters(elo_rating):
    """Convert ELO rating to appropriate algorithm parameters."""
    if elo_rating < MIN_ELO:
        elo_rating = MIN_ELO
    elif elo_rating > MAX_ELO:
        elo_rating = MAX_ELO
    
    # Use the global DIFFICULTY_POINTS array
    difficulty_points = [(elo, temp, sims, algo, model) for elo, temp, sims, algo, model, _ in DIFFICULTY_POINTS]
    
    # Find the appropriate segment for linear interpolation
    for i in range(len(difficulty_points) - 1):
        elo_low, temp_low, sims_low, algo_low, model_low = difficulty_points[i]
        elo_high, temp_high, sims_high, algo_high, model_high = difficulty_points[i + 1]
        
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
            
            # Model selection: use discrete cutoff (use lower model until threshold reached)
            model = model_low if elo_rating < elo_high else model_high
            
            if algorithm == "policy":
                return {
                    "algorithm": "policy",
                    "temperature": temperature,
                    "num_simulations": 0,
                    "exploration_constant": 0,
                    "enable_gumbel": False,
                    "model": model
                }
            else:  # mcts
                return {
                    "algorithm": "mcts",
                    "temperature": 1.0,
                    "temperature_end": 0.1,
                    "num_simulations": num_simulations,
                    "exploration_constant": 2.8,
                    "enable_gumbel": True,
                    "gumbel_max_sims": 500,
                    "model": model
                }
    
    # Fallback (should not reach here with proper bounds checking)
    return {
        "algorithm": "policy",
        "temperature": 0.5,
        "num_simulations": 0,
        "exploration_constant": 0,
        "enable_gumbel": False,
        "model": "best"
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_game_state_from_trmph(trmph, display_board_size=DEFAULT_DISPLAY_BOARD_SIZE, context=""):
    """
    Create a masked game state for the selected display-board size.
    
    Args:
        trmph (str): User-visible TRMPH string (without virtual-board prefill)
        display_board_size (int): Top-left playable board size
        context (str): Context for logging (e.g., "for MCTS move")
        
    Returns:
        HexGameState: The game state
        
    Raises:
        Exception: If TRMPH is invalid
    """
    display_board_size = validate_display_board_size(display_board_size)
    user_bare_moves = normalize_user_trmph_for_display(
        trmph, display_board_size, field_name="trmph"
    )
    full_trmph = compose_full_trmph_from_user_trmph(user_bare_moves, display_board_size)

    state = HexGameState.from_trmph(full_trmph)
    state = apply_display_mask_to_state(state, display_board_size)
    app.logger.info(
        "Loaded masked game state %s: display=%sx%s, user_moves=%s",
        context,
        display_board_size,
        display_board_size,
        len(fc.split_trmph_moves(user_bare_moves)) if user_bare_moves else 0,
    )
    return state

def build_game_response(
    state,
    elo_rating,
    display_board_size=DEFAULT_DISPLAY_BOARD_SIZE,
    trmph_for_inference=None,
    additional_fields=None,
):
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
    
    # Get difficulty parameters and apply temperature scaling
    difficulty_params = get_difficulty_parameters(elo_rating)
    temperature = difficulty_params["temperature"]
    model_id = difficulty_params["model"]
    
    # Model inference
    model = get_model(model_id)
    if trmph_for_inference is None:
        trmph_for_inference = state.to_trmph()
    
    policy_logits, value_signed = model.simple_infer(trmph_for_inference)
    
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    policy_dict = {fc.tensor_to_trmph(i): float(prob) for i, prob in enumerate(policy_probs)}
    win_probability = ValuePredictor.get_win_probability(value_signed, player_enum)
    
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
        "win_probability": win_probability,
        "display_board_size": display_board_size,
        "network_board_size": BOARD_SIZE,
    }
    
    # Add any additional fields
    if additional_fields:
        response.update(additional_fields)
    
    return response

def moves_to_trmph(moves):
    return [fc.rowcol_to_trmph(row, col) for row, col in moves]

def build_move_response(
    state,
    display_board_size=DEFAULT_DISPLAY_BOARD_SIZE,
    move_made=None,
    success=True,
    error=None,
    additional_fields=None,
):
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
        "new_trmph": state_to_user_trmph(state, display_board_size),
        "board": state.board.tolist(),
        "player": winner_to_color(state.current_player_enum),
        "legal_moves": moves_to_trmph(state.get_legal_moves()),
        "winner": winner_to_color(state.winner) if state.winner is not None else None,
        "move_made": move_made,
        "game_over": state.game_over,
        "display_board_size": display_board_size,
        "network_board_size": BOARD_SIZE,
    }
    
    if error:
        response["error"] = error

    if additional_fields:
        response.update(additional_fields)
        
    return response

def _check_game_over_early_return(state, display_board_size):
    """Check if game is over and return early response if so."""
    if state.game_over:
        app.logger.info("Game is over, returning current state")
        result = build_move_response(
            state, display_board_size=display_board_size, move_made=None
        )
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
        confidence_termination_threshold=INTERACTIVE_CONFIDENCE_TERMINATION_THRESHOLD,
        sims=num_simulations,
        c_puct=exploration_constant,
        temperature_start=temperature,
        temperature_end=temperature_end,
        enable_gumbel_root_selection=enable_gumbel,
        gumbel_sim_threshold=gumbel_max_sims
    )
    
    # Log detailed MCTS configuration
    app.logger.info(f"=== DETAILED MCTS CONFIG ===")
    app.logger.info(f"Simulations: {mcts_config.sims}")
    # app.logger.info(f"Batch cap: {mcts_config.batch_cap}")
    # app.logger.info(f"C_PUCT: {mcts_config.c_puct}")
    # app.logger.info(f"Cache size: {mcts_config.cache_size}")
    # app.logger.info(f"Temperature start: {mcts_config.temperature_start}")
    # app.logger.info(f"Temperature end: {mcts_config.temperature_end}")
    # app.logger.info(f"Temperature decay type: {mcts_config.temperature_decay_type}")
    # app.logger.info(f"Add root noise: {mcts_config.add_root_noise}")
    # app.logger.info(f"Dirichlet alpha: {mcts_config.dirichlet_alpha}")
    # app.logger.info(f"Dirichlet eps: {mcts_config.dirichlet_eps}")
    # app.logger.info(f"Enable terminal move detection: {mcts_config.enable_terminal_move_detection}")
    # app.logger.info(f"Terminal detection max depth: {mcts_config.terminal_detection_max_depth}")
    # app.logger.info(f"Terminal move boost: {mcts_config.terminal_move_boost}")
    # app.logger.info(f"Prefer immediate terminal: {mcts_config.prefer_immediate_terminal}")
    # app.logger.info(f"Enable confidence termination: {mcts_config.enable_confidence_termination}")
    # app.logger.info(f"Confidence termination threshold: {mcts_config.confidence_termination_threshold}")
    # app.logger.info(f"Enable depth discounting: {mcts_config.enable_depth_discounting}")
    # app.logger.info(f"Depth discount factor: {mcts_config.depth_discount_factor}")
    # app.logger.info(f"Gumbel temperature enabled: {mcts_config.gumbel_temperature_enabled}")
    # app.logger.info(f"Temperature deterministic cutoff: {mcts_config.temperature_deterministic_cutoff}")
    # app.logger.info(f"Distinct target: {mcts_config.distinct_target}")
    # app.logger.info(f"Adaptive distinct target: {mcts_config.adaptive_distinct_target}")
    # app.logger.info(f"Distinct target min: {mcts_config.distinct_target_min}")
    # app.logger.info(f"Distinct target max: {mcts_config.distinct_target_max}")
    
    return mcts_config

def _execute_mcts_search(state, model_id, mcts_config):
    """Execute MCTS search and return the selected move."""
    # Create game engine
    engine = HexGameEngine()
    app.logger.info("Game engine created")
    
    # Get cached model wrapper for MCTS
    app.logger.info(f"Getting cached ModelWrapper for model_id={model_id}")
    model_wrapper = get_cached_model_wrapper(model_id)
    
    # Run MCTS search
    app.logger.info("Starting MCTS search...")
    try:
        move, stats, tree_data, algorithm_termination_info = run_mcts_move(engine, model_wrapper, state, mcts_config)
        app.logger.info("run_mcts_move completed successfully")
    except Exception as e:
        app.logger.error(f"run_mcts_move failed with exception: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    return move

def _apply_move_and_build_response(state, move, display_board_size):
    """Apply the selected move and build the response."""
    selected_move_trmph = fc.rowcol_to_trmph(*move)
    app.logger.info(f"Selected move TRMPH: {selected_move_trmph}")
    
    # Apply the move
    app.logger.info(f"Applying move: {selected_move_trmph}")
    state = apply_move_to_state_trmph(state, selected_move_trmph)
    app.logger.info(f"Move applied. New state game_over: {state.game_over}")
    
    return build_move_response(
        state,
        display_board_size=display_board_size,
        move_made=selected_move_trmph
    )

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

def _execute_mcts_move_workflow(state, model_id, mcts_params, display_board_size):
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
    return _apply_move_and_build_response(state, move, display_board_size)

def make_mcts_move(trmph, model_id, num_simulations, exploration_constant, 
                   temperature, temperature_end, verbose, enable_gumbel, gumbel_max_sims,
                   display_board_size=DEFAULT_DISPLAY_BOARD_SIZE):
    """Make one computer move using MCTS and return the new state with diagnostics."""
    try:
        app.logger.info(f"=== MCTS MOVE START ===")
        app.logger.info(f"Input: model_id={model_id}, sims={num_simulations}, temp={temperature}->{temperature_end}, verbose={verbose}, gumbel={enable_gumbel}, gumbel_max_sims={gumbel_max_sims}")
        app.logger.info(f"Input TRMPH: '{trmph}'")
        
        # Create game state from TRMPH (validation already done by calling API endpoint)
        state = create_game_state_from_trmph(
            trmph,
            display_board_size=display_board_size,
            context="for MCTS move"
        )
        app.logger.info(f"Game state created: game_over={state.game_over}, current_player={state.current_player_enum}")
        
        # Check if game is over (early return)
        early_result = _check_game_over_early_return(state, display_board_size)
        if early_result:
            return early_result
        
        # Prepare MCTS parameters
        mcts_params = _prepare_mcts_parameters(
            num_simulations, exploration_constant, temperature, temperature_end, 
            enable_gumbel, gumbel_max_sims
        )
        
        # Log the complete configuration that will be used
        app.logger.info(f"=== MCTS CONFIGURATION ===")
        app.logger.info(f"Model: {model_id}")
        app.logger.info(f"Simulations: {mcts_params['num_simulations']}")
        # app.logger.info(f"Exploration constant (c_puct): {mcts_params['exploration_constant']}")
        # app.logger.info(f"Temperature: {mcts_params['temperature']} -> {mcts_params['temperature_end']}")
        app.logger.info(f"Gumbel enabled: {mcts_params['enable_gumbel']}")
        # app.logger.info(f"Gumbel max sims: {mcts_params['gumbel_max_sims']}")
        
        # Execute MCTS workflow
        result = _execute_mcts_move_workflow(
            state, model_id, mcts_params, display_board_size
        )
        
        # Add configuration to result for frontend verification
        result['mcts_config'] = {
            'model': model_id,
            'num_simulations': mcts_params['num_simulations'],
            'exploration_constant': mcts_params['exploration_constant'],
            'temperature': mcts_params['temperature'],
            'temperature_end': mcts_params['temperature_end'],
            'enable_gumbel': mcts_params['enable_gumbel'],
            'gumbel_max_sims': mcts_params['gumbel_max_sims']
        }
        
        # app.logger.debug(f"=== MCTS MOVE COMPLETE ===")
        # app.logger.debug(f"Move made: {result.get('move_made', 'N/A')}")
        # app.logger.debug(f"Game over: {result.get('game_over', 'N/A')}")
        # app.logger.debug(f"Winner: {result.get('winner', 'N/A')}")
        
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
@rate_limit(ENDPOINT_COSTS['api_constants'])
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
        },
        "DISPLAY_BOARD_SIZE_OPTIONS": DISPLAY_BOARD_SIZE_OPTIONS,
        "DEFAULT_DISPLAY_BOARD_SIZE": DEFAULT_DISPLAY_BOARD_SIZE,
        "MIN_DISPLAY_BOARD_SIZE": MIN_DISPLAY_BOARD_SIZE,
        "DEFAULT_PIE_RULE_ENABLED": DEFAULT_PIE_RULE_ENABLED,
        "DIFFICULTY_LEVELS": get_difficulty_levels(),
        "ELO_CONFIG": {
            "MIN_ELO": MIN_ELO,
            "MAX_ELO": MAX_ELO,
            "DEFAULT_ELO": DEFAULT_ELO
        }
    })


@app.route("/api/state", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_state'])
def api_state():
    data = request.get_json()
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=None,  # No required fields
        optional_fields=['trmph', 'elo_rating', 'display_board_size', 'pie_rule_enabled']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", DEFAULT_ELO)  # Default to configured default difficulty
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    
    app.logger.info(
        "api_state called with trmph='%s', elo_rating=%s, display_board_size=%s",
        trmph,
        elo_rating,
        display_board_size,
    )
    
    # Create game state from TRMPH (validation already done by validate_api_input)
    state = create_game_state_from_trmph(
        trmph,
        display_board_size=display_board_size
    )
    trmph_for_inference = state.to_trmph()
    
    # Build response using helper function
    response = build_game_response(
        state,
        elo_rating,
        display_board_size=display_board_size,
        trmph_for_inference=trmph_for_inference,
        additional_fields={
            "trmph": trmph,
            **build_pie_rule_response_fields(
                trmph=trmph,
                state=state,
                pie_rule_enabled=pie_rule_enabled,
            ),
        },
    )

    trmph_stats = _build_trmph_stats(trmph)
    seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
    log_usage_event(
        "state",
        status=200,
        elo_rating=elo_rating,
        display_board_size=display_board_size,
        pie_rule_enabled=pie_rule_enabled,
        **trmph_stats,
        **seq_info
    )
    return jsonify(response)


@app.route("/api/move_heatmap", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_move_heatmap'])
def api_move_heatmap():
    """
    Return value-head win-rate heatmap for every legal next move.

    Scores are always from the *current player to move* perspective.
    """
    data = request.get_json()

    is_valid, error_msg, validated_data = validate_api_input(
        data,
        required_fields=None,
        optional_fields=[
            "trmph",
            "elo_rating",
            "display_board_size",
            "model_id",
            "score_type",
            "selection_mode",
            "top_k",
            "policy_temperature",
        ]
    )
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400

    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", DEFAULT_ELO)
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    model_id = validated_data.get("model_id")
    score_type = validated_data.get("score_type", "policy_value")
    selection_mode = validated_data.get("selection_mode", "all_legal")
    top_k = validated_data.get("top_k", 12)
    policy_temperature = validated_data.get("policy_temperature", 1.0)

    try:
        if score_type != "policy_value":
            return jsonify({"success": False, "error": f"Unsupported score_type: {score_type}"}), 400
        if selection_mode not in {"policy_top_k", "all_legal"}:
            return jsonify({"success": False, "error": f"Invalid selection_mode: {selection_mode}"}), 400
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "top_k must be an integer"}), 400
        if top_k < 1:
            return jsonify({"success": False, "error": "top_k must be >= 1"}), 400
        try:
            policy_temperature = float(policy_temperature)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "policy_temperature must be numeric"}), 400
        if policy_temperature <= 0:
            return jsonify({"success": False, "error": "policy_temperature must be > 0"}), 400

        state = create_game_state_from_trmph(
            trmph,
            display_board_size=display_board_size,
            context="for move heatmap"
        )
        if model_id is None:
            model_id = get_difficulty_parameters(elo_rating)["model"]

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
            "display_board_size": display_board_size,
            "network_board_size": BOARD_SIZE,
        }
        response.update(heatmap.to_dict())

        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "move_heatmap",
            status=200,
            elo_rating=elo_rating,
            model_id=model_id,
            score_type=score_type,
            selection_mode=selection_mode,
            top_k=top_k,
            selected_move_count=response["selected_move_count"],
            legal_move_count=response["legal_move_count"],
            display_board_size=display_board_size,
            **trmph_stats,
            **seq_info
        )
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in api_move_heatmap: {e}")
        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "move_heatmap",
            status=500,
            success=False,
            reason="exception",
            elo_rating=elo_rating,
            model_id=model_id,
            display_board_size=display_board_size,
            **trmph_stats,
            **seq_info
        )
        return jsonify({"success": False, "error": "Failed to compute move heatmap"}), 500

@app.route("/api/apply_move", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_apply_move'])
def api_apply_move():
    """Apply only a human move without making a computer move."""
    data = request.get_json()
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=['move'],  # Move is required
        optional_fields=['trmph', 'elo_rating', 'display_board_size', 'pie_rule_enabled']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    move = validated_data.get("move")
    elo_rating = validated_data.get("elo_rating", 1000)
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    
    app.logger.info(
        "api_apply_move called with trmph='%s', move='%s', elo_rating=%s, display_board_size=%s",
        trmph,
        move,
        elo_rating,
        display_board_size,
    )
    
    # Create game state from TRMPH (validation already done by validate_api_input)
    state = create_game_state_from_trmph(
        trmph,
        display_board_size=display_board_size,
        context="for move"
    )
    
    try:
        state = apply_move_to_state_trmph(state, move)
    except Exception as e:
        # Silently ignore invalid moves (e.g., clicking on already filled hex)
        app.logger.debug(f"Invalid move ignored: {e}")
        # Return current state without error - user will learn not to click filled hexes
        response = build_game_response(state, elo_rating, display_board_size, state.to_trmph(), {
            "new_trmph": trmph,
            "model_move": None,  # No computer move made
            **build_pie_rule_response_fields(
                trmph=trmph,
                state=state,
                pie_rule_enabled=pie_rule_enabled,
            ),
        })
        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "apply_move",
            status=200,
            move=move,
            move_valid=False,
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            moves_requested=1,
            **trmph_stats,
            **seq_info
        )
        return jsonify(response)

    new_trmph = state_to_user_trmph(state, display_board_size)
    
    # Build response using helper function
    response = build_game_response(state, elo_rating, display_board_size, state.to_trmph(), {
        "new_trmph": new_trmph,
        "model_move": None,  # No computer move made
        **build_pie_rule_response_fields(
            trmph=new_trmph,
            state=state,
            pie_rule_enabled=pie_rule_enabled,
        ),
    })
    trmph_stats = _build_trmph_stats(trmph)
    seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
    log_usage_event(
        "apply_move",
        status=200,
        move=move,
        move_valid=True,
        elo_rating=elo_rating,
        display_board_size=display_board_size,
        pie_rule_enabled=pie_rule_enabled,
        moves_requested=1,
        new_trmph_len=len(fc.strip_trmph_preamble((new_trmph or "").strip())),
        new_trmph_moves=_safe_count_trmph_moves(new_trmph),
        **trmph_stats,
        **seq_info
    )
    return jsonify(response)

@app.route("/api/policy_move", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_policy_move'])
def api_policy_move():
    """Make a computer move using policy sampling."""
    data = request.get_json()
    app.logger.info(f"=== POLICY API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=None,  # No required fields
        optional_fields=['trmph', 'elo_rating', 'display_board_size', 'pie_rule_enabled']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", 1000)
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    
    app.logger.info(
        "Parsed parameters: trmph='%s', elo_rating=%s, display_board_size=%s",
        trmph,
        elo_rating,
        display_board_size,
    )
    
    try:
        # Create game state from TRMPH (validation already done by validate_api_input)
        state = create_game_state_from_trmph(
            trmph,
            display_board_size=display_board_size,
            context="for policy move"
        )
        
        # Get difficulty parameters
        difficulty_params = get_difficulty_parameters(elo_rating)
        temperature = difficulty_params["temperature"]
        model_id = difficulty_params["model"]
        pie_decision = evaluate_pie_rule_swap_decision(
            trmph=trmph,
            display_board_size=display_board_size,
            model_id=model_id,
            pie_rule_enabled=pie_rule_enabled,
            state=state,
        )
        
        # Log the policy configuration being used
        app.logger.info(f"=== POLICY CONFIGURATION ===")
        app.logger.info(f"ELO Rating: {elo_rating}")
        app.logger.info(f"Algorithm: {difficulty_params['algorithm']}")
        app.logger.info(f"Model: {model_id}")
        app.logger.info(f"Temperature: {temperature}")
        app.logger.info(f"Simulations: {difficulty_params['num_simulations']}")
        app.logger.info(f"Exploration constant: {difficulty_params['exploration_constant']}")
        app.logger.info(f"Enable Gumbel: {difficulty_params['enable_gumbel']}")

        if pie_decision and pie_decision["should_swap"]:
            app.logger.info(
                "Pie-rule swap selected for first move %s (opening_p=%.4f, swap_p=%.4f, roll=%.4f)",
                pie_decision["first_move"],
                pie_decision["opening_win_probability"],
                pie_decision["swap_probability"],
                pie_decision["swap_roll"],
            )
            result = build_move_response(
                state,
                display_board_size=display_board_size,
                move_made=None,
                additional_fields={
                    "pie_rule_swap_computer_colors": True,
                    **build_pie_rule_response_fields(
                        trmph=trmph,
                        state=state,
                        pie_rule_enabled=pie_rule_enabled,
                        pie_decision=pie_decision,
                        pie_rule_action="swapped",
                    ),
                },
            )
            result["mcts_config"] = {
                "model": model_id,
                "num_simulations": difficulty_params["num_simulations"],
                "exploration_constant": difficulty_params["exploration_constant"],
                "temperature": temperature,
                "temperature_end": temperature,
                "enable_gumbel": difficulty_params["enable_gumbel"],
                "gumbel_max_sims": difficulty_params.get("gumbel_max_sims", 0),
                "algorithm": difficulty_params["algorithm"],
            }

            trmph_stats = _build_trmph_stats(trmph)
            seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
            log_usage_event(
                "policy_move",
                status=200,
                success=True,
                reason="pie_rule_swap",
                elo_rating=elo_rating,
                algorithm=difficulty_params["algorithm"],
                model_id=model_id,
                temperature=temperature,
                num_simulations=difficulty_params["num_simulations"],
                exploration_constant=difficulty_params["exploration_constant"],
                enable_gumbel=difficulty_params["enable_gumbel"],
                gumbel_max_sims=difficulty_params.get("gumbel_max_sims", 0),
                move_made=None,
                pie_rule_enabled=pie_rule_enabled,
                pie_rule_action="swapped",
                pie_rule_first_move=pie_decision["first_move"],
                pie_rule_opening_win_probability=pie_decision["opening_win_probability"],
                pie_rule_swap_probability=pie_decision["swap_probability"],
                pie_rule_cache_hit=pie_decision["cache_hit"],
                display_board_size=display_board_size,
                moves_requested=0,
                **trmph_stats,
                **seq_info
            )
            return jsonify(result)
        
        # Get model and make policy move
        model = get_model(model_id)
        move = select_policy_move(state, model, temperature)
        
        if move is None:
            trmph_stats = _build_trmph_stats(trmph)
            seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
            log_usage_event(
                "policy_move",
                status=400,
                success=False,
                reason="no_valid_moves",
                elo_rating=elo_rating,
                algorithm=difficulty_params["algorithm"],
                model_id=model_id,
                temperature=temperature,
                num_simulations=difficulty_params["num_simulations"],
                display_board_size=display_board_size,
                pie_rule_enabled=pie_rule_enabled,
                **trmph_stats,
                **seq_info
            )
            return jsonify({"success": False, "error": "No valid moves available"}), 400
        
        # Apply the move
        move_trmph = fc.rowcol_to_trmph(move[0], move[1])
        new_state = apply_move_to_state_trmph(state, move_trmph)
        
        result = build_move_response(
            new_state,
            display_board_size=display_board_size,
            move_made=move_trmph,
            additional_fields=build_pie_rule_response_fields(
                trmph=trmph,
                state=state,
                pie_rule_enabled=pie_rule_enabled,
                pie_decision=pie_decision,
                pie_rule_action="declined" if pie_decision else "none",
            ),
        )
        
        # Add configuration to result for frontend verification
        result['mcts_config'] = {
            'model': model_id,
            'num_simulations': difficulty_params['num_simulations'],
            'exploration_constant': difficulty_params['exploration_constant'],
            'temperature': temperature,
            'temperature_end': temperature,  # Policy moves use same temperature throughout
            'enable_gumbel': difficulty_params['enable_gumbel'],
            'gumbel_max_sims': difficulty_params.get('gumbel_max_sims', 0),
            'algorithm': difficulty_params['algorithm']
        }
        
        app.logger.info(f"=== POLICY API RESPONSE ===")
        app.logger.info(f"Selected move: {move_trmph}")

        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "policy_move",
            status=200,
            success=True,
            elo_rating=elo_rating,
            algorithm=difficulty_params["algorithm"],
            model_id=model_id,
            temperature=temperature,
            num_simulations=difficulty_params["num_simulations"],
            exploration_constant=difficulty_params["exploration_constant"],
            enable_gumbel=difficulty_params["enable_gumbel"],
            gumbel_max_sims=difficulty_params.get("gumbel_max_sims", 0),
            move_made=move_trmph,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            pie_rule_action="declined" if pie_decision else "none",
            moves_requested=1,
            **trmph_stats,
            **seq_info
        )
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Policy move error: {e}")
        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "policy_move",
            status=500,
            success=False,
            reason="exception",
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            **trmph_stats,
            **seq_info
        )
        return jsonify({"success": False, "error": "Policy move generation failed. Please try again."}), 500

@app.route("/api/mcts_move", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_mcts_move'])
def api_mcts_move():
    """Make a computer move using MCTS with diagnostic output."""
    data = request.get_json()
    app.logger.info(f"=== MCTS API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=None,  # No required fields
        optional_fields=['trmph', 'elo_rating', 'display_board_size', 'pie_rule_enabled']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", 1000)
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    
    app.logger.info(
        "Parsed parameters: trmph='%s', elo_rating=%s, display_board_size=%s",
        trmph,
        elo_rating,
        display_board_size,
    )
    
    # Get difficulty parameters
    difficulty_params = get_difficulty_parameters(elo_rating)
    pie_decision = None

    if difficulty_params["algorithm"] != "policy":
        state = create_game_state_from_trmph(
            trmph,
            display_board_size=display_board_size,
            context="for mcts move",
        )
        pie_decision = evaluate_pie_rule_swap_decision(
            trmph=trmph,
            display_board_size=display_board_size,
            model_id=difficulty_params["model"],
            pie_rule_enabled=pie_rule_enabled,
            state=state,
        )
        if pie_decision and pie_decision["should_swap"]:
            app.logger.info(
                "Pie-rule swap selected (MCTS) for first move %s (opening_p=%.4f, swap_p=%.4f, roll=%.4f)",
                pie_decision["first_move"],
                pie_decision["opening_win_probability"],
                pie_decision["swap_probability"],
                pie_decision["swap_roll"],
            )
            result = build_move_response(
                state,
                display_board_size=display_board_size,
                move_made=None,
                additional_fields={
                    "pie_rule_swap_computer_colors": True,
                    **build_pie_rule_response_fields(
                        trmph=trmph,
                        state=state,
                        pie_rule_enabled=pie_rule_enabled,
                        pie_decision=pie_decision,
                        pie_rule_action="swapped",
                    ),
                },
            )
            result["mcts_config"] = {
                "model": difficulty_params["model"],
                "num_simulations": difficulty_params["num_simulations"],
                "exploration_constant": difficulty_params["exploration_constant"],
                "temperature": difficulty_params["temperature"],
                "temperature_end": difficulty_params["temperature_end"],
                "enable_gumbel": difficulty_params["enable_gumbel"],
                "gumbel_max_sims": difficulty_params["gumbel_max_sims"],
                "algorithm": difficulty_params["algorithm"],
            }
            trmph_stats = _build_trmph_stats(trmph)
            seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
            log_usage_event(
                "mcts_move",
                status=200,
                success=True,
                reason="pie_rule_swap",
                elo_rating=elo_rating,
                algorithm=difficulty_params["algorithm"],
                model_id=difficulty_params["model"],
                temperature=difficulty_params["temperature"],
                temperature_end=difficulty_params["temperature_end"],
                num_simulations=difficulty_params["num_simulations"],
                exploration_constant=difficulty_params["exploration_constant"],
                enable_gumbel=difficulty_params["enable_gumbel"],
                gumbel_max_sims=difficulty_params["gumbel_max_sims"],
                move_made=None,
                pie_rule_enabled=pie_rule_enabled,
                pie_rule_action="swapped",
                pie_rule_first_move=pie_decision["first_move"],
                pie_rule_opening_win_probability=pie_decision["opening_win_probability"],
                pie_rule_swap_probability=pie_decision["swap_probability"],
                pie_rule_cache_hit=pie_decision["cache_hit"],
                display_board_size=display_board_size,
                moves_requested=0,
                **trmph_stats,
                **seq_info
            )
            return jsonify(result)
    
    if difficulty_params["algorithm"] == "policy":
        # Use policy move for lower difficulties
        return api_policy_move()
    
    # Use MCTS for higher difficulties
    result = make_mcts_move(
        trmph,
        difficulty_params["model"],
        difficulty_params["num_simulations"],
        difficulty_params["exploration_constant"],
        difficulty_params["temperature"],
        difficulty_params["temperature_end"],
        0,  # verbose
        difficulty_params["enable_gumbel"],
        difficulty_params["gumbel_max_sims"],
        display_board_size=display_board_size,
    )

    if result.get("success"):
        result.update(
            build_pie_rule_response_fields(
                trmph=trmph,
                state=state,
                pie_rule_enabled=pie_rule_enabled,
                pie_decision=pie_decision,
                pie_rule_action="declined" if pie_decision else "none",
            )
        )
    
    app.logger.info(f"=== MCTS API RESPONSE ===")
    app.logger.info(f"Result success: {result.get('success', 'MISSING')}")
    if result.get('success'):
        app.logger.info(f"Move made: {result.get('move_made', 'MISSING')}")
        app.logger.info(f"Game over: {result.get('game_over', 'MISSING')}")
        app.logger.info(f"Winner: {result.get('winner', 'MISSING')}")
    else:
        app.logger.error(f"Result error: {result.get('error', 'MISSING')}")

    trmph_stats = _build_trmph_stats(trmph)
    seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
    log_usage_event(
        "mcts_move",
        status=200,
        success=bool(result.get("success")),
        elo_rating=elo_rating,
        algorithm=difficulty_params["algorithm"],
        model_id=difficulty_params["model"],
        temperature=difficulty_params["temperature"],
        temperature_end=difficulty_params["temperature_end"],
        num_simulations=difficulty_params["num_simulations"],
        exploration_constant=difficulty_params["exploration_constant"],
        enable_gumbel=difficulty_params["enable_gumbel"],
        gumbel_max_sims=difficulty_params["gumbel_max_sims"],
        move_made=result.get("move_made"),
        display_board_size=display_board_size,
        pie_rule_enabled=pie_rule_enabled,
        pie_rule_action=result.get("pie_rule_action"),
        moves_requested=1,
        **trmph_stats,
        **seq_info
    )
    
    return jsonify(result)

@app.route("/api/apply_trmph_sequence", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_apply_trmph_sequence'])
def api_apply_trmph_sequence():
    """Apply a sequence of TRMPH moves to the current game state."""
    data = request.get_json()
    
    # Validate input using centralized validation
    is_valid, error_msg, validated_data = validate_api_input(
        data, 
        required_fields=['trmph_sequence'],  # Sequence is required
        optional_fields=['trmph', 'elo_rating', 'display_board_size', 'pie_rule_enabled']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    trmph_sequence = validated_data.get("trmph_sequence", "")
    elo_rating = validated_data.get("elo_rating", 1000)
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    
    app.logger.info(
        "api_apply_trmph_sequence called with trmph='%s', sequence='%s', elo_rating=%s, display_board_size=%s",
        trmph,
        trmph_sequence,
        elo_rating,
        display_board_size,
    )
    
    try:
        # Create game state from TRMPH (validation already done by validate_api_input)
        state = create_game_state_from_trmph(
            trmph,
            display_board_size=display_board_size,
            context="for TRMPH sequence"
        )
        
        # Apply the TRMPH sequence
        moves_applied = 0
        if trmph_sequence and trmph_sequence.strip():
            # Use the proper TRMPH parsing utility instead of naive string slicing
            try:
                app.logger.info(f"Attempting to split TRMPH sequence: '{trmph_sequence}'")
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
                trmph_stats = _build_trmph_stats(trmph)
                seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
                log_usage_event(
                    "apply_trmph_sequence",
                    status=400,
                    success=False,
                    reason="invalid_sequence",
                    elo_rating=elo_rating,
                    display_board_size=display_board_size,
                    **trmph_stats,
                    **seq_info
                )
                return jsonify({"error": f"Invalid TRMPH sequence format [DEBUG-CHECK]: {str(e)}"}), 400
        
        new_trmph = state_to_user_trmph(state, display_board_size)
        
        # Build response using helper function
        response = build_game_response(state, elo_rating, display_board_size, state.to_trmph(), {
            "new_trmph": new_trmph,
            "moves_applied": moves_applied,
            **build_pie_rule_response_fields(
                trmph=new_trmph,
                state=state,
                pie_rule_enabled=pie_rule_enabled,
            ),
        })
        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "apply_trmph_sequence",
            status=200,
            success=True,
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            moves_requested=_safe_count_trmph_moves(trmph_sequence),
            moves_applied=moves_applied,
            **trmph_stats,
            **seq_info
        )
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"TRMPH sequence application error: {e}")
        trmph_stats = _build_trmph_stats(trmph)
        seq_info = _update_sequence_info(getattr(g, "analytics_client_id", None), trmph) if ANALYTICS_ENABLED else {}
        log_usage_event(
            "apply_trmph_sequence",
            status=500,
            success=False,
            reason="exception",
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            **trmph_stats,
            **seq_info
        )
        return jsonify({"error": "Failed to apply TRMPH sequence. Please check the sequence and try again."}), 500


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(os.path.dirname(__file__)), "favicon3_cropped.png")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static_public"), path)

@app.route("/shared/<path:path>")
def serve_shared(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static_shared"), path)

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
