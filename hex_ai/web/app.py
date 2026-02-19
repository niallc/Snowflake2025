from flask import Flask, request, jsonify, send_from_directory, g
import os
from flask_cors import CORS
import json
import logging
import time
import ipaddress
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import uuid
import random
import threading
import numpy as np

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state_trmph
from hex_ai.value_utils import (
    select_policy_move,
)
from hex_ai.enums import Player, Piece
from hex_ai.config import BOARD_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.inference.model_config import (
    is_valid_model_id,
    get_normalized_path,
    get_model_path_with_fallback,
)
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
from hex_ai.web.small_board_opening_calibration import (
    remap_small_board_opening_scores,
    should_apply_small_board_opening_remap,
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
    validate_boolean_flag as core_validate_boolean_flag,
    validate_elo_rating as core_validate_elo_rating,
    validate_trmph_input as core_validate_trmph_input,
)

CORS_ALLOW_ALL = os.getenv("SF25_CORS_ALLOW_ALL", "0").lower() not in ("0", "false", "no")
CORS_ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("SF25_CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)

app = Flask(__name__, static_folder="static_public")
if CORS_ALLOW_ALL:
    CORS(app)
elif CORS_ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": list(CORS_ALLOWED_ORIGINS)}})

# =============================================================================
# ANALYTICS CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ANALYTICS_LOG_PATH = BASE_DIR / "logs" / "web_usage.jsonl"

ANALYTICS_ENABLED = os.getenv("SF25_ANALYTICS_ENABLED", "1").lower() not in ("0", "false", "no")
ANALYTICS_LOG_PATH = os.getenv("SF25_ANALYTICS_LOG_PATH", str(DEFAULT_ANALYTICS_LOG_PATH))
ANALYTICS_SALT = os.getenv("SF25_ANALYTICS_SALT", "")
ANALYTICS_HASH_IP = os.getenv("SF25_ANALYTICS_HASH_IP", "1").lower() not in ("0", "false", "no")
ANALYTICS_INCLUDE_IP = os.getenv("SF25_ANALYTICS_INCLUDE_IP", "0").lower() not in ("0", "false", "no")
ANALYTICS_REQUIRE_CONSENT = os.getenv("SF25_ANALYTICS_REQUIRE_CONSENT", "1").lower() not in ("0", "false", "no")
TRUST_PROXY_HEADERS = os.getenv("SF25_TRUST_PROXY_HEADERS", "0").lower() not in ("0", "false", "no")
ANALYTICS_CONSENT_COOKIE_NAME = os.getenv("SF25_ANALYTICS_CONSENT_COOKIE_NAME", "sf25_cookie_consent")
ANALYTICS_CONSENT_ACCEPT_VALUE = os.getenv("SF25_ANALYTICS_CONSENT_ACCEPT_VALUE", "accepted")
ANALYTICS_COOKIE_NAME = os.getenv("SF25_ANALYTICS_COOKIE_NAME", "sf25_cid")
ANALYTICS_COOKIE_DAYS = int(os.getenv("SF25_ANALYTICS_COOKIE_DAYS", "365"))
ANALYTICS_SEQUENCE_TTL_SECONDS = int(os.getenv("SF25_ANALYTICS_SEQUENCE_TTL_SECONDS", "3600"))
ANALYTICS_SEQUENCE_MAX_CLIENTS = int(os.getenv("SF25_ANALYTICS_SEQUENCE_MAX_CLIENTS", "10000"))
ANALYTICS_INCLUDE_COUNTRY = os.getenv("SF25_ANALYTICS_INCLUDE_COUNTRY", "1").lower() not in ("0", "false", "no")
ANALYTICS_COUNTRY_HEADERS = tuple(
    header.strip()
    for header in os.getenv(
        "SF25_ANALYTICS_COUNTRY_HEADERS",
        "CF-IPCountry,X-Country-Code,X-Appengine-Country",
    ).split(",")
    if header.strip()
)

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
        if ANALYTICS_INCLUDE_IP and ANALYTICS_HASH_IP and not ANALYTICS_SALT:
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
DEFAULT_ELO = 600

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
    remapped_scores, remap_meta = remap_small_board_opening_scores(
        heatmap.scores,
        display_board_size=display_board_size,
        network_board_size=BOARD_SIZE,
    )
    return remapped_scores, remap_meta


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

    scores, remap_meta = _compute_pie_rule_opening_scores(model_id, display_board_size)
    new_entry = {
        "scores": scores,
        "model_id": model_identity["model_id"],
        "model_path": model_identity["model_path"],
        "model_mtime_ns": model_identity["model_mtime_ns"],
        "epoch": model_identity["epoch"],
        "mini": model_identity["mini"],
        "opening_score_remap_applied": bool(remap_meta.get("applied", False)),
        "opening_score_remap_reason": remap_meta.get("reason"),
        "opening_score_remap_anchor_moves": remap_meta.get("anchor_moves"),
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


def _maybe_remap_first_move_heatmap_scores(
    *,
    trmph: str,
    display_board_size: int,
    model_id: str,
    scores: dict[str, float],
) -> tuple[dict[str, float], bool]:
    """
    Apply opening-score remap for opening-position heatmaps on small boards.

    If anchors are unavailable in the provided subset (e.g. policy_top_k), reuse
    the pie-rule opening cache and project those calibrated scores onto the subset.
    """
    if _safe_count_trmph_moves(trmph) != 0:
        return scores, False
    if not should_apply_small_board_opening_remap(
        display_board_size=display_board_size,
        network_board_size=BOARD_SIZE,
    ):
        return scores, False

    remapped_scores, remap_meta = remap_small_board_opening_scores(
        scores,
        display_board_size=display_board_size,
        network_board_size=BOARD_SIZE,
    )
    if remap_meta.get("applied"):
        return remapped_scores, True

    if remap_meta.get("reason") != "missing_anchor_moves":
        return scores, False

    opening_scores, _cache_meta = _get_pie_rule_opening_scores(model_id, display_board_size)
    projected_scores = {}
    changed = False
    for move, score in scores.items():
        calibrated = opening_scores.get(move)
        if calibrated is None:
            projected_scores[move] = float(score)
            continue
        calibrated = float(calibrated)
        projected_scores[move] = calibrated
        if abs(calibrated - float(score)) > 1e-12:
            changed = True
    return projected_scores, changed


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
        "opening_score_remap_applied": bool(
            cache_meta.get("opening_score_remap_applied", False)
        ),
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
            "pie_rule_opening_score_remap_applied": pie_decision[
                "opening_score_remap_applied"
            ],
        }
    )
    return response


def _build_mcts_config_response_fields(difficulty_params, temperature_end=None):
    """Build consistent MCTS config payload fields for move responses."""
    resolved_temperature_end = (
        difficulty_params["temperature"] if temperature_end is None else temperature_end
    )
    return {
        "model": difficulty_params["model"],
        "num_simulations": difficulty_params["num_simulations"],
        "exploration_constant": difficulty_params["exploration_constant"],
        "temperature": difficulty_params["temperature"],
        "temperature_end": resolved_temperature_end,
        "enable_gumbel": difficulty_params["enable_gumbel"],
        "gumbel_max_sims": difficulty_params.get("gumbel_max_sims", 0),
        "algorithm": difficulty_params["algorithm"],
    }


def _build_pie_rule_swap_move_response(
    state,
    trmph,
    display_board_size,
    pie_rule_enabled,
    pie_decision,
    model_id,
    mcts_config,
    inline_heatmap_options,
):
    """Build a consistent response payload for pie-rule swap outcomes."""
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
    result["mcts_config"] = mcts_config
    maybe_attach_inline_move_heatmap(
        result=result,
        state=state,
        model_id=model_id,
        heatmap_options=inline_heatmap_options,
        model_getter=get_model,
        logger=app.logger,
    )
    return result


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
_ALLOWED_STATE_REASONS = {
    "initial_load",
    "manual_refresh",
    "reset",
    "undo",
    "redo",
    "pie_rule_toggle",
    "other",
}

def _is_request_secure():
    if request.is_secure:
        return True
    if TRUST_PROXY_HEADERS:
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        if forwarded_proto:
            return forwarded_proto.split(",")[0].strip().lower() == "https"
    return False

def _get_client_ip():
    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            # With nginx `proxy_add_x_forwarded_for`, the right-most item is
            # the direct client IP observed by our trusted proxy.
            parts = [part.strip() for part in forwarded_for.split(",") if part.strip()]
            if parts:
                return parts[-1]
    return (request.remote_addr or "unknown").strip()


def _extract_country_code():
    """Return two-letter country code from trusted proxy headers when available."""
    if (
        not ANALYTICS_INCLUDE_COUNTRY
        or not ANALYTICS_COUNTRY_HEADERS
        or not TRUST_PROXY_HEADERS
    ):
        return None
    for header in ANALYTICS_COUNTRY_HEADERS:
        value = request.headers.get(header)
        if not value:
            continue
        country = value.strip().upper()
        if country in {"XX", "A1", "A2", "T1"}:
            return None
        if re.fullmatch(r"[A-Z]{2}", country):
            return country
    return None

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


def _normalize_state_reason(value):
    if value is None:
        return "manual_refresh"
    normalized = str(value).strip().lower()
    if normalized in _ALLOWED_STATE_REASONS:
        return normalized
    return "other"


def _build_heatmap_analytics_fields(heatmap_options):
    if not heatmap_options:
        return {}
    return {
        "heatmap_enabled": bool(heatmap_options.get("enabled", False)),
        "heatmap_selection_mode": heatmap_options.get("selection_mode"),
        "heatmap_top_k": heatmap_options.get("top_k"),
        "heatmap_policy_temperature": heatmap_options.get("policy_temperature"),
    }

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

def _build_sequence_info_for_trmph(trmph):
    if not ANALYTICS_ENABLED:
        return {}
    return _update_sequence_info(getattr(g, "analytics_client_id", None), trmph)

def _log_usage_event_with_trmph_context(event, trmph, **fields):
    """Log analytics event with shared TRMPH and request-sequence context."""
    log_usage_event(
        event,
        **fields,
        **_build_trmph_stats(trmph),
        **_build_sequence_info_for_trmph(trmph),
    )

def _prune_none_values(payload):
    return {k: v for k, v in payload.items() if v is not None}

def log_usage_event(event, **fields):
    if not ANALYTICS_ENABLED:
        return
    if not getattr(g, "analytics_enabled_for_request", False):
        return
    try:
        status = fields.pop("status", None)
        duration_ms = fields.pop("duration_ms", None)
        if duration_ms is None and hasattr(g, "analytics_start"):
            duration_ms = int((time.time() - g.analytics_start) * 1000)

        client_ip = _get_client_ip() if ANALYTICS_INCLUDE_IP else None
        ip_value = None
        if ANALYTICS_INCLUDE_IP:
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
            "ip_hash": ip_value if (ANALYTICS_INCLUDE_IP and ANALYTICS_HASH_IP) else None,
            "ip": ip_value if (ANALYTICS_INCLUDE_IP and not ANALYTICS_HASH_IP) else None,
            "country": _extract_country_code(),
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
    if ANALYTICS_REQUIRE_CONSENT:
        consent_value = request.cookies.get(ANALYTICS_CONSENT_COOKIE_NAME)
        if consent_value != ANALYTICS_CONSENT_ACCEPT_VALUE:
            g.analytics_enabled_for_request = False
            return

    g.analytics_enabled_for_request = True
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
    if not getattr(g, "analytics_enabled_for_request", False):
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
    """Validate TRMPH input format for security."""
    return core_validate_trmph_input(trmph_string, board_size=BOARD_SIZE)


def _validate_move_within_display_board(move_text: str, display_board_size: int) -> None:
    row, col = fc.trmph_move_to_rowcol(move_text, board_size=BOARD_SIZE)
    if row >= display_board_size or col >= display_board_size:
        raise ValueError(
            f"Move '{move_text}' is outside top-left {display_board_size}x{display_board_size} display board"
        )

def validate_api_input(data, required_fields=None, optional_fields=None):
    """Centralized validation for API endpoints."""
    return core_validate_api_input(
        data,
        required_fields=required_fields,
        optional_fields=optional_fields,
        logger=app.logger,
        reject_unexpected=True,
        trmph_validator=validate_trmph_input,
        default_display_board_size=DEFAULT_DISPLAY_BOARD_SIZE,
        display_board_size_validator=validate_display_board_size,
        normalize_trmph_for_display_fn=normalize_user_trmph_for_display,
        move_in_display_validator=_validate_move_within_display_board,
        elo_validator=validate_elo_rating,
        boolean_fields={"pie_rule_enabled"},
    )

def validate_elo_rating(value):
    """Validate and convert ELO rating to integer."""
    return core_validate_elo_rating(value, min_elo=MIN_ELO, max_elo=MAX_ELO)


def validate_boolean_flag(value, field_name):
    """Validate and normalize a boolean feature flag from JSON input."""
    return core_validate_boolean_flag(value, field_name)

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
            client_ip = _get_client_ip()
            
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

def _resolve_registered_model_path(model_id: str) -> str:
    """Resolve a registered model identifier to an absolute model path."""
    if not is_valid_model_id(model_id):
        raise ValueError(f"Unknown model_id: {model_id}")
    return get_model_path_with_fallback(model_id)


def get_model(model_id="best"):
    """Get or create a model instance for the given model_id using centralized cache with fallback support."""
    app.logger.debug(f"get_model called with model_id: {model_id}")

    model_path = _resolve_registered_model_path(model_id)
    app.logger.debug(f"Found model {model_id} -> {model_path}")
    return MODEL_CACHE.get_simple_model(model_path)

def get_cached_model_wrapper(model_id: str):
    """Get or create a cached ModelWrapper instance for the given model_id using centralized cache with fallback support."""
    app.logger.debug(f"get_cached_model_wrapper called with model_id: {model_id}")

    model_path = _resolve_registered_model_path(model_id)
    app.logger.debug(f"Getting ModelWrapper for path: {model_path}")
    return MODEL_CACHE.get_wrapper_model(model_path)

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
    state = core_create_game_state_from_trmph_input(
        trmph,
        context=context,
        display_board_size=display_board_size,
        normalize_user_trmph_fn=normalize_user_trmph_for_display,
        compose_full_trmph_fn=compose_full_trmph_from_user_trmph,
        apply_display_mask_fn=apply_display_mask_to_state,
    )
    user_bare_moves = normalize_user_trmph_for_display(trmph, display_board_size, field_name="trmph")
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
    # Get difficulty parameters and apply temperature scaling
    difficulty_params = get_difficulty_parameters(elo_rating)
    temperature = difficulty_params["temperature"]
    model_id = difficulty_params["model"]

    model = get_model(model_id)
    response = build_game_state_response(
        state,
        model=model,
        temperature=temperature,
        trmph_for_inference=trmph_for_inference,
        additional_fields={
            "display_board_size": display_board_size,
            "network_board_size": BOARD_SIZE,
            **(additional_fields or {}),
        },
    )
    return response

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
    return build_engine_move_response(
        state,
        new_trmph=state_to_user_trmph(state, display_board_size),
        move_made=move_made,
        success=success,
        error=error,
        additional_fields={
            "display_board_size": display_board_size,
            "network_board_size": BOARD_SIZE,
            **(additional_fields or {}),
        },
    )


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
    mcts_config, _ = create_interactive_mcts_config(
        num_simulations=num_simulations,
        exploration_constant=exploration_constant,
        temperature=temperature,
        temperature_end=temperature_end,
        enable_gumbel=enable_gumbel,
        gumbel_max_sims=gumbel_max_sims,
        confidence_termination_threshold=INTERACTIVE_CONFIDENCE_TERMINATION_THRESHOLD,
        logger=app.logger,
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
    # Get cached model wrapper for MCTS
    app.logger.info(f"Getting cached ModelWrapper for model_id={model_id}")
    model_wrapper = get_cached_model_wrapper(model_id)

    move, _, _, _ = run_interactive_mcts_search(
        state=state,
        model_wrapper=model_wrapper,
        mcts_config=mcts_config,
        verbose=0,
        logger=app.logger,
    )
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
        return build_engine_error_payload(
            model_error,
            reason="model_load_failed",
        )
    
    # Create MCTS configuration
    mcts_config = _create_mcts_configuration(**mcts_params)
    
    # Execute MCTS search
    move = _execute_mcts_search(state, model_id, mcts_config)
    
    # Apply move and build response
    return _apply_move_and_build_response(state, move, display_board_size)


def _is_loopback_host(host: str) -> bool:
    """Return True when host is loopback-only."""
    normalized = (host or "").strip().lower()
    if normalized in {"localhost", "[::1]"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


INTERACTIVE_MOVE_OPTIONAL_FIELDS = [
    "trmph",
    "elo_rating",
    "display_board_size",
    "pie_rule_enabled",
]


def _validate_interactive_move_request(data):
    """
    Validate shared interactive move-request fields and parse inline heatmap options.

    Returns:
        tuple:
            - (validated_data, inline_heatmap_options) on success, None on failure
            - Flask error response tuple on failure, None on success
    """
    validated_data, inline_heatmap_options, error_msg = validate_request_with_inline_heatmap(
        data,
        required_fields=None,
        optional_fields=INTERACTIVE_MOVE_OPTIONAL_FIELDS,
        validate_api_input_fn=validate_api_input,
    )
    if error_msg:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return None, (
            jsonify(
                build_engine_error_payload(
                    error_msg,
                    reason="validation_error",
                )
            ),
            400,
        )

    return (validated_data, inline_heatmap_options), None

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
        return build_engine_error_payload(
            "MCTS move generation failed. Please try again.",
            reason="engine_failure",
        )

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
        optional_fields=['trmph', 'elo_rating', 'display_board_size', 'pie_rule_enabled', 'state_reason']
    )
    
    if not is_valid:
        app.logger.warning(f"Invalid input rejected: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    trmph = validated_data.get("trmph", "")
    elo_rating = validated_data.get("elo_rating", DEFAULT_ELO)  # Default to configured default difficulty
    display_board_size = validated_data.get("display_board_size", DEFAULT_DISPLAY_BOARD_SIZE)
    pie_rule_enabled = validated_data.get("pie_rule_enabled", DEFAULT_PIE_RULE_ENABLED)
    state_reason = _normalize_state_reason(validated_data.get("state_reason"))
    
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

    _log_usage_event_with_trmph_context(
        "state",
        trmph,
        status=200,
        elo_rating=elo_rating,
        display_board_size=display_board_size,
        pie_rule_enabled=pie_rule_enabled,
        state_reason=state_reason,
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

    try:
        parsed_heatmap_params = parse_move_heatmap_request_params(
            validated_data,
            default_model_id=None,
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    score_type = parsed_heatmap_params["score_type"]
    selection_mode = parsed_heatmap_params["selection_mode"]
    top_k = parsed_heatmap_params["top_k"]
    policy_temperature = parsed_heatmap_params["policy_temperature"]
    model_id = parsed_heatmap_params["model_id"]

    try:
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
        heatmap_payload = heatmap.to_dict()
        remap_applied = False
        remapped_scores, remap_applied = _maybe_remap_first_move_heatmap_scores(
            trmph=trmph,
            display_board_size=display_board_size,
            model_id=model_id,
            scores=dict(heatmap_payload["scores"]),
        )
        if remap_applied:
            heatmap_payload["scores"] = remapped_scores
            if remapped_scores:
                heatmap_payload["min_score"] = min(remapped_scores.values())
                heatmap_payload["max_score"] = max(remapped_scores.values())
            else:
                heatmap_payload["min_score"] = None
                heatmap_payload["max_score"] = None
        heatmap_payload["opening_score_remap_applied"] = bool(remap_applied)

        response = {
            "success": True,
            "trmph": trmph,
            "model_id": model_id,
            "score_type": score_type,
            "display_board_size": display_board_size,
            "network_board_size": BOARD_SIZE,
        }
        response.update(heatmap_payload)

        _log_usage_event_with_trmph_context(
            "move_heatmap",
            trmph,
            status=200,
            elo_rating=elo_rating,
            model_id=model_id,
            score_type=score_type,
            selection_mode=selection_mode,
            top_k=top_k,
            selected_move_count=response["selected_move_count"],
            legal_move_count=response["legal_move_count"],
            display_board_size=display_board_size,
            heatmap_enabled=True,
        )
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in api_move_heatmap: {e}")
        _log_usage_event_with_trmph_context(
            "move_heatmap",
            trmph,
            status=500,
            success=False,
            reason="exception",
            elo_rating=elo_rating,
            model_id=model_id,
            display_board_size=display_board_size,
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
        sanitized_error = sanitize_exception_message(e)
        app.logger.warning("Invalid move rejected: %s", sanitized_error)
        _log_usage_event_with_trmph_context(
            "apply_move",
            trmph,
            status=400,
            move=move,
            move_valid=False,
            reason="invalid_move",
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            moves_requested=1,
        )
        return jsonify({"error": f"Invalid move: {sanitized_error}"}), 400

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
    _log_usage_event_with_trmph_context(
        "apply_move",
        trmph,
        status=200,
        move=move,
        move_valid=True,
        elo_rating=elo_rating,
        display_board_size=display_board_size,
        pie_rule_enabled=pie_rule_enabled,
        moves_requested=1,
        new_trmph_len=len(fc.strip_trmph_preamble((new_trmph or "").strip())),
        new_trmph_moves=_safe_count_trmph_moves(new_trmph),
    )
    return jsonify(response)

def _execute_policy_move_from_validated_data(validated_data, inline_heatmap_options):
    """Shared policy-move workflow used by policy endpoint and MCTS policy fallback."""
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
            result = _build_pie_rule_swap_move_response(
                state=state,
                trmph=trmph,
                display_board_size=display_board_size,
                pie_rule_enabled=pie_rule_enabled,
                pie_decision=pie_decision,
                model_id=model_id,
                mcts_config=_build_mcts_config_response_fields(
                    difficulty_params,
                    temperature_end=temperature,
                ),
                inline_heatmap_options=inline_heatmap_options,
            )

            _log_usage_event_with_trmph_context(
                "policy_move",
                trmph,
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
                **_build_heatmap_analytics_fields(inline_heatmap_options),
            )
            return jsonify(result)
        
        # Get model and make policy move
        model = get_model(model_id)
        move = select_policy_move(state, model, temperature)
        
        if move is None:
            _log_usage_event_with_trmph_context(
                "policy_move",
                trmph,
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
                **_build_heatmap_analytics_fields(inline_heatmap_options),
            )
            return jsonify(
                build_engine_error_payload(
                    "No valid moves available",
                    reason="no_valid_moves",
                )
            ), 400
        
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
        result["mcts_config"] = _build_mcts_config_response_fields(
            difficulty_params,
            temperature_end=temperature,
        )
        maybe_attach_inline_move_heatmap(
            result=result,
            state=new_state,
            model_id=model_id,
            heatmap_options=inline_heatmap_options,
            model_getter=get_model,
            logger=app.logger,
        )
        
        app.logger.info(f"=== POLICY API RESPONSE ===")
        app.logger.info(f"Selected move: {move_trmph}")

        _log_usage_event_with_trmph_context(
            "policy_move",
            trmph,
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
            **_build_heatmap_analytics_fields(inline_heatmap_options),
        )
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Policy move error: {e}")
        _log_usage_event_with_trmph_context(
            "policy_move",
            trmph,
            status=500,
            success=False,
            reason="exception",
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            **_build_heatmap_analytics_fields(inline_heatmap_options),
        )
        return jsonify(
            build_engine_error_payload(
                "Policy move generation failed. Please try again.",
                reason="engine_failure",
            )
        ), 500


@app.route("/api/policy_move", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_policy_move'])
def api_policy_move():
    """Make a computer move using policy sampling."""
    data = request.get_json()
    app.logger.info(f"=== POLICY API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    request_payload, error_response = _validate_interactive_move_request(data)
    if error_response:
        return error_response

    validated_data, inline_heatmap_options = request_payload
    return _execute_policy_move_from_validated_data(
        validated_data, inline_heatmap_options
    )

@app.route("/api/mcts_move", methods=["POST"])
@rate_limit(ENDPOINT_COSTS['api_mcts_move'])
def api_mcts_move():
    """Make a computer move using MCTS with diagnostic output."""
    data = request.get_json()
    app.logger.info(f"=== MCTS API CALL ===")
    app.logger.info(f"Request data: {data}")
    
    request_payload, error_response = _validate_interactive_move_request(data)
    if error_response:
        return error_response

    validated_data, inline_heatmap_options = request_payload
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
            result = _build_pie_rule_swap_move_response(
                state=state,
                trmph=trmph,
                display_board_size=display_board_size,
                pie_rule_enabled=pie_rule_enabled,
                pie_decision=pie_decision,
                model_id=difficulty_params["model"],
                mcts_config=_build_mcts_config_response_fields(difficulty_params),
                inline_heatmap_options=inline_heatmap_options,
            )
            _log_usage_event_with_trmph_context(
                "mcts_move",
                trmph,
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
                **_build_heatmap_analytics_fields(inline_heatmap_options),
            )
            return jsonify(result)
    
    if difficulty_params["algorithm"] == "policy":
        # Use policy move for lower difficulties
        return _execute_policy_move_from_validated_data(
            validated_data, inline_heatmap_options
        )
    
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
        if inline_heatmap_options.get("enabled"):
            heatmap_state = create_game_state_from_trmph(
                result.get("new_trmph", ""),
                display_board_size=display_board_size,
                context="for inline move heatmap",
            )
            maybe_attach_inline_move_heatmap(
                result=result,
                state=heatmap_state,
                model_id=difficulty_params["model"],
                heatmap_options=inline_heatmap_options,
                model_getter=get_model,
                logger=app.logger,
            )
    
    app.logger.info(f"=== MCTS API RESPONSE ===")
    app.logger.info(f"Result success: {result.get('success', 'MISSING')}")
    if result.get('success'):
        app.logger.info(f"Move made: {result.get('move_made', 'MISSING')}")
        app.logger.info(f"Game over: {result.get('game_over', 'MISSING')}")
        app.logger.info(f"Winner: {result.get('winner', 'MISSING')}")
    else:
        app.logger.error(f"Result error: {result.get('error', 'MISSING')}")

    _log_usage_event_with_trmph_context(
        "mcts_move",
        trmph,
        status=200 if result.get("success") else 500,
        success=bool(result.get("success")),
        reason=None if result.get("success") else "engine_failure",
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
        **_build_heatmap_analytics_fields(inline_heatmap_options),
    )
    
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code

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
        try:
            state, _, moves_applied = apply_trmph_sequence_to_state(state, trmph_sequence)
        except ValueError as e:
            app.logger.error(f"Invalid TRMPH sequence format: {e}")
            _log_usage_event_with_trmph_context(
                "apply_trmph_sequence",
                trmph,
                status=400,
                success=False,
                reason="invalid_sequence",
                elo_rating=elo_rating,
                display_board_size=display_board_size,
            )
            return jsonify({"error": f"Invalid TRMPH sequence format: {str(e)}"}), 400
        
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
        _log_usage_event_with_trmph_context(
            "apply_trmph_sequence",
            trmph,
            status=200,
            success=True,
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
            moves_requested=_safe_count_trmph_moves(trmph_sequence),
            moves_applied=moves_applied,
        )
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"TRMPH sequence application error: {e}")
        _log_usage_event_with_trmph_context(
            "apply_trmph_sequence",
            trmph,
            status=500,
            success=False,
            reason="exception",
            elo_rating=elo_rating,
            display_board_size=display_board_size,
            pie_rule_enabled=pie_rule_enabled,
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
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable Flask debug mode')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='Disable Flask debug mode')
    parser.set_defaults(debug=None)
    args = parser.parse_args()

    if args.debug is None:
        debug_enabled = os.getenv("SF25_WEB_DEBUG", "0").lower() not in ("0", "false", "no")
    else:
        debug_enabled = args.debug
    if debug_enabled and not _is_loopback_host(args.host):
        raise RuntimeError(
            "Refusing to start with debug enabled on a non-loopback host. "
            "Use --no-debug or bind to 127.0.0.1/localhost."
        )

    log_level_name = os.getenv("SF25_WEB_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    logging.basicConfig(level=log_level)
    app.logger.info("=" * 50)
    app.logger.info("Hex AI Public Web Server Starting...")
    app.logger.info("Debug mode: %s", debug_enabled)
    app.logger.info("Trust proxy headers: %s", TRUST_PROXY_HEADERS)
    app.logger.info("CORS allow all: %s", CORS_ALLOW_ALL)
    app.logger.info("CORS allowed origins: %s", list(CORS_ALLOWED_ORIGINS))
    app.logger.info("=" * 50)
    app.run(
        debug=debug_enabled,
        use_reloader=False,
        use_debugger=debug_enabled,
        threaded=False,
        host=args.host,
        port=args.port,
    )
