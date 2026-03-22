"""
Web-specific configuration constants.

These are intentionally scoped to the web UI / interactive play, rather than the
training/tournament pipeline.
"""

# Default search budget for manual interactive MCTS from the dev web page.
INTERACTIVE_DEFAULT_MCTS_NUM_SIMULATIONS = 90

# Confidence-based early termination threshold for interactive MCTS.
#
# This is compared against the *signed value head output* in the player-to-move
# reference frame. If |v| >= threshold, MCTS can terminate early and select the
# top policy move (see `hex_ai.inference.mcts.AlgorithmTerminationChecker`).
INTERACTIVE_CONFIDENCE_TERMINATION_THRESHOLD = 0.95

