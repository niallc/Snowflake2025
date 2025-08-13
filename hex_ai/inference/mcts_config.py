"""
Configuration objects for MCTS batching/orchestration.

Centralizes knobs for batch timing and scheduling so callers can tune
throughput vs latency without editing core MCTS logic.
"""

from dataclasses import dataclass


@dataclass
class MCTSCoreConfig:
    """
    Core MCTS algorithm parameters.

    Centralized defaults so engines inherit consistent values unless callers
    explicitly override. Call sites should generally NOT override these unless
    exploring or with a clearly documented, context-specific reason.
    """

    # PUCT exploration constant
    exploration_constant: float = 1.4
    # Terminal state absolute value (win/loss magnitude)
    win_value: float = 1.5
    # Discount factor for move-count penalty during backpropagation
    discount_factor: float = 0.98


@dataclass
class BatchedMCTSOrchestration:
    """
    Orchestration parameters for batched MCTS.

    - evaluator_max_wait_ms: timeout for batch coalescing in the evaluator
    - optimal_batch_size: target batch size
    - selection_wait_ms: minimum wait before selection to let callbacks apply
    - min_batch_before_force: only force-process if queue has at least this many
    - min_wait_before_force_ms: force-process if we haven't forced in this time
    - max_inflight: cap for concurrently pending evaluations
    - batch_wait_s: condition-variable wait between scheduler iterations
    """

    evaluator_max_wait_ms: int = 200    # up from 30ms
    optimal_batch_size: int = 64        # 64 suggested
    selection_wait_ms: int = 150        # down from 200
    min_batch_before_force: int = 32    # changed from 24 to suggested 32
    min_wait_before_force_ms: int = 60  # changed from 20ms to suggested 60ms
    max_inflight: int = 128             # no suggestion
    batch_wait_s: float = 0.01          # up from 0.005


