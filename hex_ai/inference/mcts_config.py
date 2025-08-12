"""
Configuration objects for MCTS batching/orchestration.

Centralizes knobs for batch timing and scheduling so callers can tune
throughput vs latency without editing core MCTS logic.
"""

from dataclasses import dataclass


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

    evaluator_max_wait_ms: int = 30
    optimal_batch_size: int = 64
    selection_wait_ms: int = 200
    min_batch_before_force: int = 24
    min_wait_before_force_ms: int = 20
    max_inflight: int = 128
    batch_wait_s: float = 0.005


