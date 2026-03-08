"""
Gumbel-specific MCTS orchestration helpers.

This module isolates Gumbel root-selection flow and diagnostics from the
default batched MCTS search implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from hex_ai.enums import Player
from hex_ai.inference.mcts_config import (
    DEFAULT_GUMBEL_ROOT_TEMPERATURE,
    DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF,
)
from hex_ai.utils.format_conversion import rowcol_to_trmph, tensor_to_trmph
from hex_ai.utils.gumbel_utils import gumbel_alpha_zero_root_batched
from hex_ai.utils.state_utils import board_key
from hex_ai.utils.timing import MCTSTimingTracker
from hex_ai.value_utils import red_ref_signed_to_ptm_ref_signed, signed_to_prob

if TYPE_CHECKING:
    from hex_ai.inference.mcts import MCTSNode


class MCTSGumbelMixin:
    """Mixin for Gumbel root selection flow and associated diagnostics."""

    def _decorate_gumbel_score_rows_with_moves(
        self, rows: List[Dict[str, Any]], board_size: int
    ) -> List[Dict[str, Any]]:
        """Attach TRMPH move labels to rows that contain tensor action indices."""
        decorated_rows: List[Dict[str, Any]] = []
        for row in rows:
            row_copy = dict(row)
            action = row_copy.get("tensor_action", None)
            if action is not None:
                try:
                    row_copy["move"] = tensor_to_trmph(int(action), board_size)
                except Exception:
                    row_copy["move"] = None
            decorated_rows.append(row_copy)
        return decorated_rows

    def _record_gumbel_trace_event(self, event: Dict[str, Any], board_size: int) -> None:
        """Record a Gumbel-specific detailed trace event with human-readable move labels."""
        if not self.detailed_exploration_enabled:
            return

        event_copy = dict(event)

        # Decorate single-action fields
        if event_copy.get("selected_action", None) is not None:
            try:
                event_copy["selected_move"] = tensor_to_trmph(
                    int(event_copy["selected_action"]), board_size
                )
            except Exception:
                event_copy["selected_move"] = None
        if event_copy.get("tensor_action", None) is not None:
            try:
                event_copy["move"] = tensor_to_trmph(
                    int(event_copy["tensor_action"]), board_size
                )
            except Exception:
                event_copy["move"] = None

        # Decorate action-list fields
        for actions_key, moves_key in (
            ("kept_actions", "kept_moves"),
            ("dropped_actions", "dropped_moves"),
        ):
            if actions_key in event_copy and isinstance(event_copy[actions_key], list):
                moves: List[str] = []
                for action in event_copy[actions_key]:
                    try:
                        moves.append(tensor_to_trmph(int(action), board_size))
                    except Exception:
                        continue
                event_copy[moves_key] = moves

        # Decorate score row collections
        for rows_key in ("candidate_rows", "selected_rows", "excluded_rows", "final_rank_rows"):
            if rows_key in event_copy and isinstance(event_copy[rows_key], list):
                event_copy[rows_key] = self._decorate_gumbel_score_rows_with_moves(
                    event_copy[rows_key], board_size
                )

        self.exploration_trace.append(event_copy)

    @staticmethod
    def _player_to_color_label(player: Player) -> str:
        """Convert Player enum to a simple lowercase color label."""
        if player == Player.RED:
            return "red"
        if player == Player.BLUE:
            return "blue"
        return str(player)

    @staticmethod
    def _red_ref_signed_to_root_ref_signed(value_signed_red_ref: float, root_player: Player) -> float:
        """Convert a red-reference signed value into root-player reference frame."""
        return float(value_signed_red_ref if root_player == Player.RED else -value_signed_red_ref)

    def _debug_value_summary_for_state(self, state, root_player: Player) -> Dict[str, Any]:
        """
        Get value-head summary for an arbitrary state.

        Uses cache when available; otherwise performs a direct model eval without mutating cache.
        """
        cached = self._get_from_cache(board_key(state))
        from_cache = cached is not None

        if cached is not None:
            _, value_signed_red_ref = cached
        else:
            enc = state.get_board_tensor().to(dtype=torch.float32)
            batch = torch.stack([enc], dim=0)
            _, value_cpu, _ = self.model.infer_timed(batch)
            value_signed_red_ref = float(value_cpu[0].item())

        value_signed_red_ref = float(value_signed_red_ref)
        value_signed_root_ref = self._red_ref_signed_to_root_ref_signed(value_signed_red_ref, root_player)
        value_signed_ptm_ref = float(red_ref_signed_to_ptm_ref_signed(value_signed_red_ref, state.current_player_enum))

        return {
            "red_ref_signed": value_signed_red_ref,
            "root_ref_signed": value_signed_root_ref,
            "root_win_prob": float(signed_to_prob(value_signed_root_ref)),
            "ptm_ref_signed": value_signed_ptm_ref,
            "ptm_win_prob": float(signed_to_prob(value_signed_ptm_ref)),
            "to_play": self._player_to_color_label(state.current_player_enum),
            "from_cache": bool(from_cache),
        }

    def _build_gumbel_action_dive_event(
        self, root: MCTSNode, tensor_action: int, board_size: int, top_replies: int = 6
    ) -> Optional[Dict[str, Any]]:
        """
        Build a deep-dive debug event for one root action:
        - Root child tree Q/N
        - Value-head eval after root move
        - Opponent reply diagnostics by visits and by prior
        """
        if tensor_action not in root.legal_indices:
            return None

        root_player = root.to_play
        action = int(tensor_action)
        child_idx = root.legal_indices.index(action)
        move = tensor_to_trmph(action, board_size)
        root_child_visits = int(root.N[child_idx])
        root_child_q_ptm_signed = float(root.Q[child_idx]) if root_child_visits > 0 else 0.0
        root_child_q_01 = float((root_child_q_ptm_signed + 1.0) / 2.0) if root_child_visits > 0 else 0.5

        # Construct state after root move and evaluate value head.
        move_row, move_col = root.legal_moves[child_idx]
        state_after_root = root.state.make_move(move_row, move_col)
        value_after_root = self._debug_value_summary_for_state(state_after_root, root_player)

        event: Dict[str, Any] = {
            "type": "gumbel_action_dive",
            "tensor_action": action,
            "move": move,
            "root_player": self._player_to_color_label(root_player),
            "opponent_to_play": self._player_to_color_label(state_after_root.current_player_enum),
            "root_child_visits": root_child_visits,
            "root_child_q_ptm_signed": root_child_q_ptm_signed,
            "root_child_q_01": root_child_q_01,
            "value_after_root": value_after_root,
        }

        child_node = root.children[child_idx]
        if child_node is None:
            event["note"] = "Child node was not realized during search; no opponent reply tree stats available."
            event["reply_count_total"] = int(len(state_after_root.get_legal_moves()))
            event["reply_count_visited"] = 0
            event["top_replies_by_visits"] = []
            event["top_replies_by_policy"] = []
            return event

        reply_count_total = len(child_node.legal_moves)
        visited_indices = [i for i, n in enumerate(child_node.N) if int(n) > 0]
        visited_indices.sort(key=lambda i: int(child_node.N[i]), reverse=True)
        visited_top = visited_indices[:top_replies]

        policy_top: List[int] = []
        if child_node.is_expanded and len(child_node.P) == reply_count_total and reply_count_total > 0:
            policy_top = list(np.argsort(child_node.P)[::-1][:top_replies].astype(int))

        # Evaluate value head after replies for the union of indices shown in the report.
        eval_indices = sorted(set(visited_top + policy_top))
        value_after_reply_by_idx: Dict[int, Dict[str, Any]] = {}
        for idx in eval_indices:
            reply_row, reply_col = child_node.legal_moves[idx]
            reply_state = child_node.state.make_move(reply_row, reply_col)
            value_after_reply_by_idx[idx] = self._debug_value_summary_for_state(reply_state, root_player)

        def build_reply_row(idx: int) -> Dict[str, Any]:
            reply_row, reply_col = child_node.legal_moves[idx]
            reply_move = rowcol_to_trmph(reply_row, reply_col, board_size)
            visits = int(child_node.N[idx])
            prior = float(child_node.P[idx]) if child_node.is_expanded and len(child_node.P) == reply_count_total else None

            if visits > 0:
                q_opp_ptm_signed = float(child_node.Q[idx])  # Child node is opponent-to-play.
                q_root_ref_signed = float(-q_opp_ptm_signed)  # Opponent perspective -> root perspective.
                q_root_win_prob = float(signed_to_prob(q_root_ref_signed))
                q_source = "tree"
            else:
                q_opp_ptm_signed = None
                q_root_ref_signed = None
                q_root_win_prob = None
                q_source = "unvisited"

            return {
                "move": reply_move,
                "visits": visits,
                "prior": prior,
                "q_source": q_source,
                "q_opp_ptm_signed": q_opp_ptm_signed,
                "q_root_ref_signed": q_root_ref_signed,
                "q_root_win_prob": q_root_win_prob,
                "value_after_reply": value_after_reply_by_idx.get(idx, None),
            }

        event["reply_count_total"] = int(reply_count_total)
        event["reply_count_visited"] = int(len(visited_indices))
        top_replies_by_visits = [build_reply_row(idx) for idx in visited_top]
        top_replies_by_policy = [build_reply_row(idx) for idx in policy_top]
        event["top_replies_by_visits"] = top_replies_by_visits
        event["top_replies_by_policy"] = top_replies_by_policy

        # Debug-only stress-check on the top policy replies:
        # summarizes whether potentially dangerous unvisited responses exist.
        policy_rows_for_stats: List[Dict[str, Any]] = []
        for row in top_replies_by_policy:
            prior_val = row.get("prior", None)
            value_summary = row.get("value_after_reply", None)
            root_win_prob = value_summary.get("root_win_prob", None) if isinstance(value_summary, dict) else None
            if prior_val is None:
                continue
            try:
                prior = float(prior_val)
                if not np.isfinite(prior) or prior <= 0.0:
                    continue
            except Exception:
                continue

            if root_win_prob is not None:
                try:
                    root_win_prob = float(root_win_prob)
                except Exception:
                    root_win_prob = None

            policy_rows_for_stats.append(
                {
                    "move": row.get("move", None),
                    "visits": int(row.get("visits", 0)),
                    "prior": prior,
                    "root_win_prob": root_win_prob,
                }
            )

        total_policy_mass = float(sum(item["prior"] for item in policy_rows_for_stats))
        visited_policy_mass = float(sum(item["prior"] for item in policy_rows_for_stats if item["visits"] > 0))
        policy_rows_with_value = [item for item in policy_rows_for_stats if item["root_win_prob"] is not None]

        weighted_value_head_root_win = None
        if policy_rows_with_value:
            denom = float(sum(item["prior"] for item in policy_rows_with_value))
            if denom > 1e-12:
                weighted_value_head_root_win = float(
                    sum(item["prior"] * float(item["root_win_prob"]) for item in policy_rows_with_value) / denom
                )

        worst_policy_reply = None
        if policy_rows_with_value:
            worst = min(policy_rows_with_value, key=lambda item: float(item["root_win_prob"]))
            worst_policy_reply = {
                "move": worst["move"],
                "visits": int(worst["visits"]),
                "prior": float(worst["prior"]),
                "root_win_prob": float(worst["root_win_prob"]),
            }

        event["policy_reply_stress"] = {
            "top_policy_count": int(len(policy_rows_for_stats)),
            "top_policy_count_visited": int(sum(1 for item in policy_rows_for_stats if item["visits"] > 0)),
            "top_policy_prior_mass": total_policy_mass,
            "top_policy_prior_mass_visited": visited_policy_mass,
            "top_policy_prior_mass_visited_ratio": (
                float(visited_policy_mass / total_policy_mass) if total_policy_mass > 1e-12 else None
            ),
            "weighted_value_head_root_win": weighted_value_head_root_win,
            "worst_policy_reply": worst_policy_reply,
        }
        event["note"] = (
            "Reply Q values are from opponent perspective at depth-1 child (q_opp_ptm_signed). "
            "q_root_ref_signed flips sign to root perspective. "
            "value_after_reply is direct value-head eval of the resulting position."
        )
        return event

    def _load_gumbel_policy_inputs(
        self,
        root: MCTSNode,
        timing_tracker: MCTSTimingTracker
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Load policy logits and legal mask for Gumbel root selection."""
        timing_tracker.start_timing("gumbel_policy_retrieval")
        board_size = int(root.state.get_board_tensor().shape[-1])
        policy_logits_full, legal_mask = self._get_policy_logits_and_legal_mask(root.state, root.legal_indices)
        timing_tracker.end_timing("gumbel_policy_retrieval")
        return board_size, policy_logits_full, legal_mask

    def _compute_gumbel_priors(
        self,
        policy_logits_full: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute legal priors for Gumbel root selection."""
        return self._root_priors_from_logits(policy_logits_full, legal_mask, apply_dirichlet=False)

    def _validate_gumbel_temperature_contract(self) -> None:
        """
        Validate the fixed-temperature contract for Gumbel root selection.

        Gumbel root selection currently does not support temperature scaling/cutoff
        semantics. Config validation enforces this too, but we guard at runtime so
        direct config mutation still fails fast.
        """
        if not self.cfg.gumbel_temperature_enabled:
            raise ValueError(
                "gumbel_temperature_enabled=False is unsupported: "
                f"Gumbel root selection uses fixed temperature={DEFAULT_GUMBEL_ROOT_TEMPERATURE}."
            )
        if self.cfg.gumbel_temperature_deterministic_cutoff != DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF:
            raise ValueError(
                "gumbel_temperature_deterministic_cutoff is unsupported and must remain "
                f"{DEFAULT_GUMBEL_TEMPERATURE_DETERMINISTIC_CUTOFF}."
            )

    def _set_gumbel_selected_action(self, selected_action: int, selected_tensor_action: int) -> None:
        """Persist selected Gumbel root action in both local and tensor-index forms."""
        self._gumbel_selected_action = int(selected_action)
        self._gumbel_selected_tensor_action = int(selected_tensor_action)
        self._used_gumbel_root_selection = True

    def _build_gumbel_child_accessors(
        self,
        root: MCTSNode
    ) -> Tuple[Callable[[int], float], Callable[[int], int]]:
        """Build Q and N accessors expected by gumbel_alpha_zero_root_batched."""
        action_to_legal_idx = {int(action): idx for idx, action in enumerate(root.legal_indices)}

        def q_of_child(action: int) -> float:
            legal_move_idx = action_to_legal_idx[int(action)]
            if root.N[legal_move_idx] == 0:
                return 0.5
            q_raw = root.Q[legal_move_idx]
            return (q_raw + 1.0) / 2.0

        def n_of_child(action: int) -> int:
            legal_move_idx = action_to_legal_idx[int(action)]
            return int(root.N[legal_move_idx])

        return q_of_child, n_of_child

    def _build_gumbel_trace_callback(self, board_size: int) -> Optional[Callable[[Dict[str, Any]], None]]:
        """Build optional trace callback for detailed Gumbel diagnostics."""
        if not self.detailed_exploration_enabled:
            return None
        return lambda event: self._record_gumbel_trace_event(event, board_size)

    def _run_batched_gumbel_algorithm(
        self,
        root: MCTSNode,
        total_sims: int,
        board_size: int,
        priors_full: np.ndarray,
        q_of_child: Callable[[int], float],
        n_of_child: Callable[[int], int],
        verbose: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """Run the batched Gumbel root-selection algorithm and return selection + metrics."""
        legal_actions = root.legal_indices.copy()
        logits_for_gumbel = np.log(np.clip(priors_full, 1e-12, 1.0))

        if verbose >= 5:
            print("MCTS GUMBEL CALL DEBUG:")
            print(f"  Temperature: {DEFAULT_GUMBEL_ROOT_TEMPERATURE}")
            print(f"  Total sims: {total_sims}")
            print(f"  Legal actions: {len(legal_actions)}")
            print(f"  Logits range: [{np.min(logits_for_gumbel):.3f}, {np.max(logits_for_gumbel):.3f}]")
            print(f"  Top policy action: {int(np.argmax(logits_for_gumbel))}")

        trace_event_cb = self._build_gumbel_trace_callback(board_size)
        return gumbel_alpha_zero_root_batched(
            mcts=self,
            root=root,
            policy_logits=logits_for_gumbel,
            total_sims=total_sims,
            legal_actions=legal_actions,
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            m=self.cfg.gumbel_m_candidates,
            c_visit=self.cfg.gumbel_c_visit,
            c_scale=self.cfg.gumbel_c_scale,
            temperature=DEFAULT_GUMBEL_ROOT_TEMPERATURE,
            verbose=verbose,
            candidate_power_scale=self.cfg.gumbel_candidate_power_scale,
            candidate_power_rate=self.cfg.gumbel_candidate_power_rate,
            candidate_power_offset=self.cfg.gumbel_candidate_power_offset,
            candidate_min=self.cfg.gumbel_candidate_min,
            candidate_max=self.cfg.gumbel_candidate_max,
            use_gumbel_in_final_eval=self.cfg.gumbel_use_gumbel_in_final_eval,
            eval_mode=False,
            trace_event=trace_event_cb,
        )

    def _record_gumbel_metrics(self, gumbel_metrics: Dict[str, Any]) -> None:
        """Record Gumbel performance metrics for inclusion in final stats."""
        self._gumbel_nn_calls_per_move = gumbel_metrics["nn_calls_per_move"]
        self._gumbel_total_leaves_evaluated = gumbel_metrics["total_leaves_evaluated"]
        # For distinct leaves, we'll use final MCTS metrics since per-batch stats don't include this.
        self._gumbel_distinct_leaves_evaluated = 0
        self._gumbel_candidates_m = gumbel_metrics["candidates_m"]
        self._gumbel_rounds_R = gumbel_metrics["rounds_R"]
        self._gumbel_timing_breakdown = gumbel_metrics.get("timing_breakdown", {})

    @staticmethod
    def _extract_required_gumbel_final_rank_rows(
        gumbel_metrics: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """
        Extract compact final rank rows required for downstream target generation.

        This is not debug-only state; self-play policy-target construction depends
        on these rows being present and valid.
        """
        final_rows_raw = gumbel_metrics.get("final_rank_rows", None)
        if not isinstance(final_rows_raw, list) or not final_rows_raw:
            raise RuntimeError(
                "Missing required Gumbel final_rank_rows in gumbel metrics."
            )

        compact_rows: List[Dict[str, float]] = []
        seen_actions: set[int] = set()
        previous_score: Optional[float] = None
        for row in final_rows_raw:
            if not isinstance(row, dict):
                raise RuntimeError(
                    "Invalid Gumbel final_rank_rows entry type: expected dict."
                )
            action_raw = row.get("tensor_action", None)
            score_raw = row.get("score_without_gumbel", None)
            if action_raw is None or score_raw is None:
                raise RuntimeError(
                    "Gumbel final_rank_rows entries must include tensor_action and score_without_gumbel."
                )

            action = int(action_raw)
            if action in seen_actions:
                raise RuntimeError(
                    f"Duplicate tensor_action {action} in gumbel final_rank_rows."
                )
            seen_actions.add(action)

            score = float(score_raw)
            if not np.isfinite(score):
                raise RuntimeError(
                    f"Non-finite score_without_gumbel in gumbel final_rank_rows: {score_raw!r}"
                )
            if previous_score is not None and score > previous_score + 1e-12:
                raise RuntimeError(
                    "Gumbel final_rank_rows are not sorted by descending score_without_gumbel."
                )
            previous_score = score

            compact_rows.append(
                {
                    "tensor_action": action,
                    "score_without_gumbel": score,
                }
            )

        return compact_rows

    @staticmethod
    def _extract_required_gumbel_top_m_candidate_rows(
        gumbel_metrics: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """
        Extract compact top-m candidate rows required for richer Gumbel targets.

        These rows preserve the clean pre-search log-priors for the full Gumbel
        candidate set so self-play can build policy targets over all searched
        candidates rather than only the final ranked survivors.
        """
        candidate_rows_raw = gumbel_metrics.get("top_m_selected_rows", None)
        if not isinstance(candidate_rows_raw, list) or not candidate_rows_raw:
            raise RuntimeError(
                "Missing required Gumbel top_m_selected_rows in gumbel metrics."
            )

        compact_rows: List[Dict[str, float]] = []
        seen_actions: set[int] = set()
        for row in candidate_rows_raw:
            if not isinstance(row, dict):
                raise RuntimeError(
                    "Invalid Gumbel top_m_selected_rows entry type: expected dict."
                )
            action_raw = row.get("tensor_action", None)
            log_prior_raw = row.get("log_prior", None)
            prior_raw = row.get("prior", None)
            if action_raw is None or log_prior_raw is None or prior_raw is None:
                raise RuntimeError(
                    "Gumbel top_m_selected_rows entries must include tensor_action, log_prior, and prior."
                )

            action = int(action_raw)
            if action in seen_actions:
                raise RuntimeError(
                    f"Duplicate tensor_action {action} in gumbel top_m_selected_rows."
                )
            seen_actions.add(action)

            log_prior = float(log_prior_raw)
            if not np.isfinite(log_prior):
                raise RuntimeError(
                    f"Non-finite log_prior in gumbel top_m_selected_rows: {log_prior_raw!r}"
                )

            prior = float(prior_raw)
            if not np.isfinite(prior) or prior < 0.0:
                raise RuntimeError(
                    f"Invalid prior in gumbel top_m_selected_rows: {prior_raw!r}"
                )

            compact_rows.append(
                {
                    "tensor_action": action,
                    "log_prior": log_prior,
                    "prior": prior,
                }
            )

        return compact_rows

    def _record_required_gumbel_target_fields(
        self,
        gumbel_metrics: Dict[str, Any],
    ) -> None:
        """
        Record required Gumbel fields used outside inference-time debugging.

        Keeping this separate from debug capture avoids accidental loss of
        required fields due to best-effort formatting failures.
        """
        compact_rows = self._extract_required_gumbel_final_rank_rows(gumbel_metrics)
        compact_candidate_rows = self._extract_required_gumbel_top_m_candidate_rows(
            gumbel_metrics
        )
        self._gumbel_final_rank_rows = compact_rows
        self._gumbel_top_m_candidate_rows = compact_candidate_rows
        self._gumbel_final_score_gap_top1_top2 = (
            float(compact_rows[0]["score_without_gumbel"] - compact_rows[1]["score_without_gumbel"])
            if len(compact_rows) >= 2
            else None
        )
        self._gumbel_final_score_gap_top1_top3 = (
            float(compact_rows[0]["score_without_gumbel"] - compact_rows[2]["score_without_gumbel"])
            if len(compact_rows) >= 3
            else None
        )

    def _merge_forced_round_stats_into_timing_tracker(
        self,
        timing_tracker: MCTSTimingTracker,
        gumbel_metrics: Dict[str, Any],
    ) -> None:
        """
        Merge forced-round MCTS timing totals into the outer run timing tracker.

        Gumbel root selection delegates core simulations via run_forced_root_actions(),
        which uses its own timing tracker. Without this merge, top-level MCTS stats
        can show zero NN/CPU timing despite real simulations being executed.
        """
        forced = gumbel_metrics.get("forced_stats_totals", None)
        if not isinstance(forced, dict):
            return

        timing_tracker.batch_count += int(forced.get("batch_count", 0))
        batch_sizes = forced.get("batch_sizes", [])
        if isinstance(batch_sizes, list):
            timing_tracker.batch_sizes.extend(batch_sizes)

        timing_tracker.h2d_ms_total += float(forced.get("h2d_ms", 0.0))
        timing_tracker.forward_ms_total += float(forced.get("forward_ms", 0.0))
        timing_tracker.pure_forward_ms_total += float(forced.get("pure_forward_ms", 0.0))
        timing_tracker.sync_ms_total += float(forced.get("sync_ms", 0.0))
        timing_tracker.d2h_ms_total += float(forced.get("d2h_ms", 0.0))

        for source_key, tracker_key in (
            ("select_ms", "select"),
            ("encode_ms", "encode"),
            ("stack_ms", "stack"),
            ("expand_ms", "expand"),
            ("backprop_ms", "backprop"),
            ("cache_lookup_ms", "cache_lookup"),
            ("state_creation_ms", "state_creation"),
            ("make_move_ms", "make_move"),
        ):
            timing_tracker.timings[tracker_key] = (
                timing_tracker.timings.get(tracker_key, 0.0)
                + float(forced.get(source_key, 0.0))
            )

    def _finalize_gumbel_selection(
        self,
        root: MCTSNode,
        selected_tensor_action: int,
        timing_tracker: MCTSTimingTracker,
        verbose: int
    ) -> int:
        """Finalize selected Gumbel action and close timing scopes."""
        timing_tracker.start_timing("gumbel_final_conversion")
        selected_action = root.legal_indices.index(int(selected_tensor_action))
        timing_tracker.end_timing("gumbel_final_conversion")
        timing_tracker.end_timing("gumbel_selection")

        if verbose >= 2:
            print(f"Gumbel selection completed. Selected tensor action {selected_tensor_action} "
                  f"-> legal action {selected_action} ({root.legal_moves[selected_action]})")

        self._set_gumbel_selected_action(selected_action, int(selected_tensor_action))
        return selected_action

    def _capture_gumbel_debug_fields(
        self,
        root: MCTSNode,
        selected_tensor_action: int,
        gumbel_metrics: Dict[str, Any],
        board_size: int
    ) -> None:
        """Best-effort capture of extra Gumbel debug fields and deep-dive events."""
        try:
            self._gumbel_v_pi_01 = gumbel_metrics.get("v_pi_01", None)
            final_rows_raw = gumbel_metrics.get("final_rank_rows", []) or []
            final_rows = self._decorate_gumbel_score_rows_with_moves(final_rows_raw, board_size)
            self._gumbel_final_rank_top5 = final_rows[:5] if final_rows else None
            self._gumbel_final_rank_top_move_trmph = final_rows[0]["move"] if final_rows else None

            if self.detailed_exploration_enabled:
                dive_actions: List[int] = []

                for row in (gumbel_metrics.get("last_round_rows", []) or []):
                    action = row.get("tensor_action", None)
                    if action is None:
                        continue
                    action_int = int(action)
                    if action_int not in dive_actions:
                        dive_actions.append(action_int)

                if not dive_actions:
                    for row in final_rows_raw:
                        action = row.get("tensor_action", None)
                        if action is None:
                            continue
                        action_int = int(action)
                        if action_int not in dive_actions:
                            dive_actions.append(action_int)

                selected_action_int = int(selected_tensor_action)
                if selected_action_int in dive_actions:
                    dive_actions = [selected_action_int] + [a for a in dive_actions if a != selected_action_int]
                else:
                    dive_actions = [selected_action_int] + dive_actions

                for action in dive_actions[:3]:
                    dive_event = self._build_gumbel_action_dive_event(root, action, board_size, top_replies=6)
                    if dive_event is not None:
                        self.exploration_trace.append(dive_event)
        except Exception:
            # Best-effort debug only: do not risk crashing inference due to debug formatting.
            self._gumbel_final_rank_top5 = None
            self._gumbel_final_rank_top_move_trmph = None
            self._gumbel_v_pi_01 = None

    def _run_gumbel_root_selection(self, root: MCTSNode, total_sims: int,
                                 timing_tracker: MCTSTimingTracker, verbose: int) -> Dict[str, Any]:
        """
        Run Gumbel-AlphaZero root selection for small simulation budgets.

        This method uses the batched Gumbel implementation that reuses the existing
        MCTS batching infrastructure for maximum efficiency.
        """
        self._validate_gumbel_temperature_contract()
        timing_tracker.start_timing("gumbel_selection")
        board_size, policy_logits_full, legal_mask = self._load_gumbel_policy_inputs(root, timing_tracker)

        timing_tracker.start_timing("gumbel_algorithm")
        priors_full = self._compute_gumbel_priors(policy_logits_full, legal_mask)

        q_of_child, n_of_child = self._build_gumbel_child_accessors(root)
        selected_tensor_action, gumbel_metrics = self._run_batched_gumbel_algorithm(
            root=root,
            total_sims=total_sims,
            board_size=board_size,
            priors_full=priors_full,
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            verbose=verbose,
        )

        self._record_gumbel_metrics(gumbel_metrics)
        self._record_required_gumbel_target_fields(gumbel_metrics)
        self._merge_forced_round_stats_into_timing_tracker(timing_tracker, gumbel_metrics)
        if verbose >= 4:
            print(f"Gumbel root: fixed temperature={DEFAULT_GUMBEL_ROOT_TEMPERATURE:.3f}")
        timing_tracker.end_timing("gumbel_algorithm")

        self._finalize_gumbel_selection(root, selected_tensor_action, timing_tracker, verbose)
        self._capture_gumbel_debug_fields(root, selected_tensor_action, gumbel_metrics, board_size)
        return timing_tracker.get_final_stats()
