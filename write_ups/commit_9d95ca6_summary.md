# Summary of Changes in Commit 9d95ca6c29b3d662b6daab67543502972035f221

This document summarizes the changes made in commit 9d95ca6c29b3d662b6daab67543502972035f221, which aimed to refactor the MCTS implementation to consistently use signed `[-1, 1]` values instead of a mix of `[0, 1]` probabilities and signed values.

## Key Changes by File

### `hex_ai/inference/mcts.py`

The most significant changes occurred in this file, where the core MCTS logic was updated to handle signed values.

- **Value Representation**: The internal representation of node values (`W` scores) and neural network evaluations was switched from probabilities `[0, 1]` to signed values `[-1, 1]`.
- **Variable Renaming**: The variable `value_logit` was consistently renamed to `value_signed` to more accurately describe the `tanh` activated output of the model.
- **Backpropagation**: The `_backpropagate_path` function was modified to work with signed values. It now uses `signed_for_player_to_move` to correctly attribute the value from the perspective of the player whose turn it was.
- **Depth Discounting**: Discounting is now applied in the signed value space, which shrinks the value towards `0` (a neutral outcome) rather than an arbitrary value. The new function `apply_depth_discount_signed` is used for this.
- **Terminal Node Values**: The `_get_terminal_value` method was updated to return `1.0` for a Red win and `-1.0` for a Blue win, aligning with the signed value convention.
- **Neural Network Value**: The `_get_neural_network_value` method now directly returns the signed `[-1, 1]` value from the model's cache.
- **Edge Conversions**: Probabilities are now only computed at the "edges" of the algorithm, such as when providing the final win probability or for the confidence check for early termination. This is handled by the `to_prob` function.

### `hex_ai/value_utils.py`

This file introduced a new set of utility functions to abstract the logic for handling and converting between different value representations.

- **`to_prob(v_signed)`**: Converts a signed value in `[-1, 1]` to a probability in `[0, 1]`.
- **`to_signed(p_prob)`**: Converts a probability in `[0, 1]` to a signed value in `[-1, 1]`.
- **`signed_for_player_to_move(v_red_signed, player)`**: Flips the sign of a Red-centric signed value if it's Blue's turn to move.
- **`prob_for_player_to_move(p_red, player)`**: Converts a Red-centric win probability to the perspective of the current player.
- **`apply_depth_discount_signed(v, gamma, dist)`**: Applies a discount factor to a signed value, shrinking it towards 0.
- **`apply_depth_discount_toward_neutral_prob(p, gamma, dist)`**: Applies a discount factor to a probability, shrinking it towards the neutral value of 0.5.

### `hex_ai/models.py`

- **Variable Renaming**: The output of the `TwoHeadedResNet.forward` method was renamed from `value_logit` to `value_signed` to maintain consistency with the rest of the codebase.

## Potential Sources of Bugs

Given the goal of this refactoring was to fix bugs related to value representation, the changes themselves are the primary place to look for new bugs. The following areas are particularly suspect:

1.  **Backpropagation Logic**: The change in `_backpropagate_path` to use `signed_for_player_to_move` is a critical change. If the logic here is incorrect, the MCTS tree will be updated with incorrect values.
2.  **Depth Discounting**: The new `apply_depth_discount_signed` function changes how depth discounting is applied. An incorrect implementation or an improperly tuned discount factor could lead to suboptimal play.
3.  **Edge Conversions**: Any place where a signed value is converted back to a probability (e.g., for the final output or for termination checks) is a potential source of error if the conversion is not done correctly.
4.  **PUCT Hyperparameter**: The commit adds a `TODO` to retune the `c_puct` hyperparameter. The existing value may not be optimal for Q-values in the `[-1, 1]` range, potentially unbalancing exploration and exploitation.

By examining these specific areas in the provided diff, it should be possible to identify the source of the regression.

## Full Code Diff

```diff
diff --git a/hex_ai/inference/mcts.py b/hex_ai/inference/mcts.py
index e0884ac..5757a96 100644
--- a/hex_ai/inference/mcts.py
+++ b/hex_ai/inference/mcts.py
@@ -28,7 +28,7 @@ from collections import OrderedDict
 
 # ---- Package imports ----
 from hex_ai.enums import Player, Winner
-from hex_ai.value_utils import player_to_winner
+from hex_ai.value_utils import player_to_winner, prob_for_player_to_move, apply_depth_discount_toward_neutral_prob, signed_for_player_to_move, apply_depth_discount_signed, distance_to_leaf, to_prob
 from hex_ai.inference.game_engine import HexGameState, HexGameEngine
 from hex_ai.inference.model_wrapper import ModelWrapper
 from hex_ai.utils.perf import PERF
@@ -255,18 +255,16 @@ class AlgorithmTerminationChecker:
         return None  # Continue with MCTS
     
     def _get_root_win_probability(self, root: MCTSNode, eval_cache: OrderedDict[int, Tuple[np.ndarray, float]]) -> float:
-        """Get win probability for current player from neural network."""
+        """Get win probability for current player from neural network (edge conversion)."""
         cached = eval_cache.get(root.state_hash)
         if cached is None:
             raise RuntimeError(f"Root state not found in cache: {root.state_hash}")
-        _, value_logit = cached
-        # Convert from [-1, 1] range to [0, 1] probability using centralized utility
-        nn_win_prob = float(ValuePredictor.model_output_to_probability(value_logit))
-        
-        if root.to_play == Player.RED:
-            return nn_win_prob
-        else:
-            return 1.0 - nn_win_prob
+        _, value_signed = cached
+        # Convert signed value to probability only at the edge (for confidence termination)
+        # value_signed is the tanh-activated output in [-1,1] range
+        v_red_signed = float(value_signed)
+        v_player_signed = signed_for_player_to_move(v_red_signed, root.to_play)
+        return to_prob(v_player_signed)
     
     def _is_position_clearly_decided(self, win_prob: float) -> bool:
         """Check if position is clearly won or lost."""
@@ -414,8 +412,9 @@ class BaselineMCTS:
         # Always use global RNG state - randomness should be controlled externally
         # This ensures MCTS instances don't interfere with each other's randomness
 
-        # LRU Cache: board_key -> (policy_logits_np [A], value_logit_float)
+        # LRU Cache: board_key -> (policy_logits_np [A], value_signed_float)
         # Uses OrderedDict for O(1) LRU eviction
+        # Note: value_signed is the tanh-activated output in [-1,1] range (not a logit)
         self.eval_cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
         self.cache_hits = 0
         self.cache_misses = 0
@@ -531,7 +530,7 @@ class BaselineMCTS:
         cached = self._get_from_cache(root.state_hash)
         if cached is not None:
             self.cache_hits += 1
-            policy_np, value_logit = cached
+            policy_np, value_signed = cached
             self._expand_node_from_policy(root, policy_np, board_size, action_size)
         else:
             self.cache_misses += 1
@@ -541,10 +540,10 @@ class BaselineMCTS:
             
             policy_cpu, value_cpu, _ = self.model.infer_timed(batch)
             policy_np = policy_cpu[0].numpy()
-            value_logit = float(value_cpu[0].item())
+            value_signed = float(value_cpu[0].item())  # tanh-activated output in [-1,1] range
             
             # Cache results
-            self._put_in_cache(root.state_hash, policy_np, value_logit)
+            self._put_in_cache(root.state_hash, policy_np, value_signed)
             self._expand_node_from_policy(root, policy_np, board_size, action_size)
 
     def _check_algorithm_termination(self, root: MCTSNode, verbose: int) -> Optional[AlgorithmTerminationInfo]:
@@ -686,7 +685,7 @@ class BaselineMCTS:
         Returns:
           encodings          – tensors for unique, uncached leaves (order aligned with need_eval_idxs)
           need_eval_idxs     – indices into `leaves` for those unique encodings
-          cached_expansions  – (leaf_idx, policy_np, value_logit) for cache hits
+          cached_expansions  – (leaf_idx, policy_np, value_signed) for cache hits
         """
         encodings: List[torch.Tensor] = []
         need_eval_idxs: List[int] = []
@@ -709,8 +708,8 @@ class BaselineMCTS:
 
             if cached is not None:
                 self.cache_hits += 1
-                policy_np, value_logit = cached
-                cached_expansions.append((i, policy_np, value_logit))
+                policy_np, value_signed = cached
+                cached_expansions.append((i, policy_np, value_signed))
                 continue
 
             # Uncached → only encode the first occurrence of this state in the batch
@@ -749,10 +748,10 @@ class BaselineMCTS:
         for j, leaf_idx in enumerate(need_eval_idxs):
             leaf = leaves[leaf_idx]
             pol = policy_cpu[j].numpy()
-            val_logit = float(value_cpu[j].item())
+            val_signed = float(value_cpu[j].item())  # tanh-activated output in [-1,1] range
             
             # Cache CPU-native results
-            self._put_in_cache(leaf.state_hash, pol, val_logit)
+            self._put_in_cache(leaf.state_hash, pol, val_signed)
             self._expand_node_from_policy(leaf, pol, board_size, action_size)
 
     def _expand_cached_leaves(self, cached_expansions: List[Tuple[int, np.ndarray, float]], 
@@ -765,7 +764,7 @@ class BaselineMCTS:
         board_size = int(leaves[0].state.get_board_tensor().shape[-1])
         action_size = board_size * board_size
         
-        for (leaf_idx, policy_np, value_logit) in cached_expansions:
+        for (leaf_idx, policy_np, _) in cached_expansions:
             leaf = leaves[leaf_idx]
             if not leaf.is_expanded and not leaf.is_terminal:
                 self._expand_node_from_policy(leaf, policy_np, board_size, action_size)
@@ -779,46 +778,48 @@ class BaselineMCTS:
         
         simulations_completed = 0
         for leaf, path in zip(leaves, paths):
-            # Determine value for this leaf
+            # Determine signed value for this leaf
             if leaf.is_terminal:
-                p_red = self._get_terminal_value(leaf)
+                v_red_signed = self._get_terminal_value(leaf)
             else:
-                p_red = self._get_neural_network_value(leaf)
+                v_red_signed = self._get_neural_network_value(leaf)
             
-            # Backpropagate along path
-            self._backpropagate_path(path, p_red)
+            # Backpropagate signed value along path
+            self._backpropagate_path(path, v_red_signed)
             simulations_completed += 1
         
         timing_tracker.end_timing("backprop")
         return simulations_completed
 
     def _get_terminal_value(self, leaf: MCTSNode) -> float:
-        """Get value for a terminal leaf."""
-        if leaf.winner_str is None:
-            return 0.5  # Draw
-        
-        p_red = 1.0 if leaf.winner_str == "red" else 0.0
-    
-        return p_red
+        """Get signed value for a terminal leaf in [-1,1] range."""
+        if leaf.winner_str == "red":
+            return 1.0  # +1 = certain Red win
+        elif leaf.winner_str == "blue":
+            return -1.0  # -1 = certain Blue win
+        else:
+            raise ValueError(f"Invalid winner_str for terminal Hex node: {leaf.winner_str!r} (draws are not possible in Hex)")
 
     def _get_neural_network_value(self, leaf: MCTSNode) -> float:
-        """Get value for a non-terminal leaf from neural network."""
+        """Get signed value for a non-terminal leaf from neural network in [-1,1] range."""
         cached = self._get_from_cache(leaf.state_hash)
         if cached is None:
             raise RuntimeError(f"Leaf state not found in cache: {leaf.state_hash}")
-        _, value_logit = cached
-        # Convert from [-1, 1] range to [0, 1] probability using centralized utility
-        return float(ValuePredictor.model_output_to_probability(value_logit))
+        _, value_signed = cached
+        # Return signed value directly - tanh activation gives values in [-1,1] range
+        return float(value_signed)
 
-    def _backpropagate_path(self, path: List[Tuple[MCTSNode, int]], p_red: float):
-        """Backpropagate value along a path from leaf to root."""
+    def _backpropagate_path(self, path: List[Tuple[MCTSNode, int]], v_red_signed: float):
+        """Backpropagate signed value along a path from leaf to root."""
         for (node, a_idx) in reversed(path):
-            v_node = p_red if node.to_play == Player.RED else (1.0 - p_red)
+            # Convert Red's signed value to player-to-move perspective
+            v_node = signed_for_player_to_move(v_red_signed, node.to_play)
             
-            # Apply depth discounting
+            # Apply depth discounting in signed space (shrink toward 0)
             if self.cfg.enable_depth_discounting and node.depth > 0:
-                discount = self.cfg.depth_discount_factor ** node.depth
-                v_node *= discount
+                # TODO(step: tune): consider using distance_to_leaf instead of absolute depth
+                # TODO(step: tune): retune depth_discount_factor for signed space
+                v_node = apply_depth_discount_signed(v_node, self.cfg.depth_discount_factor, node.depth)
             
             node.N[a_idx] += 1
             node.W[a_idx] += v_node
@@ -967,18 +968,13 @@ class BaselineMCTS:
         }
 
     def _compute_win_probability(self, root: MCTSNode, root_state: HexGameState) -> float:
-        """Compute win probability for the current player based on root value."""
+        """Compute win probability for the current player based on root value (edge conversion)."""
         tree_data = self._compute_tree_data(root)
         root_value = tree_data["root_value"]
         
-        # Convert root value to win probability
-        # Root value is from the perspective of the player to move
-        # For RED player: root_value is probability RED wins
-        # For BLUE player: root_value is probability BLUE wins
-        if root_state.current_player_enum == Player.RED:
-            return root_value
-        else:
-            return 1.0 - root_value
+        # Convert signed value to probability only at the edge (for external API)
+        # Root value is now signed from player-to-move perspective in [-1,1]
+        return to_prob(root_value)
 
     def _extract_principal_variation(self, root: MCTSNode, max_length: int = 10) -> List[Tuple[int, int]]:
         """Extract the principal variation (best move sequence) from the MCTS tree."""
@@ -1094,6 +1090,7 @@ class BaselineMCTS:
         """Return index into node.legal_moves of the action maximizing PUCT score."""
         # PUCT: U = c_puct * P * sqrt(sum(N)) / (1 + N)
         # score = Q + U
+        # TODO(step: tune): retune c_puct for signed Q values in [-1,1] range
         
         # Detect terminal moves if enabled and appropriate
         if self.cfg.enable_terminal_move_detection:
@@ -1175,7 +1172,7 @@ class BaselineMCTS:
         Args:
             board_key: The board state key
             policy: The policy logits
-            value: The value logit
+            value: The signed value (tanh-activated output in [-1,1] range)
         """
         # If key already exists, remove it first (will be re-added at end)
         if board_key in self.eval_cache:
diff --git a/hex_ai/models.py b/hex_ai/models.py
index eaa6215..2e664c8 100644
--- a/hex_ai/models.py
+++ b/hex_ai/models.py
@@ -185,9 +185,9 @@ class TwoHeadedResNet(nn.Module):
             x: Input tensor of shape (batch_size, 3, 13, 13)
             
         Returns:
-            Tuple of (policy_logits, value_logit):
+            Tuple of (policy_logits, value_signed):
             - policy_logits: Shape (batch_size, 169)
-            - value_logit: Shape (batch_size, 1) - Raw logit for Red's win probability
+            - value_signed: Shape (batch_size, 1) - Signed value in [-1,1] range (tanh-activated)
         """
         # Shared trunk
         features = self.forward_shared(x)
@@ -209,9 +209,9 @@ class TwoHeadedResNet(nn.Module):
             value_features = self.global_pool(features)
             value_features = value_features.view(value_features.size(0), -1)
         
-        value_logit = torch.tanh(self.value_head(value_features))  # (batch_size, 1)
+        value_signed = torch.tanh(self.value_head(value_features))  # (batch_size, 1)
         
-        return policy_logits, value_logit
+        return policy_logits, value_signed
 
     @torch.no_grad()
     def forward_value_only(self, x: torch.Tensor) -> torch.Tensor:
@@ -289,6 +289,6 @@ Architecture:
 
 Output:
 - Policy Logits: (batch_size, 169)
-- Value Logit: (batch_size, 1) with tanh activation
+- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
 """
     return summary 
\ No newline at end of file
diff --git a/hex_ai/value_utils.py b/hex_ai/value_utils.py
index 0b4a43c..474724b 100644
--- a/hex_ai/value_utils.py
+++ b/hex_ai/value_utils.py
@@ -640,6 +640,130 @@ def select_policy_move(state, model, temperature: float = 1.0) -> Tuple[int, int
         chosen_idx = np.random.choice(len(legal_moves), p=legal_policy)
         return legal_moves[chosen_idx] 
 
+# =============================
+# Value-Semantics Abstraction (Step 1: Contract Establishment)
+# =============================
+
+def to_prob(v_signed: float) -> float:
+    """
+    Map a signed value v ∈ [-1,1] to probability p ∈ [0,1], with 0.5 == neutral.
+    
+    Args:
+        v_signed: Signed value in [-1, 1] range where +1 = certain Red win, -1 = certain Blue win
+        
+    Returns:
+        Probability in [0, 1] range where 0.5 = neutral
+    """
+    return 0.5 * (v_signed + 1.0)
+
+def to_signed(p_prob: float) -> float:
+    """
+    Map probability p ∈ [0,1] to signed value v ∈ [-1,1], with 0 == neutral.
+    
+    Args:
+        p_prob: Probability in [0, 1] range where 0.5 = neutral
+        
+    Returns:
+        Signed value in [-1, 1] range where +1 = certain Red win, -1 = certain Blue win
+    """
+    return 2.0 * p_prob - 1.0
+
+def signed_for_player_to_move(v_red_signed: float, player) -> float:
+    """
+    Given Red's signed value v_red ∈ [-1,1], return value from 'player-to-move' perspective.
+    If RED to move: return v_red; if BLUE to move: return -v_red.
+    
+    Args:
+        v_red_signed: Red's signed value in [-1, 1] range
+        player: Player enum (RED or BLUE)
+        
+    Returns:
+        Signed value from player-to-move perspective in [-1, 1] range
+    """
+    return v_red_signed if player == Player.RED else -v_red_signed
+
+def prob_for_player_to_move(p_red: float, player) -> float:
+    """
+    Given Red's win probability p_red ∈ [0,1], return probability from 'player-to-move' perspective.
+    If RED to move: p_red; else: 1 - p_red.
+    
+    Args:
+        p_red: Red's win probability in [0, 1] range
+        player: Player enum (RED or BLUE)
+        
+    Returns:
+        Win probability from player-to-move perspective in [0, 1] range
+    """
+    return p_red if player == Player.RED else (1.0 - p_red)
+
+def apply_depth_discount_signed(v: float, gamma: float, dist: int) -> float:
+    """
+    Signed-space discount: shrink v ∈ [-1,1] toward 0 (neutral) by gamma^dist.
+    
+    Args:
+        v: A signed value (either Red-centric or already flipped to player-to-move)
+        gamma: Discount factor in (0,1]; gamma=1 keeps v unchanged; smaller gamma shrinks faster
+        dist: Non-negative integer 'distance' along the path (prefer distance-to-leaf over absolute depth
+              if the intent is to prefer shorter wins/losses during backup)
+        
+    Returns:
+        (gamma ** dist) * v (still in [-1,1])
+    """
+    return (gamma ** max(0, int(dist))) * v
+
+def apply_depth_discount_toward_neutral_prob(p: float, gamma: float, dist: int) -> float:
+    """
+    Probability-space analog: shrink p ∈ [0,1] toward 0.5 by gamma^dist.
+    p' = 0.5 + (gamma ** dist) * (p - 0.5)
+    
+    Args:
+        p: Probability in [0, 1] range
+        gamma: Discount factor in (0,1]
+        dist: Non-negative integer distance
+        
+    Returns:
+        Discounted probability in [0, 1] range, shrunk toward 0.5
+    """
+    return 0.5 + (gamma ** max(0, int(dist))) * (p - 0.5)
+
+def distance_to_leaf(current_depth: int, leaf_depth: int) -> int:
+    """
+    Prefer this over absolute node depth when discounting backups from a single simulation:
+    dist = leaf_depth - current_depth (>= 0).
+    This aligns with "prefer shorter wins" without penalizing the static position of a node in the tree.
+    
+    Args:
+        current_depth: Current node's depth in the tree
+        leaf_depth: Leaf node's depth in the tree
+        
+    Returns:
+        Distance from current node to leaf (>= 0)
+    """
+    return max(0, int(leaf_depth) - int(current_depth))
+
+def backprop_value_current_behavior(p_red_leaf: float, player_to_move, gamma: float, depth_or_dist: int) -> float:
+    """
+    Illustrative only: replicate current behavior exactly (probability space),
+    but route through the abstraction so we can flip to signed in a later step.
+    
+    Args:
+        p_red_leaf: Red's win probability from leaf evaluation
+        player_to_move: Player whose turn it is to move
+        gamma: Discount factor
+        depth_or_dist: Depth or distance for discounting
+        
+    Returns:
+        Value for backpropagation (current probability-space behavior)
+    """
+    # Current code uses probability space for v_node:
+    v_node_prob = prob_for_player_to_move(p_red_leaf, player_to_move)
+    # If discounting is enabled, keep current behavior or switch to neutral-centered variant later:
+    #   - If the existing code multiplies by gamma^depth directly on probabilities, mirror that here.
+    #   - Otherwise, prefer the neutral-centered version below (to be enabled in a later step).
+    # Neutral-centered (recommended for later):
+    # v_node_prob = apply_depth_discount_toward_neutral_prob(v_node_prob, gamma, depth_or_dist)
+    return v_node_prob
+
 # =============================
 # Move Application Utilities
 # =============================
diff --git a/hex_ai/web/static/app.js b/hex_ai/web/static/app.js
index 6a1ccf6..ec54713 100644
--- a/hex_ai/web/static/app.js
+++ b/hex_ai/web/static/app.js
@@ -90,7 +90,7 @@ let state = {
   last_move: null,
   last_move_player: null, // Track which player made the last move
   blue_model_id: 'model1',
-  red_model_id: 'model1',
+  red_model_id: 'model2',  // Use different model for red by default
   blue_temperature: 0.2,
   red_temperature: 0.2,
   // MCTS settings
@@ -706,9 +706,28 @@ document.addEventListener('DOMContentLoaded', async () => {
       redSelect.appendChild(option);
     });
     
-    // Set default selections
-    blueSelect.value = state.blue_model_id;
-    redSelect.value = state.red_model_id;
+    // Set default selections - check if the default values exist in the dropdown
+    if (state.available_models.some(model => model.id === state.blue_model_id)) {
+      blueSelect.value = state.blue_model_id;
+    } else if (state.available_models.length > 0) {
+      // Fallback to first available model
+      state.blue_model_id = state.available_models[0].id;
+      blueSelect.value = state.blue_model_id;
+      console.log(`Blue model not found, using: ${state.blue_model_id}`);
+    }
+    
+    if (state.available_models.some(model => model.id === state.red_model_id)) {
+      redSelect.value = state.red_model_id;
+    } else if (state.available_models.length > 0) {
+      // Fallback to first available model if red model not found
+      const redFallbackIndex = Math.min(1, state.available_models.length - 1);
+      state.red_model_id = state.available_models[redFallbackIndex].id;
+      redSelect.value = state.red_model_id;
+      console.log(`Red model not found, using: ${state.red_model_id}`);
+    }
+    
+    console.log(`Model dropdowns initialized. Available models: ${state.available_models.map(m => m.id).join(', ')}`);
+    console.log(`Selected models - Blue: ${state.blue_model_id}, Red: ${state.red_model_id}`);
   } catch (err) {
     console.error('Failed to load models:', err);
   }
diff --git a/scripts/run_deterministic_tournament.py b/scripts/run_deterministic_tournament.py
index 7a760e0..5c3da99 100755
--- a/scripts/run_deterministic_tournament.py
+++ b/scripts/run_deterministic_tournament.py
@@ -608,7 +608,7 @@ def run_deterministic_tournament(
         game_results = []
         for opening_idx, opening in enumerate(openings):
             if verbose >= 1:
-                if opening_idx == 1:
+                if opening_idx == 0:
                     print(f"  Opening {opening_idx + 1}/{len(openings)}", end="", flush=True)
                 else:
                     print(",", opening_idx + 1, end="", flush=True)
```