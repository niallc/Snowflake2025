# Tournament Efficiency Improvement Plan

## Background

The current tournament system for Hex AI (see `hex_ai/inference/tournament.py` and `hex_ai/inference/fixed_tree_search.py`) allows for model-vs-model evaluation using either direct policy+value move selection or a fixed-width, fixed-depth minimax tree search (with optional alpha-beta pruning). Model inference is performed using `SimpleModelInference` from `hex_ai/inference/simple_model_inference.py`.

- **Direct move selection**: Uses the model's policy and value heads to pick a move, no search.
- **Tree search**: Uses `minimax_policy_value_search` (in `fixed_tree_search.py`) to explore a search tree of configurable width and depth, evaluating leaf positions in batches.

## Current Inefficiencies

- **Policy evaluation for child nodes is not batched**: When expanding the first level of the search tree, the code loops over each child and calls the model separately for each, rather than batching all child boards in a single call.
- **Leaf value evaluation is batched** (good): All leaf boards are evaluated in a single batch (or in batches of up to `batch_size`).
- **Deeper search would exacerbate the inefficiency**: As search depth increases, the number of policy calls grows rapidly if not batched.

## Relevant Classes, Functions, and Files

- **Tournament orchestration**: `hex_ai/inference/tournament.py`
  - `TournamentConfig`, `TournamentPlayConfig`, `run_round_robin_tournament`, `play_single_game`
- **Tree search logic**: `hex_ai/inference/fixed_tree_search.py`
  - `minimax_policy_value_search`
    - `get_topk_moves(state, k)` (calls model policy)
    - `minimax(...)` (recursive search)
    - Leaf value evaluation (batched)
- **Model inference**: `hex_ai/inference/simple_model_inference.py`
  - `SimpleModelInference.infer` (supports batch input)

## Step-by-Step Refactor Plan

1. **Refactor `get_topk_moves` to support batch policy evaluation**
   - Instead of calling `model.infer` for each child, collect all child boards and call `model.infer` in a batch.
   - This may require modifying `SimpleModelInference.infer` to always accept a list/array of boards and return a batch of policy/value outputs.

2. **Update the minimax recursion to batch at each level**
   - At each depth, collect all boards to be evaluated for policy, and call the model in a batch.
   - This will require restructuring the recursion to operate on lists of states at each level, rather than one at a time.

3. **Maintain batching for leaf value evaluation**
   - The current code already batches leaf value calls; keep this logic.

4. **Test with various `search_widths` and batch sizes**
   - Ensure that the new batching logic produces the same results as the old code, but with fewer network calls and improved speed.

5. **Update tournament logging to record batch sizes and timing**
   - For benchmarking, log the number of network calls and average batch size per game.

## Impact

- **Speed**: Batching will significantly reduce the number of network calls, especially for wide/deep searches, and will better utilize GPU/CPU resources.
- **Reproducibility**: Results should remain deterministic for a given seed/config, but timing and resource usage will improve.
- **Scalability**: Enables deeper/wider search and larger tournaments without prohibitive slowdowns.

## Open Questions / Design Tradeoffs

- **Memory usage**: Large batches may require more GPU memory; need to balance batch size with available resources.
- **API design**: Should `SimpleModelInference.infer` always accept batches, or should there be a separate batch method?
- **Parallelization**: In the future, could further parallelize across games or search trees.

## Future Extensions

- **Deeper search**: With efficient batching, can explore deeper trees for stronger play.
- **Parallel tournaments/self-play**: Run multiple games in parallel for data generation or large-scale evaluation.
- **Dynamic search width/depth**: Adapt search parameters based on position or model confidence.

## Pointers to Code

- `hex_ai/inference/fixed_tree_search.py`: Main place to refactor for batching.
  - See `minimax_policy_value_search` and `get_topk_moves`.
- `hex_ai/inference/simple_model_inference.py`: Update `infer` to support batch input if needed.
- `hex_ai/inference/tournament.py`: For passing search config and logging batch/timing info.

---

**This plan provides a roadmap for making tournament and search evaluation much more efficient, scalable, and ready for future extensions.** 