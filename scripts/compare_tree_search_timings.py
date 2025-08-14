#!/usr/bin/env python3
import argparse
import json
import random
import time
import numpy as np

from hex_ai.inference.model_config import get_model_path
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import (
    minimax_policy_value_search_with_batching,
)
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.batched_mcts import BatchedNeuralMCTS
from hex_ai.inference.mcts_config import BatchedMCTSOrchestration


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare NN timing between fixed-tree batching and batched MCTS"
    )
    p.add_argument("--model", default="current_best", help="Model to use (current_best|previous_best)")
    p.add_argument("--device", default=None, help="cpu|mps|cuda (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-cache", action="store_true", default=True)
    p.add_argument("--fixed-tree-widths", default="13,8")
    p.add_argument("--fixed-tree-batch-size", type=int, default=64)
    p.add_argument("--mcts-sims", type=int, default=120)
    p.add_argument("--mcts-optimal-batch-size", type=int, default=64)
    p.add_argument("--mcts-wait-ms", type=int, default=200)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--output", choices=["table", "json"], default="table")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def summarize_batches(batch_records):
    if not batch_records:
        return {
            "batches": 0,
            "boards": 0,
            "avg_batch": 0.0,
            "predict_ms_total": 0.0,
            "predict_ms_per_sample_median": 0.0,
            "predict_ms_median": 0.0,
            "stack_ms_total": 0.0,
            "post_ms_total": 0.0,
        }
    sizes = [b["size"] for b in batch_records]
    predict_ms = [b["predict_ms"] for b in batch_records]
    stack_ms = [b["stack_ms"] for b in batch_records]
    post_ms = [b["post_ms"] for b in batch_records]
    per_sample = []
    for b in batch_records:
        if b["size"] > 0:
            per_sample.append(b["predict_ms"] / b["size"])
    return {
        "batches": len(batch_records),
        "boards": int(sum(sizes)),
        "avg_batch": float(np.mean(sizes)),
        "predict_ms_total": float(sum(predict_ms)),
        "predict_ms_per_sample_median": float(np.median(per_sample) if per_sample else 0.0),
        "predict_ms_median": float(np.median(predict_ms)),
        "stack_ms_total": float(sum(stack_ms)),
        "post_ms_total": float(sum(post_ms)),
    }


def run_fixed_tree(model: SimpleModelInference, widths, batch_size: int):
    # Reset stats storage to isolate this trial
    model.reset_stats()
    # Build and run batched policy waves + leaf values
    state = HexGameState()
    _ = minimax_policy_value_search_with_batching(
        state=state,
        model=model,
        widths=widths,
        batch_size=batch_size,
        verbose=1,
        return_tree=False,
    )
    # Summarize per-batch stats gathered by SimpleModelInference
    s = summarize_batches(model.stats.get("batch_records", []))
    s["conv_ms_total"] = float(model.stats.get("conv_ms_total", 0.0))
    return s


def run_mcts(model: SimpleModelInference, sims: int, optimal_batch_size: int, wait_ms: int):
    # Reset stats storage to isolate this trial
    model.reset_stats()
    orch = BatchedMCTSOrchestration()
    # Use default configuration from mcts_config.py
    # Only override if explicitly provided
    if optimal_batch_size != 64:
        orch.optimal_batch_size = optimal_batch_size
    if wait_ms != 200:  # Default from mcts_config.py
        orch.evaluator_max_wait_ms = wait_ms

    mcts = BatchedNeuralMCTS(
        model=model,
        optimal_batch_size=optimal_batch_size,
        verbose=1,
        selection_wait_ms=200,
        orchestration=orch,
    )
    state = HexGameState()
    _ = mcts.search(state, num_simulations=sims)
    # Pull NN-level batch stats from the same source as fixed-tree (SimpleModelInference)
    s = summarize_batches(model.stats.get("batch_records", []))
    s["conv_ms_total"] = float(model.stats.get("conv_ms_total", 0.0))
    # Also expose MCTS-side aggregation
    stats = mcts.get_search_statistics()
    s["mcts_boards"] = int(stats.get("total_inferences", 0))
    s["mcts_batches"] = int(stats.get("total_batches_processed", 0))
    s["mcts_avg_batch"] = float(stats.get("average_batch_size", 0.0))
    return s


def main():
    args = parse_args()
    seed_everything(args.seed)
    model_path = get_model_path(args.model)

    model = SimpleModelInference(
        checkpoint_path=model_path,
        device=args.device,
        enable_caching=not args.disable_cache,
        max_batch_size=max(args.fixed_tree_batch_size, args.mcts_optimal_batch_size),
    )

    widths = [int(x) for x in args.fixed_tree_widths.split(",") if x]

    fixed_runs = []
    mcts_runs = []
    for _ in range(args.repeats):
        fr = run_fixed_tree(model, widths, args.fixed_tree_batch_size)
        fixed_runs.append(fr)
        mr = run_mcts(model, args.mcts_sims, args.mcts_optimal_batch_size, args.mcts_wait_ms)
        mcts_runs.append(mr)

    def agg(runs):
        if not runs:
            return {}
        # Aggregate by median for robustness
        keys = runs[0].keys()
        out = {}
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            if not vals:
                continue
            out[k] = float(np.median(vals)) if isinstance(vals[0], (int, float)) else vals[0]
        return out

    fixed_med = agg(fixed_runs)
    mcts_med = agg(mcts_runs)

    result = {
        "device": str(model.device),
        "model": args.model,
        "fixed_tree": fixed_med,
        "mcts": mcts_med,
    }

    if args.output == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    # Human-readable table
    print("\n=== Compare Tree Search NN Timings ===")
    print(f"Device: {result['device']}  Model: {result['model']}")
    print("-- Fixed Tree --")
    print(
        f"boards={int(fixed_med.get('boards',0))} batches={int(fixed_med.get('batches',0))} avg_batch={fixed_med.get('avg_batch',0):.2f} "
        f"predict_total_ms={fixed_med.get('predict_ms_total',0):.1f} per_sample_median_ms={fixed_med.get('predict_ms_per_sample_median',0):.3f} "
        f"conv_ms_total={fixed_med.get('conv_ms_total',0):.1f} stack_ms_total={fixed_med.get('stack_ms_total',0):.1f} post_ms_total={fixed_med.get('post_ms_total',0):.1f}"
    )
    print("-- MCTS --")
    print(
        f"boards={int(mcts_med.get('boards',0))} batches={int(mcts_med.get('batches',0))} avg_batch={mcts_med.get('avg_batch',0):.2f} "
        f"predict_total_ms={mcts_med.get('predict_ms_total',0):.1f} per_sample_median_ms={mcts_med.get('predict_ms_per_sample_median',0):.3f} "
        f"conv_ms_total={mcts_med.get('conv_ms_total',0):.1f} stack_ms_total={mcts_med.get('stack_ms_total',0):.1f} post_ms_total={mcts_med.get('post_ms_total',0):.1f} "
        f"(mcts_agg boards={int(mcts_med.get('mcts_boards',0))} batches={int(mcts_med.get('mcts_batches',0))} avg_batch={mcts_med.get('mcts_avg_batch',0):.2f})"
    )
    # Quick diff highlight
    ft = fixed_med.get('predict_ms_per_sample_median', 0.0)
    mc = mcts_med.get('predict_ms_per_sample_median', 0.0)
    if ft > 0 and mc > 0:
        ratio = mc / ft
        print(f"Per-sample forward median ratio (MCTS / FixedTree): {ratio:.2f}x")
    print("========================================\n")


if __name__ == "__main__":
    main()


