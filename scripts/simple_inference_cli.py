#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.inference.simple_model_inference import SimpleModelInference

def main():
    parser = argparse.ArgumentParser(description="Hex AI Simple Inference CLI")
    parser.add_argument('--trmph', type=str, required=True, help='TRMPH string or link describing the board')
    parser.add_argument('--model_dir', type=str, default="checkpoints/hex_ai_MainTraining_15M_samples_20250715_005413/bs_512_wd_5e-4_policy_0.2_value_0.8/", help='Directory containing the model checkpoint')
    parser.add_argument('--model_file', type=str, default="best_model.pt", help='Model checkpoint file name')
    parser.add_argument('--topk', type=int, default=3, help='Number of top policy moves to display')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    model_path = f"{args.model_dir.rstrip('/')}/{args.model_file}"
    infer = SimpleModelInference(model_path, device=args.device)

    print("\n--- Board Position ---")
    infer.display_board(args.trmph)

    print("\n--- Model Predictions ---")
    policy_probs, value, raw_value = infer.infer(args.trmph)
    top_moves = infer.get_top_k_moves(policy_probs, k=args.topk)
    print(f"Top {args.topk} moves (trmph format, probability):")
    for move, prob in top_moves:
        print(f"  {move}: {prob:.3f}")
    print(f"\nValue estimate (Probability Blue Wins): {value*100:.1f}%\n")
    print(f"Raw value logit: {raw_value}")

if __name__ == "__main__":
    # For command line execution, don't forget to run
    # source hex_ai_env/bin/activate
    main() 