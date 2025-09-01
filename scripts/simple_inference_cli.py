#!/usr/bin/env python3
"""
Simple inference CLI for Hex AI models.

NOTE: Value head terminology - We use 'value_signed' as a shorthand for [-1, 1] scores
returned by the value head (tanh activated) and used by MCTS, as opposed to 'value_logits'
which were the old sigmoid-based outputs.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.config import TRAINING_BLUE_WIN, TRAINING_RED_WIN, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.value_utils import ValuePerspective, ValuePredictor, Winner, get_policy_probs_from_logits
from hex_ai.models import TwoHeadedResNet

def main():
    parser = argparse.ArgumentParser(description="Hex AI Simple Inference CLI")
    parser.add_argument('--trmph', type=str, required=True, help='TRMPH string or link describing the board')
    parser.add_argument('--model_dir', type=str, default="checkpoints/hex_ai_MainTraining_15M_samples_20250715_005413/bs_512_wd_5e-4_policy_0.2_value_0.8/", help='Directory containing the model checkpoint')
    parser.add_argument('--model_file', type=str, default="best_model.pt.gz", help='Model checkpoint file name')
    parser.add_argument('--topk', type=int, default=3, help='Number of top policy moves to display')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')

    args = parser.parse_args()

    model_path = f"{args.model_dir.rstrip('/')}/{args.model_file}"
    print(f"model_path = {model_path}")
    print("Using current model architecture")
    # model = TwoHeadedResNet()  # Redundant, not used
    infer = SimpleModelInference(model_path, device=args.device)

    print("\n--- Board Position ---")
    infer.display_board(args.trmph)

    print("\n--- Model Predictions ---")
    policy_logits, value_signed = infer.simple_infer(args.trmph)
    # Use the updated method that accepts logits directly
    top_moves = infer.get_top_k_moves(policy_logits, k=args.topk)
    print(f"Top {args.topk} moves (trmph format, probability):")
    for move, prob in top_moves:
        print(f"  {move}: {prob:.3f}")
    print(f"Value head (signed): {value_signed:.3f}")
    blue_prob = ValuePredictor.get_win_probability(value_signed, Winner.BLUE)
    red_prob = ValuePredictor.get_win_probability(value_signed, Winner.RED)
    print(f"Value head (probability Blue wins): {blue_prob:.3f}")
    print(f"Value head (probability Red wins): {red_prob:.3f}")

if __name__ == "__main__":
    # source hex_ai_env/bin/activate
    # PYTHONPATH=. python scripts/simple_inference_cli.py \
    #   --trmph "https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7" \
    #   --model_dir "checkpoints/hyperparameter_tuning/pipeline_20250805_162626/pipeline_sweep_exp0__99914b_20250805_162626" \
    #   --model_file "epoch4_mini32.pt.gz" \
    #   --device mps    
    
    main() 