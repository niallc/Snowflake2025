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
    parser.add_argument('--legacy-model', action='store_true', help='Use legacy 2-channel model for old checkpoints')
    args = parser.parse_args()

    model_path = f"{args.model_dir.rstrip('/')}/{args.model_file}"
    print(f"model_path = {model_path}")
    print("Using current model architecture")
    from hex_ai.models import TwoHeadedResNet
    # model = TwoHeadedResNet()  # Redundant, not used
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
    # Symptom if you fail to do this, messages like:
    # ModuleNotFoundError: No module named 'torch'

    # To run this script from the command line, run:
    """
    PYTHONPATH=. python scripts/simple_inference_cli.py \
        --trmph https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7 \
        --model_dir checkpoints/hyperparameter_tuning/sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.01_value_weight_decay_factor25.0_20250719_052409/ \
        --model_file best_model.pt \
        --device mps
        
    """
    # Older way
    # python3 -m scripts.simple_inference_cli \    
    
    main() 