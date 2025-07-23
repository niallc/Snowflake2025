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
    # To find recent results, run:
    ls -lth checkpoints/hyperparameter_tuning | head -n 3 | less

    resCollDir="checkpoints/hyperparameter_tuning/"
    resDirTa1g="shuffled_sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.2_value_weight_decay_factor3.0_20250721_064254/"
    resDirTag2="shuffled_sweep_run_1_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.2_value_weight_decay_factor50.0_20250721_064254/"
    resDirTag3="loss_weight_sweep_exp0_pw0.01_475646_20250721_131001/"
    resDirTag4="loss_weight_sweep_exp0_pw0.001_57e0af_20250721_150933/"
    resDirTag5="loss_weight_sweep_exp1_pw0.0001_6c84d6_20250721_150933/"
    resDirTag6="loss_weight_sweep_exp2_pw0.7_c0cb27_20250721_150933/"
    resDirTag7="loss_weight_sweep_exp0_do0_pw0.2_794e88_20250722_211936"
    resDir=${resCollDir}${resDirTag7}
    blueFinal="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7"
    redFinal="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2g7"
    blueWin="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7"
    redWin="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2"
    realGameB="http://www.trmph.com/hex/board#13,a6i2d10d9f8e9g9g10i9h9i8h8i7j4g6g7f7h6g8f10h7i10j10j11h10g4e5e4f4g12i11g2h2g3h4h13g11f12f11e12l11k12h12g13l12k10i12h3j3i4i3h5g5j2k2j12i13h11j9f3d5k1l1"
    realGameR="http://www.trmph.com/hex/board#13,a12g5f5f6j4j5h6i4i5h7i7i6g7f9g8g10i9k2h4h8g9f10h9i10h10g12g11f12f11e12e11d12h11h12i11i12j11j12d11c12a13b11a11b10a10b9a9b8a8b7a7b6c11b12a6b5a5b3b4c3c4d3d4e3e4f3g4h2i3j1l2k3i2i1l3k4j2k1l4k6l5l7l6k7l11k12l12k13l13e5f4g2k9l10m7m8m6l8k10e6j3m1j6k5a3a4"
    earlyGameR="https://trmph.com/hex/board#13,a2f8g7g8h7h8i9j7e7d9b10c8"
    
    modelFile="epoch1_mini100.pt" # or best_model.pt
    boardPos=${blueWin}
    
    PYTHONPATH=. python scripts/simple_inference_cli.py \
    --trmph ${boardPos} \
    --model_dir ${resDir} \
    --model_file ${modelFile} \
    --device mps
        
    """
    # Older way
    # python3 -m scripts.simple_inference_cli \    
    
    main() 