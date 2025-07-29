#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.config import TRAINING_BLUE_WIN, TRAINING_RED_WIN, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.value_utils import ValuePerspective, get_win_prob_from_model_output, Winner, get_policy_probs_from_logits
from hex_ai.models import TwoHeadedResNet

def main():
    parser = argparse.ArgumentParser(description="Hex AI Simple Inference CLI")
    parser.add_argument('--trmph', type=str, required=True, help='TRMPH string or link describing the board')
    parser.add_argument('--model_dir', type=str, default="checkpoints/hex_ai_MainTraining_15M_samples_20250715_005413/bs_512_wd_5e-4_policy_0.2_value_0.8/", help='Directory containing the model checkpoint')
    parser.add_argument('--model_file', type=str, default="best_model.pt", help='Model checkpoint file name')
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
    policy_logits, value_logit = infer.infer(args.trmph)
    # Use the updated method that accepts logits directly
    top_moves = infer.get_top_k_moves(policy_logits, k=args.topk)
    print(f"Top {args.topk} moves (trmph format, probability):")
    for move, prob in top_moves:
        print(f"  {move}: {prob:.3f}")
    print(f"Value head (logit): {value_logit:.3f}")
    blue_prob = get_win_prob_from_model_output(value_logit, Winner.BLUE)
    red_prob = get_win_prob_from_model_output(value_logit, Winner.RED)
    print(f"Value head (probability Blue wins): {blue_prob:.3f}")
    print(f"Value head (probability Red wins): {red_prob:.3f}")

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
    resDirTag1="loss_weight_sweep_exp0_lr0.001_do0_pw0.2_17ca83_20250724_143544/"
    resDirTag2="loss_weight_sweep_exp0_bs256_98f719_20250724_233408/"
    resDir=${resCollDir}${resDirTag2}

    blueFinal="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7"
    redFinal="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2g7"
    blueWin="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7"
    redWin="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2"
    realGameB="http://www.trmph.com/hex/board#13,a6i2d10d9f8e9g9g10i9h9i8h8i7j4g6g7f7h6g8f10h7i10j10j11h10g4e5e4f4g12i11g2h2g3h4h13g11f12f11e12l11k12h12g13l12k10i12h3j3i4i3h5g5j2k2j12i13h11j9f3d5k1l1"
    realGameR="http://www.trmph.com/hex/board#13,a12g5f5f6j4j5h6i4i5h7i7i6g7f9g8g10i9k2h4h8g9f10h9i10h10g12g11f12f11e12e11d12h11h12i11i12j11j12d11c12a13b11a11b10a10b9a9b8a8b7a7b6c11b12a6b5a5b3b4c3c4d3d4e3e4f3g4h2i3j1l2k3i2i1l3k4j2k1l4k6l5l7l6k7l11k12l12k13l13e5f4g2k9l10m7m8m6l8k10e6j3m1j6k5a3a4"
    realGame5B="http://www.trmph.com/hex/board#13,a6i2d10d9f8e9g9g10i9h9i8h8i7j4g6g7f7h6g8f10h7i10j10j11h10g4e5e4f4g12i11g2h2g3h4h13g11f12f11e12l11k12h12g13l12k10i12h3j3i4i3h5g5j2k2j12i13h11"
    realGame5R="http://www.trmph.com/hex/board#13,a12g5f5f6j4j5h6i4i5h7i7i6g7f9g8g10i9k2h4h8g9f10h9i10h10g12g11f12f11e12e11d12h11h12i11i12j11j12d11c12a13b11a11b10a10b9a9b8a8b7a7b6c11b12a6b5a5b3b4c3c4d3d4e3e4f3g4h2i3j1l2k3i2i1l3k4j2k1l4k6l5l7l6k7l11k12l12k13l13e5f4g2k9l10m7m8m6l8k10e6j3"
    earlyGameR="https://trmph.com/hex/board#13,a2f8g7g8h7h8i9j7e7d9b10c8"
    earlyGameRMove="https://trmph.com/hex/board#13,a2f8g7g8h7h8i9j7e7d9b10c8c7"
        
    modelFile="epoch1_mini1.pt"
    boardPos=${redFinal}
    
    PYTHONPATH=. python scripts/simple_inference_cli.py \
    --trmph ${boardPos} \
    --model_dir ${resDir} \
    --model_file ${modelFile} \
    --device mps
        
    """
    # Older way
    # python3 -m scripts.simple_inference_cli \    
    
    main() 