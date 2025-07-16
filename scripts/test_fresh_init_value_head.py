import torch
import numpy as np
from hex_ai.models import create_model
from hex_ai.utils import format_conversion as fc

if __name__ == "__main__":
    # Create a freshly initialized model (no checkpoint loaded)
    model = create_model("resnet18")
    model.eval()

    # Prepare an empty board (all zeros, Blue to move)
    board_size = fc.BOARD_SIZE
    empty_board = np.zeros((board_size, board_size), dtype=np.int8)
    input_tensor = fc.board_nxn_to_3nxn(empty_board)  # shape (3, N, N)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # add batch dim

    # Run inference
    with torch.no_grad():
        policy_logits, value_logit = model(input_tensor)
        value_logit = value_logit[0, 0].item()  # scalar
        value_prob = torch.sigmoid(torch.tensor(value_logit)).item()

    print(f"Raw value logit: {value_logit}")
    print(f"Sigmoid(value_logit): {value_prob:.4f} (should be close to 0.5 if zero-initialized)") 