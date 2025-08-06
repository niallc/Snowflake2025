import subprocess
import sys
import os
import re
import pytest

# Adjust these paths as needed for your environment
from hex_ai.inference.model_config import get_model_dir
MODEL_DIR = get_model_dir("previous_best")
MODEL_FILE = "epoch1_mini30.pt"
SCRIPT_PATH = os.path.join("scripts", "simple_inference_cli.py")

# Known blue win and red win positions (from CLI docstring)
BLUE_WIN_TRMPH = "https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7"
RED_WIN_TRMPH = "https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2g7"

@pytest.mark.parametrize("trmph_url,expected_min,expected_max,desc", [
    (BLUE_WIN_TRMPH, 85.0, 100.0, "Blue win should yield high probability for blue"),
    (RED_WIN_TRMPH, 0.0, 15.0, "Red win should yield low probability for blue"),
])
def test_simple_inference_cli_value(trmph_url, expected_min, expected_max, desc):
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        pytest.skip(f"Model file {model_path} does not exist. Adjust MODEL_DIR and MODEL_FILE in the test.")
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--trmph", trmph_url,
        "--model_dir", MODEL_DIR,
        "--model_file", MODEL_FILE,
        "--device", "cpu"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    # Parse the output for the value estimate (new format)
    match = re.search(r"Value head \(probability Blue wins\): ([0-9.]+)", result.stdout)
    assert match, f"Could not find value estimate in output: {result.stdout}"
    value = float(match.group(1)) * 100.0
    if not (expected_min <= value <= expected_max):
        print("\n==== CLI OUTPUT FOR DEBUGGING ====")
        print(result.stdout)
        print("==== END CLI OUTPUT ====")
    assert expected_min <= value <= expected_max, f"{desc}: got {value:.2f}% (expected between {expected_min} and {expected_max})\nFull output:\n{result.stdout}" 