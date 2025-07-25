import os
import sys
import time
import requests
import subprocess
import signal
import pytest

API_URL = "http://127.0.0.1:5001"

@pytest.fixture(scope="module", autouse=True)
def flask_server():
    # Start the Flask app in a subprocess with PYTHONPATH=.
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.Popen([sys.executable, "-m", "hex_ai.web.app"], env=env)
    # Wait for server to start
    for _ in range(30):
        try:
            requests.get(API_URL + "/")
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("Flask server did not start")
    yield
    proc.send_signal(signal.SIGINT)
    proc.wait()

def test_api_state_valid():
    # A minimal valid TRMPH string for an empty 13x13 board
    trmph = "#13,"
    resp = requests.post(API_URL + "/api/state", json={"trmph": trmph})
    assert resp.status_code == 200
    data = resp.json()
    assert "board" in data
    assert "player" in data
    assert "legal_moves" in data
    assert "policy" in data
    assert "value" in data
    assert "win_prob" in data
    assert data["winner"] is None
    assert data["trmph"] == trmph

def test_api_state_invalid():
    resp = requests.post(API_URL + "/api/state", json={"trmph": "not_a_trmph"})
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data

def test_api_move_valid():
    # Play a valid move on an empty board
    trmph = "#13,"
    move = "a1"
    resp = requests.post(API_URL + "/api/move", json={"trmph": trmph, "move": move})
    assert resp.status_code == 200
    data = resp.json()
    assert "new_trmph" in data
    assert "board" in data
    assert "player" in data
    assert "legal_moves" in data
    assert "policy" in data
    assert "value" in data
    assert "win_prob" in data
    assert data["winner"] is None or data["winner"] in ("blue", "red")
    # Should include model_move (may be None if game over)
    assert "model_move" in data

def test_api_move_invalid():
    # Try to play an invalid move
    trmph = "#13,"
    move = "z99"
    resp = requests.post(API_URL + "/api/move", json={"trmph": trmph, "move": move})
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data 