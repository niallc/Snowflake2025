import os
import sys
import time
import requests
import subprocess
import signal
import pytest

STATIC_URLS = [
    ("/", "text/html"),
    ("/static/style.css", "text/css"),
    ("/static/app.js", "application/javascript"),
    ("/favicon.ico", "image/x-icon"),
]

API_URL = "http://127.0.0.1:5001"

@pytest.fixture(scope="module", autouse=True)
def flask_server():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.Popen([sys.executable, "-m", "hex_ai.web.app"], env=env)
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

def test_static_assets():
    for path, expected_type in STATIC_URLS:
        resp = requests.get(API_URL + path)
        assert resp.status_code == 200, f"{path} did not return 200"
        if expected_type == "text/html":
            assert "<html" in resp.text.lower()
        elif expected_type == "text/css":
            assert "body {" in resp.text or "#app-container {" in resp.text
        elif expected_type == "application/javascript":
            assert "function" in resp.text or "const" in resp.text
        elif expected_type == "image/x-icon":
            assert resp.content[:4] == b'\x00\x00\x01\x00' or resp.content[:4] == b'\x89PNG', "favicon is not an icon or PNG" 