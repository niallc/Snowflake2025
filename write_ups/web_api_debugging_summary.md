# Hex Web API Debugging Summary

## Overview
This document describes the implementation of the Hex web API (Flask backend), the integration tests written for it, the errors encountered (HTTP 403 on all API requests), and the troubleshooting steps taken so far. This is intended to help with external troubleshooting and to provide context for further debugging.

---

## 1. Web API Implementation

**Location:** `hex_ai/web/app.py`

### Key Features
- **Flask app** serving two main endpoints:
  - `POST /api/state`: Accepts a TRMPH string, returns board state, legal moves, winner, model policy/value, win probability, and TRMPH.
  - `POST /api/move`: Accepts a TRMPH string and a move, applies the move, and (if not game over) has the model pick and play its move. Returns updated state, model move, policy, value, win probability, and TRMPH.
- **Model inference** is handled by `SimpleModelInference` (from `hex_ai/inference/simple_model_inference.py`).
- **Game state** is managed by `HexGameState` (from `hex_ai/inference/game_engine.py`).
- **TRMPH parsing and move conversion** use utilities from `hex_ai/utils/format_conversion.py`.
- **Player/winner representation** uses enums/utilities from `hex_ai/value_utils.py`.
- **CORS** is enabled via `flask_cors.CORS(app)` to allow local frontend and test access.
- **Debug logging** is enabled for all incoming requests.
- The app can be run as a module (`python -m hex_ai.web.app`) or as a script.

### Example Endpoint Implementation
```python
@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json()
    trmph = data.get("trmph")
    state = HexGameState.from_trmph(trmph)
    ...
    model = get_model()
    policy_logits, value_logit = model.infer(trmph)
    ...
    return jsonify({...})
```

---

## 2. Integration Tests

**Location:** `tests/test_web_api.py`

### What the Tests Do
- **Start the Flask app** in a subprocess with `PYTHONPATH=.`
- **Test `/api/state` with a valid TRMPH**: Expects 200 OK and correct response fields.
- **Test `/api/state` with an invalid TRMPH**: Expects 400 error and error message.
- **Test `/api/move` with a valid move**: Expects 200 OK, correct state update, and model move if applicable.
- **Test `/api/move` with an invalid move**: Expects 400 error and error message.

### Example Test
```python
def test_api_state_valid():
    trmph = "#13,"
    resp = requests.post(API_URL + "/api/state", json={"trmph": trmph})
    assert resp.status_code == 200
    data = resp.json()
    assert "board" in data
    ...
```

---

## 3. Errors Encountered

### Main Error
- **All API tests fail with HTTP 403 Forbidden** (instead of 200/400):
  - `assert 403 == 200` (or `assert 403 == 400`)
- This occurs for both valid and invalid requests, and for both `/api/state` and `/api/move`.
- The error persists even after enabling CORS and running the app in debug mode.

### Example Failure Output
```
>       assert resp.status_code == 200
E       assert 403 == 200
E        +  where 403 = <Response [403]>.status_code
```

---

## 4. Troubleshooting Steps Taken

1. **Enabled CORS** with `flask_cors.CORS(app)` to allow all origins and methods.
2. **Ensured the Flask app can be run as a module** (`python -m hex_ai.web.app`) by adding a proper `if __name__ == "__main__" or __name__ == "hex_ai.web.app"` block.
3. **Added debug logging** for all incoming requests to verify what Flask is receiving.
4. **Checked for route conflicts** (e.g., static file serving shadowing `/api/*` routes).
5. **Tested with both valid and invalid requests** to both endpoints.
6. **Verified that the test harness sets `PYTHONPATH=.`** and starts the app in the correct environment.
7. **Confirmed that the endpoints are defined with `methods=["POST"]`** and that the test uses `requests.post`.

---

## 5. Next Steps / Open Questions

- Is there a Flask or Werkzeug security setting, proxy, or middleware that could be blocking POST requests to `/api/*`?
- Is the static file serving interfering with the `/api/*` routes?
- Would running the Flask app directly (not as a module) change the behavior?
- Is there a way to get more detailed error output from Flask/Werkzeug for 403 responses?
- Are there any platform-specific issues (macOS, Anaconda, etc.) that could cause this?

---

## 6. What to Try Next

- Run the Flask app directly (`python hex_ai/web/app.py`) and test the API with `curl` or `requests`.
- Add more detailed error logging to Flask to capture the reason for 403 responses.
- Check for any .htaccess, proxy, or server configuration files that could affect routing.
- Search for known issues with Flask, CORS, and 403 errors on POST requests in local development.

---

## 7. Summary

- The backend and tests are implemented as expected, but all POST requests to `/api/*` return 403 Forbidden.
- Usual fixes (CORS, debug mode, correct main block) have not resolved the issue.
- Further investigation is needed, possibly with external help or by searching for platform-specific Flask issues. 