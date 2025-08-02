# Advice from Claude Sonnet 4

## Root Cause Identified: macOS Port 5000 Conflict

The HTTP 403 errors you're experiencing are **not actually CORS issues or Flask problems**. Instead, this is a well-documented macOS system conflict that affects Flask development on Mac systems running macOS Monterey (12.x) or later.

### What's Actually Happening

Starting with macOS Monterey, Apple began using port 5000 for the system's AirPlay Receiver service (part of Control Centre). When your Flask app tries to bind to port 5000, macOS's AirPlay service intercepts all HTTP requests to that port and returns a 403 Forbidden response. Your Flask application never actually receives these requests, which explains why:

- Both valid and invalid requests return 403
- CORS fixes don't help
- Debug logging in Flask shows nothing
- The error is consistent across all endpoints

### Immediate Solutions

**Option 1: Change Flask Port (Recommended)**
Modify your `hex_ai/web/app.py` to use a different port:

```python
if __name__ == "__main__" or __name__ == "hex_ai.web.app":
    app.run(debug=True, port=5001)  # Change from default 5000
```

Then update your test configuration:
```python
# In tests/test_web_api.py
API_URL = "http://localhost:5001"  # Change from 5000
```

**Option 2: Disable AirPlay Receiver**
- On macOS Ventura/Sonoma: System Settings → General → AirDrop & Handoff → Turn off "AirPlay Receiver"
- On macOS Monterey: System Preferences → Sharing → Turn off "AirPlay Receiver"

**Option 3: Use IP Address Instead of localhost**
Change your test URL to `http://127.0.0.1:5000` instead of `http://localhost:5000`. This sometimes bypasses the macOS interception, though it's less reliable than the other solutions.

### Why This Wasn't in Your Troubleshooting

This issue is macOS-specific and relatively recent (introduced in 2021), so it doesn't appear in most Flask documentation or common troubleshooting guides. The 403 error is particularly misleading because it suggests a permissions or security issue rather than a port conflict.

### Verification

After implementing one of these solutions, your existing tests should pass without any other changes. The fact that you're getting consistent 403s across all request types strongly suggests this port conflict rather than any actual issues with your Flask implementation, CORS configuration, or test setup.

### Long-term Recommendation

For a production-ready Flask application, you should avoid relying on port 5000 anyway, as it can conflict with various system services across different operating systems. Consider using ports like 8000, 8080, or 5001 as your default development ports.