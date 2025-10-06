# SF18 Integration Issues and Approach

## Overview

I've created a tournament system that allows Snowflake 2025 models to play against the older Snowflake 2018 (SF18) AI. The SF18 AI runs as a separate webserver, and the SF25 tournament system communicates with it via HTTP API calls.

## Architecture

### SF25 Side (Completed)
- **SF18Client**: HTTP client that communicates with SF18 server
- **SF18Player**: Wrapper class that integrates SF18 with existing tournament infrastructure
- **run_sf18_tournament.py**: Main tournament script with interface similar to run_tournament.py
- **Tournament System**: Reuses existing opening generation, game execution, and result reporting

### SF18 Side (Needs Testing/Fixes)
- **HttpGameServer.py**: Webserver that provides `/api/move` endpoint
- **API Interface**: JSON-based communication for move requests

## Current Issues

### 1. Server Communication Error
**Problem**: The SF18 server is returning HTTP 405 "Method Not Allowed" for POST requests to `/api/move`

**Expected Request Format**:
```json
POST http://localhost:8088/api/move
Content-Type: application/json

{
  "moves": "a13d3k10h6g6k11l10g7j4",
  "size": 13,
  "difficulty": 9,
  "include_metadata": true
}
```

**Expected Response Format**:
```json
{
  "move": "m4",
  "winner": "no winner",
  "metadata": {
    "row": 3,
    "column": 12,
    "move_number": 8,
    "player": "blue"
  }
}
```

**Current Error**: `405 Client Error: METHOD NOT ALLOWED for url: http://localhost:8088/api/move`

### 2. Player Enum Comparison Error
**Problem**: There's a "Cannot compare Player with int" error occurring during game play

**Context**: This error appears in the SF25 tournament system when trying to determine game winners. The issue seems to be related to how Player enums are being compared.

**SF25 Player/Winner Enums**:
```python
class Player(StrictEnum):
    BLUE = 0
    RED = 1

class Winner(StrictEnum):
    BLUE = 0
    RED = 1
```

## Integration Approach

### SF18 Client Implementation
The SF18 client converts HexGameState objects to TRMPH format and sends them to the SF18 server:

```python
# Convert state to TRMPH format using built-in method
trmph_str = state.to_trmph()
# Remove the "#13," prefix to get just the moves
moves_str = trmph_str[4:] if trmph_str.startswith("#13,") else trmph_str

# Get move from server
result = self.get_move(moves_str, difficulty=difficulty, include_metadata=True)
```

### Tournament Flow
1. Generate opening positions (reuses existing SF25 opening generation)
2. For each opening, play two games:
   - Game 1: SF25 (Blue) vs SF18 (Red)
   - Game 2: SF18 (Blue) vs SF25 (Red)
3. Record results and generate tournament statistics

## Testing Status

### What Works
- ✅ SF25 tournament script loads and parses arguments correctly
- ✅ Opening generation works (reuses existing SF25 code)
- ✅ SF18 client can detect if server is running
- ✅ Tournament infrastructure integration is complete
- ✅ Result reporting and logging works

### What Needs Fixing
- ❌ SF18 server API endpoint (405 Method Not Allowed)
- ❌ Player enum comparison error in winner determination
- ❌ Actual game play between SF25 and SF18 (blocked by above issues)

## Files Created

### SF25 Side
- `hex_ai/inference/sf18_client.py` - SF18 communication client
- `scripts/run_sf18_tournament.py` - Main tournament script
- `scripts/example_sf18_tournament.py` - Usage examples
- `docs/sf18_tournament_guide.md` - User documentation

### SF18 Side (Existing)
- `temp/SF18Interface/HttpGameServer.py` - Webserver
- `temp/SF18Interface/API_INTERFACE_README.md` - API documentation
- `temp/SF18Interface/demo_api_usage.py` - Example usage
- `temp/SF18Interface/hex_client_example.py` - Client example

## Next Steps

1. **Fix SF18 Server API**: Ensure the `/api/move` endpoint accepts POST requests with the expected JSON format
2. **Test SF18 Server**: Verify the server can handle the TRMPH format moves and return valid responses
3. **Debug Player Enum Issue**: Investigate the "Cannot compare Player with int" error in the SF25 tournament system
4. **End-to-End Testing**: Run a complete tournament once the server issues are resolved

## Example Usage (Once Fixed)

```bash
# Start SF18 server
python temp/SF18Interface/HttpGameServer.py \
  --valueBuilderPath13=/path/to/value/model \
  --policyBuilderPath13=/path/to/policy/model \
  --twoHeadBuilderPath=/path/to/twohead/model \
  --policyNetworkType=twoHeaded \
  --portNum=8088

# Run tournament
source hex_ai_env/bin/activate
export PYTHONPATH=.
python scripts/run_sf18_tournament.py \
  --models=current_best \
  --strategies=mcts \
  --mcts-sims=30 \
  --num-openings=100 \
  --sf18-difficulty=9
```

## Questions for SF18 Side

1. Is the `/api/move` endpoint properly configured to accept POST requests?
2. Are there any specific requirements for the JSON request format?
3. Should the server handle the TRMPH format moves as expected?
4. Are there any authentication or special headers required?
5. What should the server return when the game is over (no more moves possible)?

The SF25 side is ready to integrate once the SF18 server issues are resolved.
