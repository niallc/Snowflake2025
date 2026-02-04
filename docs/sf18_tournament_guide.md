# SF18 vs SF25 Tournament Guide

This guide explains how to run tournaments between your current Snowflake 2025 models and the older Snowflake 2018 (SF18) AI.

## Overview

The SF18 vs SF25 tournament system allows you to:
- Compare your current SF25 models against the older SF18 AI
- Use the same opening generation system as regular tournaments
- Configure SF18 difficulty levels (1-10)
- Run tournaments with multiple SF25 models simultaneously
- Get detailed win/loss statistics and game logs

## Prerequisites

1. **SF18 Server**: The SF18 AI must be running as a webserver
2. **SF25 Models**: Your current model files or registry entries
3. **Environment**: Virtual environment activated (recommended: `pip install -e .` once per venv)

## Starting the SF18 Server

Before running tournaments, you need to start the SF18 server:

```bash
python temp/SF18Interface/HttpGameServer.py \
  --valueBuilderPath13=/path/to/value/model \
  --policyBuilderPath13=/path/to/policy/model \
  --twoHeadBuilderPath=/path/to/twohead/model \
  --policyNetworkType=twoHeaded \
  --portNum=8088
```

The server will be available at `http://localhost:8088` by default.

## Basic Usage

### Simple Tournament

Compare one SF25 model against SF18:

```bash
source hex_ai_env/bin/activate
python scripts/run_sf18_tournament.py \
  --models=current_best \
  --strategies=mcts \
  --mcts-sims=30 \
  --num-openings=100 \
  --sf18-difficulty=9
```

### Multiple Models

Compare multiple SF25 models against SF18:

```bash
python scripts/run_sf18_tournament.py \
  --model-files=epoch11_mini15.pt.gz,epoch16_mini23.pt.gz \
  --model-dirs=checkpoints/dir1,checkpoints/dir2 \
  --strategies=mcts,mcts \
  --mcts-sims=30,30 \
  --num-openings=100 \
  --sf18-difficulty=9
```

### Custom Openings

Use a pre-generated opening file:

```bash
python scripts/run_sf18_tournament.py \
  --models=current_best \
  --strategies=mcts \
  --mcts-sims=30 \
  --opening-file=data/deterministic_openings.txt \
  --sf18-difficulty=9
```

## Command Line Arguments

### SF25 Model Arguments

- `--models`: Comma-separated model registry names
- `--model-files`: Comma-separated model file names
- `--model-dirs`: Comma-separated model directories (used with --model-files)
- `--strategies`: Comma-separated strategies (e.g., "mcts,policy")
- `--mcts-sims`: Comma-separated MCTS simulation counts
- `--enable-gumbel`: Enable Gumbel AlphaZero (comma-separated booleans)
- `--temperature`: Global temperature for move selection

### SF18 Arguments

- `--sf18-difficulty`: Difficulty level 1-10 (default: 9)
- `--sf18-server-url`: SF18 server URL (default: http://localhost:8088)
- `--sf18-timeout`: Request timeout in seconds (default: 30)

**Note**: The system includes a configurable delay between SF18 server requests (default: 0 seconds for local servers). This can be increased for remote/internet servers by modifying `SF18_REQUEST_DELAY` in `hex_ai/inference/sf18_client.py`.

### Opening Arguments

- `--num-openings`: Number of openings to generate (default: 100)
- `--opening-length`: Moves per opening (default: 5)
- `--opening-file`: Use pre-generated openings file
- `--trmph-source`: Directory for TRMPH files (default: data/sf25/sep28)

### General Arguments

- `--seed`: Random seed for opening selection
- `--verbose`: Verbosity level (default: 1)

## Output

The tournament produces:

1. **Console Output**: Real-time progress and final results
2. **TRMPH Files**: Game sequences in TRMPH format
3. **CSV Files**: Detailed game statistics
4. **JSON Results**: Tournament summary in JSON format

### Example Output

```
SF18 vs SF25 TOURNAMENT RESULTS
============================================================
SF18 Difficulty: 9
SF25 Models: 2

Model: epoch11_mini15_mcts_t0.0_sims30
  SF25 Wins: 45
  SF18 Wins: 55
  Total Games: 100
  SF25 Win Rate: 0.450

Model: epoch16_mini23_mcts_t0.0_sims30
  SF25 Wins: 52
  SF18 Wins: 48
  Total Games: 100
  SF25 Win Rate: 0.520
============================================================
```

## Troubleshooting

### SF18 Server Not Running

If you get an error about the SF18 server not running:

```
ERROR: Cannot connect to SF18 server at http://localhost:8088
```

Make sure the SF18 server is started with the correct model paths.

### Model Files Not Found

If you get model file errors:

```
ERROR: Model file does not exist: /path/to/model.pt.gz
```

Check that your model paths are correct and the files exist.

### Invalid Strategy Configuration

If you get strategy configuration errors:

```
ERROR: Duplicate strategy configurations detected.
```

Make sure each strategy has unique parameters (different models, temperatures, etc.).

## Examples

See `scripts/example_sf18_tournament.py` for more detailed examples of different tournament configurations.

## Technical Details

- The tournament plays each SF25 model against SF18 using the same opening positions
- Each opening is played twice (SF25 as blue, SF25 as red) to eliminate color bias
- SF18 uses the HTTP API interface described in `temp/SF18Interface/API_INTERFACE_README.md`
- The system reuses existing opening generation and tournament infrastructure
- Results are compatible with the existing tournament analysis tools
