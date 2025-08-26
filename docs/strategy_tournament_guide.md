# Strategy Tournament System Guide

## Overview

The Strategy Tournament System allows you to compare different move selection strategies (policy, MCTS, fixed tree search) using the same model. This is useful for understanding which strategy performs best and for tuning strategy parameters.

## Key Features

- **Strategy vs Strategy Comparison**: Compare different move selection strategies using the same model
- **Round-robin Format**: All strategy pairs play against each other
- **Flexible Configuration**: Support for policy, MCTS, and fixed tree search strategies
- **Parameter Tuning**: Easy comparison of different MCTS simulation counts or search widths
- **Comprehensive Logging**: TRMPH format games and CSV statistics for each strategy pair
- **Win Rate Analysis**: Automatic calculation of win rates and ELO ratings

## Available Strategies

### 1. Policy-based Selection (`policy`)
- Direct policy sampling from the neural network
- Fastest strategy, good baseline
- Configurable temperature for randomness

### 2. MCTS (`mcts_<sims>`)
- Monte Carlo Tree Search with specified simulation count
- Examples: `mcts_100`, `mcts_200`, `mcts_122`
- Configurable parameters: simulations, c_puct, dirichlet noise

### 3. Fixed Tree Search (`fixed_tree_<width1>_<width2>_...`)
- Fixed-width minimax search with policy-value evaluation
- Examples: `fixed_tree_13_8`, `fixed_tree_20_10`
- Configurable search widths for each ply

## Usage Examples

### Basic Strategy Comparison

Compare policy vs MCTS vs fixed tree search:

```bash
PYTHONPATH=. python scripts/run_strategy_tournament.py \
  --model=current_best \
  --strategies=policy,mcts_122,fixed_tree_13_8 \
  --num-games=50
```

### MCTS Parameter Tuning

Compare MCTS with different simulation counts:

```bash
PYTHONPATH=. python scripts/run_strategy_tournament.py \
  --model=current_best \
  --strategies=mcts_50,mcts_100,mcts_200 \
  --num-games=100
```

### Fixed Tree Search Width Comparison

Compare different search width configurations:

```bash
PYTHONPATH=. python scripts/run_strategy_tournament.py \
  --model=current_best \
  --strategies=fixed_tree_10_5,fixed_tree_13_8,fixed_tree_20_10 \
  --num-games=50
```

### Mixed Strategy Comparison

Compare all three strategy types:

```bash
PYTHONPATH=. python scripts/run_strategy_tournament.py \
  --model=current_best \
  --strategies=policy,mcts_122,fixed_tree_13_8 \
  --num-games=100 \
  --temperature=1.2
```

## Command Line Options

### Required Arguments

- `--strategies`: Comma-separated list of strategies to compare
  - Format: `policy,mcts_100,fixed_tree_13_8`
  - MCTS format: `mcts_<simulation_count>`
  - Fixed tree format: `fixed_tree_<width1>_<width2>_...`

### Optional Arguments

- `--model`: Model to use (default: `current_best`)
- `--num-games`: Number of games per strategy pair (default: 50)
- `--temperature`: Temperature for move selection (default: 1.0)
- `--mcts-sims`: Override MCTS simulation counts (comma-separated)
- `--search-widths`: Override search widths (semicolon-separated, e.g., "13,8;20,10")
- `--seed`: Random seed (default: 42)
- `--no-pie-rule`: Disable pie rule (enabled by default)
- `--verbose`: Verbosity level (default: 1)

## Output Format

### Directory Structure

Results are saved to:
```
data/tournament_play/strategy_tournament_YYYYMMDD_HHMM/
├── strategy_a_vs_strategy_b.trmph
├── strategy_a_vs_strategy_b.csv
├── strategy_a_vs_strategy_c.trmph
├── strategy_a_vs_strategy_c.csv
└── ...
```

### TRMPH Files

Each strategy pair gets a TRMPH file containing all games in TRMPH format:
```
#13,move1move2move3... winner
#13,move1move2move3... winner
...
```

### CSV Files

Each strategy pair gets a CSV file with detailed statistics:
- `timestamp`: When the game was played
- `strategy_a`, `strategy_b`: Strategy names
- `game`: Which strategy went first ("A_first" or "B_first")
- `trmph`: TRMPH format game string
- `winner`: Winner color ('b' or 'r')
- `winner_strategy`: Which strategy won
- `num_moves`: Number of moves in the game
- `temperature`: Temperature used
- `seed`: Random seed used

## Tournament Analysis

The system automatically provides:

### Win Rates
- Percentage of games won by each strategy
- Total wins and games played

### ELO Ratings
- Order-independent ELO calculation
- Based on all head-to-head results

### Head-to-Head Results
- Detailed breakdown of each strategy pair
- Win/loss record and win percentage

## Testing

Run the test suite to validate the system:

```bash
PYTHONPATH=. python scripts/test_strategy_tournament.py
```

This runs:
- Strategy configuration parsing tests
- Small tournament validation
- Strategy config creation tests

## Implementation Details

### Strategy Configuration

Strategies are configured using the `StrategyConfig` class:

```python
# Policy strategy
policy_config = StrategyConfig("policy", "policy", {})

# MCTS strategy with 100 simulations
mcts_config = StrategyConfig("mcts_100", "mcts", {
    "mcts_sims": 100,
    "mcts_c_puct": 1.5
})

# Fixed tree strategy with widths [13, 8]
tree_config = StrategyConfig("fixed_tree_13_8", "fixed_tree", {
    "search_widths": [13, 8]
})
```

### Game Playing

Each game is played using the `play_strategy_vs_strategy_game` function:

1. Initialize game state
2. For each move, determine which strategy to use based on current player
3. Use the strategy's move selection method
4. Apply the move and continue until game over
5. Record winner and game statistics

### Tournament Structure

1. **Round-robin**: All strategy pairs play against each other
2. **Color balancing**: Each strategy plays as both Blue and Red
3. **Multiple games**: Configurable number of games per pair
4. **Result tracking**: Win/loss records for tournament analysis

## Best Practices

### For Testing
- Start with small numbers of games (2-5) to validate setup
- Use `--verbose=2` for detailed output
- Disable pie rule with `--no-pie-rule` for simpler testing

### For Production
- Use 50-100 games per pair for reliable statistics
- Test with different temperatures to understand strategy robustness
- Compare multiple parameter settings to find optimal configurations

### For Analysis
- Check both win rates and ELO ratings
- Examine head-to-head results for specific insights
- Use CSV files for detailed statistical analysis
- TRMPH files can be used for game replay and analysis

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct and the file exists
2. **Strategy parsing errors**: Check strategy name format (e.g., `mcts_100`, not `mcts100`)
3. **Memory issues**: Reduce number of games or use smaller MCTS simulation counts
4. **Slow performance**: Policy strategy is fastest, MCTS scales with simulation count

### Debug Mode

Use `--verbose=2` for detailed output showing:
- Game progress indicators
- Strategy selection details
- Move-by-move information

## Integration with Existing Code

The strategy tournament system integrates with existing components:

- **Model Cache**: Efficient model loading and caching
- **Move Selection**: Uses existing strategy registry
- **Tournament Infrastructure**: Leverages existing result tracking and analysis
- **Logging**: Compatible with existing TRMPH and CSV logging

This allows for easy comparison between the new strategy tournament system and the existing model tournament system.
