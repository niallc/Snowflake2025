# 2-Stage Tournament System Documentation

## Overview

The 2-stage tournament system efficiently identifies the best models from large sets of checkpoints through a knockout elimination stage followed by a round-robin ranking stage. This system is designed to handle 50+ model checkpoints efficiently while maintaining backward compatibility with existing tournament functionality.

## System Architecture

### High-Level Code Structure

```
TwoStageTournament (orchestrator)
├── CheckpointDiscovery
│   ├── Discovers checkpoints matching epochN_miniJ.pt.gz pattern
│   ├── Calculates sequential checkpoint numbers
│   └── Validates ordering and completeness
├── KnockoutTournament
│   ├── Executes spaced pairing strategy
│   ├── Handles byes for odd participant counts
│   └── Manages tournament bracket progression
└── Round-Robin Integration
    ├── Converts participants to StrategyConfig objects
    ├── Generates opening positions
    └── Executes using existing run_tournament infrastructure
```

### Code Flow Diagram

```
Input: Directory with N checkpoints + optional round-robin participants
  ↓
CheckpointDiscovery.discover_checkpoints()
  ├── Find files matching epochN_miniJ.pt.gz pattern
  ├── Calculate sequential checkpoint numbers: (epoch-1) * total_mini_epochs + mini + 1
  ├── Sort by creation time
  └── Validate sequential ordering
  ↓
TwoStageTournament._run_knockout_stage()
  ├── Create TournamentParticipant objects from CheckpointInfo
  ├── Initialize KnockoutTournament with spaced pairing
  └── Execute tournament with progressive game counts
  ↓
KnockoutTournament.run_tournament()
  ├── _pair_participants() - spaced pairing (1v5, 2v6, 3v7, 4v8, 9 gets bye)
  ├── _execute_round() - play matches using match_executor
  ├── _update_tournament_state() - advance winners
  └── Repeat until top_k winners remain
  ↓
TwoStageTournament._run_round_robin_stage()
  ├── Combine knockout winners + additional participants
  ├── Convert to StrategyConfig objects
  ├── Generate opening positions
  └── Execute using existing run_tournament() function
  ↓
Output: Final rankings with win rates and Elo ratings
```

## Core Components

### 1. CheckpointDiscovery (`hex_ai/inference/checkpoint_discovery.py`)

**Purpose**: Discovers and orders model checkpoints from directories

**Key Features**:
- Pattern matching for `epochN_miniJ.pt.gz` files
- Sequential checkpoint numbering: `(epoch-1) * total_mini_epochs + mini + 1`
- Creation time-based ordering with validation
- Handles missing checkpoints with warnings

**Example**:
```python
# For epoch14_mini1 through epoch14_mini9:
# epoch14_mini1 = (14-1) * 24 + 1 + 1 = 314
# epoch14_mini9 = (14-1) * 24 + 9 + 1 = 322
```

### 2. KnockoutTournament (`hex_ai/inference/knockout_tournament.py`)

**Purpose**: Executes knockout elimination tournaments with spaced pairing

**Key Features**:
- **Spaced Pairing Strategy**: First half vs second half (1v5, 2v6, 3v7, 4v8)
- **Bye Handling**: Last participant gets bye for odd numbers
- **Progressive Game Counts**: Semifinals (2x), Finals (4x)
- **Tie Handling**: Deterministic tiebreaker (first participant wins)

**Pairing Logic**:
```python
# For 9 participants: [1,2,3,4,5,6,7,8,9]
# Pairs: (1,5), (2,6), (3,7), (4,8)
# Bye: 9 advances automatically
```

### 3. TwoStageTournament (`hex_ai/inference/two_stage_tournament.py`)

**Purpose**: Orchestrates the complete 2-stage process

**Key Features**:
- Coordinates knockout and round-robin stages
- Manages opening position generation
- Handles tournament configuration and execution flow
- Provides comprehensive logging and JSON output

### 4. TournamentParticipant (`hex_ai/inference/knockout_tournament.py`)

**Purpose**: Abstract representation of tournament entrants

**Structure**:
```python
@dataclass
class TournamentParticipant:
    name: str                    # e.g., "epoch14_mini5"
    strategy_config: Dict[str, Any]  # MCTS parameters
    metadata: Dict[str, Any]     # Checkpoint info, file paths, etc.
```

## Implementation Decisions

### 1. Match Execution Integration

**Decision**: Use existing `play_deterministic_game` function from `scripts/run_tournament.py`

**Rationale**:
- Leverages proven game execution infrastructure
- Maintains consistency with existing tournament system
- Avoids code duplication

**Implementation**:
```python
# Convert TournamentParticipant to StrategyConfig
strategy_config = StrategyConfig(
    name=participant.name,
    strategy_type="mcts",
    config=clean_config,  # Remove non-MoveSelectionConfig keys
    model_path=participant.strategy_config["model_path"]
)

# Execute games using existing infrastructure
result = play_deterministic_game(
    model_cache=model_cache,
    strategy_a=strategy_a,
    strategy_b=strategy_b,
    opening=opening,
    temperature=knockout_config.get("temperature", 1.0)
)
```

### 2. Opening Position Strategy

**Decision**: Generate openings separately for knockout and round-robin stages

**Rationale**:
- Ensures sufficient unique openings for each stage
- Allows different game counts per stage
- Fails fast if insufficient openings available

**Implementation**:
```python
# Generate 1.1x required openings (fail fast if insufficient)
target_count = int(num_games * 1.1)
openings = generate_diverse_openings(trmph_files, target_count=target_count)

if len(openings) < num_games:
    raise ValueError(f"Insufficient openings generated: {len(openings)} < {num_games}")
```

### 3. Spaced Pairing Strategy

**Decision**: Pair first half with second half instead of adjacent participants

**Rationale**:
- More efficient tournament structure
- Early checkpoints face later checkpoints (assuming sequential training)
- Avoids pitting very similar models against each other

**Example**:
- **Before**: 1v2, 3v4, 5v6, 7v8, 9 gets bye
- **After**: 1v5, 2v6, 3v7, 4v8, 9 gets bye

### 4. Round-Robin Integration

**Decision**: Use existing `run_tournament` function for round-robin stage

**Rationale**:
- Leverages proven round-robin infrastructure
- Maintains consistency with existing tournament results
- Avoids reimplementing complex tournament logic

**Implementation**:
```python
# Convert participants to strategy configs
strategy_configs = [self._create_strategy_config(p) for p in all_participants]

# Execute using existing infrastructure
tournament_result = run_tournament(
    strategy_configs=strategy_configs,
    openings=openings,
    temperature=self.knockout_config.get("temperature", 1.0),
    verbose=1
)
```

### 5. Error Handling Philosophy

**Decision**: Follow project's "fail fast" principle

**Implementation**:
- **CRASH** on invalid states rather than silent failures
- **CRASH** on insufficient openings
- **CRASH** on invalid strategy configurations
- **CRASH** on missing checkpoints
- No fallback mechanisms that could hide problems

### 6. Logging and Progress Reporting

**Decision**: Comprehensive logging with real-time progress updates

**Implementation**:
- Terminal: "." every 10 games, match summaries
- Files: Streaming match results, tournament progression
- JSON: Complete tournament summary with all results
- Generous metadata logging (within few MB limit)

## CLI Integration

### New Arguments for `run_tournament.py`

```bash
# Knockout stage arguments
--knockout-dir DIR              # Directory containing checkpoints for knockout
--knockout-config CONFIG        # JSON config for knockout strategy
--games-per-match N             # Games per knockout match (default: 50)
--top-k N                       # Number of winners from knockout (default: 2)

# Round-robin stage arguments (existing)
--models MODEL1,MODEL2,...      # Models for round-robin stage
--strategies STRAT1,STRAT2,...  # Strategies for round-robin stage
--round-robin-games N           # Number of games per round-robin match (default: 100)
```

### Usage Examples

```bash
# 2-stage tournament: knockout 50 checkpoints, then round-robin top 2 + existing models
python scripts/run_tournament.py \
  --knockout-dir checkpoints/training_run \
  --knockout-config '{"enable_gumbel_root_selection": true, "mcts_sims": 220, "temperature": 1.0}' \
  --games-per-match 50 \
  --top-k 2 \
  --models current_best,previous_best \
  --strategies mcts,mcts \
  --round-robin-games 100

# Knockout only (no round-robin participants)
python scripts/run_tournament.py \
  --knockout-dir checkpoints/training_run \
  --top-k 3
```

## Current System Status

### ✅ Completed Features

1. **Core Architecture**: All components implemented and tested
2. **Match Execution**: Fully integrated with existing game execution infrastructure
3. **Round-Robin Integration**: Seamlessly integrated with existing tournament system
4. **Comprehensive Logging**: Real-time progress, match results, JSON summaries
5. **Robust Error Handling**: Fail-fast validation throughout
6. **CLI Integration**: Extended `run_tournament.py` with new arguments
7. **Backward Compatibility**: Existing tournament functionality preserved

### 🧪 Tested Scenarios

- **9 checkpoints**: Successfully demonstrated knockout → round-robin flow
- **Spaced pairing**: Verified correct pairing strategy (1v5, 2v6, 3v7, 4v8, 9 gets bye)
- **Tie handling**: Deterministic tiebreakers working correctly
- **Bye handling**: Odd participant counts handled properly
- **Integration**: Full end-to-end tournament execution successful

## Current Complexities and Suboptimal Aspects

### 1. **Import Dependencies**

**Issue**: The system imports from `scripts/run_tournament.py` for game execution functions

**Complexity**:
```python
# In two_stage_tournament.py
from scripts.run_tournament import play_deterministic_game, generate_diverse_openings
```

**Why Suboptimal**:
- Creates dependency on script-level code from library code
- Makes the system less modular
- Could cause import issues in different contexts

**Potential Solution**: Move core game execution functions to a dedicated module in `hex_ai/inference/`

### 2. **Strategy Configuration Mapping**

**Issue**: Complex mapping between `TournamentParticipant` and `StrategyConfig`

**Complexity**:
```python
def _create_strategy_config(self, participant: TournamentParticipant):
    # Clean the strategy config to only include MoveSelectionConfig parameters
    clean_config = {}
    for key, value in participant.strategy_config.items():
        if key not in ["strategy", "model_path"]:  # Remove non-MoveSelectionConfig keys
            clean_config[key] = value
```

**Why Suboptimal**:
- Requires manual filtering of configuration keys
- Tight coupling between participant representation and strategy configuration
- Could break if `MoveSelectionConfig` parameters change

**Potential Solution**: Create a more explicit configuration mapping system

### 3. **Opening Position Generation**

**Issue**: Opening generation is duplicated between knockout and round-robin stages

**Complexity**:
```python
# In _generate_match_openings()
openings = generate_diverse_openings(trmph_files, target_count=target_count)

# In _generate_round_robin_openings()  
openings = generate_diverse_openings(trmph_files, target_count=target_count)
```

**Why Suboptimal**:
- Code duplication
- Potential inconsistency in opening generation parameters
- No caching of openings between stages

**Potential Solution**: Centralize opening generation with configurable parameters

### 4. **Tournament State Management**

**Issue**: Tournament state is managed across multiple classes with some redundancy

**Complexity**:
- `KnockoutTournament` manages its own state
- `TwoStageTournament` coordinates between stages
- Some state information is duplicated

**Why Suboptimal**:
- Makes debugging more difficult
- Potential for state inconsistencies
- Harder to extend with new tournament formats

**Potential Solution**: Create a unified tournament state manager

### 5. **Error Message Specificity**

**Issue**: Some error messages could be more specific about what went wrong

**Example**:
```python
if len(openings) < num_games:
    raise ValueError(f"Insufficient openings generated: {len(openings)} < {num_games}")
```

**Why Suboptimal**:
- Doesn't explain why openings were insufficient
- Doesn't suggest potential solutions
- Could be more helpful for debugging

**Potential Solution**: Add more context to error messages and suggest solutions

### 6. **Memory Usage for Large Tournaments**

**Issue**: The system loads all checkpoint metadata at startup

**Complexity**:
```python
# All checkpoints loaded into memory at once
checkpoints = discovery.discover_checkpoints()
```

**Why Suboptimal**:
- For very large checkpoint directories (1000+ files), this could use significant memory
- No lazy loading or pagination

**Potential Solution**: Implement lazy loading or checkpoint pagination for large directories

## Future Enhancements

### Potential Improvements

1. **Parallelization**: Run multiple matches simultaneously
2. **Resume Functionality**: Restart interrupted tournaments
3. **Advanced Brackets**: Swiss system, single elimination options
4. **Dynamic Game Counts**: Adjust based on match competitiveness
5. **Real-time Monitoring**: Web interface for tournament progress

### Extensibility Points

- **Custom Tournament Formats**: Extend `KnockoutTournament` for different elimination formats
- **Participant Types**: Extend `TournamentParticipant` for different participant types
- **Result Formats**: Extensible tournament result system for different output formats

## Conclusion

The 2-stage tournament system successfully provides an efficient way to identify the best models from large checkpoint sets. While there are some areas for improvement in terms of code organization and modularity, the system is robust, well-tested, and ready for production use. The implementation follows the project's principles of failing fast, avoiding fallback mechanisms, and maintaining clean abstractions.
