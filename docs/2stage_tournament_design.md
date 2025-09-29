# 2-Stage Tournament System Design Document

## Overview

This document outlines the design for a new 2-stage tournament system that efficiently identifies the best models from large sets of checkpoints through a knockout elimination stage followed by a round-robin ranking stage.

## Problem Statement

The current all-play-all tournament system becomes inefficient when comparing many strategies (e.g., 50+ model checkpoints). We need a system that:
1. Efficiently narrows down a large set of checkpoints to the top k performers
2. Provides detailed ranking of the final candidates
3. Maintains backward compatibility with existing tournament functionality

## System Architecture

### High-Level Flow
```
Input: Directory with N checkpoints + optional round-robin participants
  ↓
Stage 1: Knockout Tournament (if N > 1)
  - Double elimination bracket
  - Configurable games per match (default: 50)
  - Progressive game counts (semifinals: 2x, finals: 4x)
  - Output: Top k winners
  ↓
Stage 2: Round Robin Tournament
  - Combines knockout winners + specified round-robin participants
  - Uses existing round-robin logic
  - Output: Final rankings
```

### Core Components

#### 1. Abstract Knockout Tournament (`KnockoutTournament`)
**Purpose**: Strategy-agnostic tournament execution
**Responsibilities**:
- Execute double elimination bracket
- Manage match progression and game counting
- Handle logging and progress reporting
- Return ordered list of winners

**Interface**:
```python
class KnockoutTournament:
    def __init__(self, participants: List[Participant], 
                 games_per_match: int = 50,
                 top_k: int = 2,
                 progress_callback: Optional[Callable] = None):
        pass
    
    def run_tournament(self) -> List[Participant]:
        """Execute knockout and return top k winners in order"""
        pass
```

#### 2. Participant Abstraction (`TournamentParticipant`)
**Purpose**: Abstract representation of tournament entrants
**Responsibilities**:
- Provide unique identifier
- Support strategy execution interface
- Handle metadata (model path, strategy config, etc.)

**Interface**:
```python
@dataclass
class TournamentParticipant:
    id: str  # Unique identifier
    name: str  # Human-readable name
    strategy_config: StrategyConfig  # Strategy configuration
    metadata: Dict[str, Any]  # Additional info (model path, etc.)
    
    def get_display_name(self) -> str:
        """Get formatted name for logging/display"""
        pass
```

#### 3. Checkpoint Discovery (`CheckpointDiscovery`)
**Purpose**: Find and order model checkpoints from directories
**Responsibilities**:
- Discover checkpoints matching pattern `epochN_miniJ.pt.gz`
- Order by creation timestamp (fallback to filename parsing)
- Validate checkpoint files exist
- Create TournamentParticipant objects

**Interface**:
```python
class CheckpointDiscovery:
    @staticmethod
    def discover_checkpoints(directory: str, 
                           strategy_type: str = "mcts",
                           strategy_config: Dict[str, Any] = None) -> List[TournamentParticipant]:
        """Discover and order checkpoints from directory"""
        pass
    
    @staticmethod
    def validate_checkpoint_pattern(directory: str) -> bool:
        """Validate directory contains expected checkpoint pattern"""
        pass
```

#### 4. Tournament Orchestration (`TwoStageTournament`)
**Purpose**: Coordinate the complete 2-stage process
**Responsibilities**:
- Manage opening position generation
- Coordinate knockout and round-robin stages
- Handle logging and output organization
- Provide unified interface

**Interface**:
```python
class TwoStageTournament:
    def __init__(self, 
                 knockout_dir: Optional[str] = None,
                 knockout_config: Dict[str, Any] = None,
                 round_robin_participants: List[TournamentParticipant] = None,
                 games_per_match: int = 50,
                 top_k: int = 2,
                 round_robin_games: int = 100):
        pass
    
    def run_tournament(self) -> TournamentResult:
        """Execute complete 2-stage tournament"""
        pass
```

## Implementation Details

### Knockout Tournament Algorithm

#### Double Elimination Structure
```
Round 1: N/2 matches (1 vs N/2+1, 2 vs N/2+2, ...)
Round 2: N/4 matches (winners advance)
...
Semifinals: 2 matches (2x games per match)
Finals: 1 match (4x games per match)
```

#### Match Execution
- Each match consists of multiple games (configurable)
- Games use different opening positions from shared pool
- Winner determined by total game wins
- Progress reported every 10 games with "."
- Match completion reported with winner and score

#### Bracket Management
- Use simple bracket structure (no losers' bracket needed)
- Track match results and advance winners
- Handle odd numbers of participants with byes

### Opening Position Management

#### Generation Strategy
- Generate openings separately for knockout and round-robin stages
- Knockout: `games_per_match * total_knockout_matches` openings
- Round-robin: `round_robin_games * total_round_robin_matches` openings
- Reuse existing `generate_diverse_openings` utility

#### Opening Distribution
```python
knockout_openings = generate_diverse_openings(trmph_files, target_count=knockout_total_games)
round_robin_openings = generate_diverse_openings(trmph_files, target_count=round_robin_total_games)
```

### Logging and Progress Reporting

#### File Organization
```
tournament_output/
├── knockout_stage/
│   ├── bracket_progress.log
│   ├── match_results.csv
│   └── individual_games/
│       ├── match_1_vs_2.trmph
│       └── match_1_vs_2.csv
├── round_robin_stage/
│   └── [existing round-robin logs]
└── tournament_summary.json
```

#### Progress Reporting
- Terminal: "." every 10 games, match summaries
- Files: Streaming match results, bracket progression
- JSON: Final tournament summary with all results

### Integration with Existing Code

#### Backward Compatibility
- `run_tournament.py` extended with new arguments
- If no knockout directory specified, runs round-robin only
- If no round-robin participants specified, uses knockout winners only
- Existing functionality preserved for current users

#### Code Reuse
- Reuse `play_deterministic_game` for individual games
- Reuse `generate_diverse_openings` for opening generation
- Reuse `TournamentResult` class for final results
- Reuse existing logging utilities where possible

## CLI Interface

### New Arguments for `run_tournament.py`

```bash
# Knockout stage arguments
--knockout-dir DIR              # Directory containing checkpoints for knockout
--knockout-config CONFIG        # JSON config for knockout strategy (default: gumbel enabled, 220 sims, temp=1.0)
--games-per-match N             # Games per knockout match (default: 50)
--top-k N                       # Number of winners from knockout (default: 2)

# Round-robin stage arguments (existing)
--models MODEL1,MODEL2,...      # Models for round-robin stage
--strategies STRAT1,STRAT2,...  # Strategies for round-robin stage
--round-robin-games N           # Number of games per round-robin match (default: 100)
# ... other existing arguments

# Tournament configuration
--output-dir DIR                # Tournament output directory
```

### Knockout Configuration Format

The `--knockout-config` parameter accepts a JSON string with MCTS strategy parameters:

**Default configuration** (if not specified):
```json
{
  "enable_gumbel_root_selection": true,
  "mcts_sims": 220,
  "temperature": 1.0
}
```

**Example custom configurations**:
```bash
# Custom simulation count
--knockout-config '{"mcts_sims": 100, "enable_gumbel_root_selection": true, "temperature": 1.0}'

# Different temperature
--knockout-config '{"mcts_sims": 220, "enable_gumbel_root_selection": true, "temperature": 0.5}'

# Disable Gumbel
--knockout-config '{"mcts_sims": 220, "enable_gumbel_root_selection": false, "temperature": 1.0}'
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
  --mcts-sims 200,200 \
  --round-robin-games 100

# Knockout only (no round-robin participants)
python scripts/run_tournament.py \
  --knockout-dir checkpoints/training_run \
  --top-k 3

# Round-robin only (no knockout)
python scripts/run_tournament.py \
  --models model1,model2,model3 \
  --strategies mcts,mcts,mcts \
  --round-robin-games 100
```

## Implementation Stages

### Stage 1: Core Abstractions
1. **Create `TournamentParticipant` class**
   - Abstract participant representation
   - Strategy configuration integration
   - Metadata handling

2. **Create `CheckpointDiscovery` utility**
   - Pattern matching for checkpoint files
   - Timestamp-based ordering
   - Validation and error handling

3. **Create `KnockoutTournament` class**
   - Double elimination bracket logic
   - Match execution and progress tracking
   - Winner determination

### Stage 2: Integration Layer
4. **Create `TwoStageTournament` orchestrator**
   - Coordinate knockout and round-robin stages
   - Opening position management
   - Result aggregation

5. **Extend `run_tournament.py`**
   - Add new CLI arguments
   - Integrate 2-stage logic
   - Maintain backward compatibility

### Stage 3: Logging and Polish
6. **Implement comprehensive logging**
   - Streaming match results
   - Progress reporting
   - File organization

7. **Development validation**
   - Test with real checkpoint directories
   - Validate against existing tournament results
   - Debug output and manual testing

## Error Handling and Validation

### Checkpoint Discovery
- **CRASH** if directory doesn't exist
- **CRASH** if no files match expected pattern
- **CRASH** if checkpoint files are corrupted/inaccessible
- **CRASH** if timestamps suggest non-sequential creation

### Tournament Execution
- **CRASH** if insufficient openings generated
- **CRASH** if strategy configuration is invalid
- **CRASH** if model loading fails
- **CRASH** if game execution fails

### Configuration Validation
- **CRASH** if conflicting arguments provided
- **CRASH** if required arguments missing
- **CRASH** if invalid parameter values

## Performance Considerations

### Memory Management
- Generate openings in batches to avoid memory issues
- Stream game results to disk rather than keeping in memory
- Use existing model caching for efficiency

### Computational Efficiency
- Sequential execution for now (parallelization future work)
- Reuse existing game execution infrastructure
- Optimize opening position generation

## Future Enhancements

### Potential Improvements
1. **Parallelization**: Run multiple matches simultaneously
2. **Resume functionality**: Restart interrupted tournaments
3. **Advanced brackets**: Swiss system, single elimination options
4. **Dynamic game counts**: Adjust based on match competitiveness
5. **Real-time monitoring**: Web interface for tournament progress

### Extensibility
- Plugin system for different tournament formats
- Custom participant types beyond model checkpoints
- Integration with external tournament systems

## Development and Validation Strategy

### Development Approach
- Start with debug output and manual testing
- Run with small checkpoint sets initially
- Add verbose logging for tournament progression
- Test with real checkpoint directories as development progresses

### Validation Points
- Checkpoint discovery with various directory structures
- Knockout bracket execution with mock participants
- Integration with existing tournament infrastructure
- Error handling with invalid inputs and edge cases

## Discussion

We aim to follow project's principles of failing fast, avoiding fallback mechanisms, and preferring clean abstractions over compatibility layers.

This design aims for clear separation of concerns between tournament orchestration, strategy execution, and checkpoint management.
