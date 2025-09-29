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

**Pairing Strategy**: Space out participants for more efficient tournaments:
- Pair first half with second half (1 vs N/2+1, 2 vs N/2+2, etc.)
- This ensures early checkpoints face later checkpoints (assuming sequential training)
- Avoids pitting very similar models against each other
- Example with 9 participants: 1v5, 2v6, 3v7, 4v8, 9 gets bye

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

## Implementation Status

### ✅ Completed Components

#### Core Architecture
- **`hex_ai/inference/knockout_tournament.py`** - Abstract knockout tournament class
  - Strategy-agnostic tournament execution
  - Configurable games per match with semifinal/final multipliers
  - Proper tournament state management and elimination tracking
  - Clean separation between tournament logic and strategy execution

- **`hex_ai/inference/checkpoint_discovery.py`** - Checkpoint discovery and ordering
  - Automatic discovery of checkpoints matching `epochN_miniJ.pt.gz` pattern
  - Sequential ordering based on file creation time
  - Validation of checkpoint ordering and completeness
  - Support for checkpoint ranges and latest N selection

- **`hex_ai/inference/two_stage_tournament.py`** - Orchestration layer
  - Coordinates knockout and round-robin stages
  - Handles checkpoint-to-participant conversion
  - Manages tournament configuration and execution flow
  - Provides tournament summary and results

#### CLI Integration
- **Extended `scripts/run_tournament.py`** with 2-stage tournament support
  - New arguments: `--knockout-dir`, `--knockout-config`, `--games-per-match`, `--top-k`, `--round-robin-games`
  - Backward compatibility with existing tournament functionality
  - Support for knockout-only tournaments (no round-robin participants)
  - Clean argument validation and error handling

#### Code Quality Improvements
- **Refactored main function** from ~200 lines to ~50 lines of clear orchestration
- **Extracted helper functions** for model parsing and strategy configuration
- **Eliminated inline imports** and complex repeated conditions
- **Added comprehensive error handling** without fallback mechanisms

### 🔄 In Progress / TODO

#### Match Execution Integration
- **Current Status**: Placeholder implementation with random winner assignment
- **Next Steps**: Integrate with existing game execution infrastructure
- **Key Integration Points**:
  - Connect to `hex_ai.inference.game_engine` for actual game execution
  - Integrate with opening position generation system
  - Use existing MCTS strategy execution from `hex_ai.inference.mcts`
  - Handle game result collection and match outcome determination

#### Round-Robin Stage Implementation
- **Current Status**: Placeholder ranking based on participant order
- **Next Steps**: Integrate with existing `hex_ai.inference.tournament.Tournament` class
- **Requirements**: 
  - Execute actual round-robin matches between knockout winners and additional participants
  - Generate proper tournament rankings based on match results
  - Support configurable games per round-robin match

#### Logging and Progress Reporting
- **Current Status**: Basic logging implemented
- **Next Steps**: Add streaming progress reporting and detailed match logging
- **Requirements**:
  - Real-time tournament progress updates
  - Match-by-match result logging
  - Tournament summary generation
  - Integration with existing tournament logging infrastructure

### 🚨 Implementation Discoveries & Notes

#### Unexpected Complications
1. **Complex Argument Validation**: The interaction between knockout-only tournaments and traditional tournaments required careful handling of optional arguments and validation logic.

2. **Strategy Configuration Complexity**: The existing strategy configuration system is quite complex with many parameters. Integration required careful mapping between the new tournament system and existing configuration patterns.

3. **File Path Handling**: Checkpoint discovery needed robust handling of different file naming patterns and validation of sequential ordering.

#### Key Design Decisions Made
1. **Helper Function Extraction**: Broke down the monolithic main function into focused helper functions (`parse_model_specifications`, `create_strategy_configurations`, `is_knockout_only_tournament`) for better maintainability.

2. **Placeholder Implementation Strategy**: Used clear placeholder implementations with detailed TODO comments rather than incomplete integrations, following the project's "fail fast" principle.

3. **Configuration Flexibility**: Made knockout configuration JSON-based to allow easy customization of MCTS parameters without code changes.

#### Notes for Future Developers

##### Integration Priorities
1. **Match Execution**: The highest priority is integrating the placeholder match execution with the existing game engine. This involves:
   - Understanding the existing `play_deterministic_game` infrastructure
   - Mapping `TournamentParticipant` objects to the existing strategy configuration system
   - Handling opening position generation and distribution

2. **Round-Robin Integration**: The existing `Tournament` class in `hex_ai.inference.tournament` should be adapted for the round-robin stage rather than creating a new implementation.

3. **Error Handling**: The current implementation follows the project's "fail fast" principle. Any integration should maintain this approach and avoid fallback mechanisms.

##### Code Organization
- **Separation of Concerns**: The current architecture cleanly separates tournament orchestration, checkpoint management, and strategy execution. Maintain this separation when adding new features.

- **Configuration Management**: The knockout configuration system is designed to be extensible. New parameters can be added to the JSON configuration without code changes.

- **Testing Strategy**: The current implementation uses placeholder random results for testing. When integrating with real game execution, start with small test cases and gradually scale up.

##### Performance Considerations
- **Checkpoint Discovery**: The current implementation loads all checkpoint metadata at startup. For very large checkpoint directories (1000+ files), consider lazy loading or pagination.

- **Memory Usage**: The tournament system maintains state for all participants and match results. Monitor memory usage for large tournaments.

- **Opening Generation**: The system generates openings separately for knockout and round-robin stages. Consider caching and reuse strategies for large tournaments.

##### Extension Points
- **Custom Tournament Formats**: The abstract `KnockoutTournament` class can be extended for different elimination formats (single elimination, Swiss system, etc.).

- **Participant Types**: The `TournamentParticipant` abstraction can be extended to support different types of participants beyond model checkpoints.

- **Result Formats**: The tournament result system is designed to be extensible for different output formats and analysis needs.

## Implementation Decisions

### Match Execution Integration
- **Approach**: Use existing `play_deterministic_game` function from `hex_ai.utils.deterministic_tournament_utils`
- **Strategy Mapping**: Map `TournamentParticipant` to existing `StrategyConfig` class using the pattern:
  ```python
  StrategyConfig(
      name=participant.name,
      strategy_type="mcts", 
      config=participant.strategy_config,
      model_path=participant.strategy_config["model_path"]
  )
  ```
- **Testing**: Use single-game matches initially, rely on `CURRENT_BEST_MODEL_DIR` configuration

### Opening Position Strategy
- **TRMPH Files**: Use same TRMPH files for both knockout and round-robin stages
- **Generation**: Set `target_count` to 1.1x required number of unique openings (fail fast if insufficient)
- **Caching**: No caching needed - generation is fast, keep implementation simple

### Progress Reporting
- **Terminal**: Simple match results (e.g., "epochXminiY beat epochAminiB")
- **Log Files**: Stream all games to .trmph files using existing utilities with proper headers
- **Metadata**: Be generous with logging (within few MB limit, no performance impact)
- **Real-time**: Follow design doc suggestions for progress updates
- **Format**: JSON for tournament summary

### Testing Strategy
- **Checkpoint Directory**: Use `CURRENT_BEST_MODEL_DIR` from `hex_ai/inference/model_config.py`
- **Initial Scale**: Start with 8 checkpoints, single-game tournaments
- **Round-robin Participants**: 
  - `checkpoints/hyperparameter_tuning/pipeline_20250922_071957/pipeline_sweep_exp0__99914b_20250922_072446/epoch11_mini15.pt.gz`
  - `checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0__99914b_20250917_192629/epoch7_mini105.pt.gz`

## Discussion

We aim to follow project's principles of failing fast, avoiding fallback mechanisms, and preferring clean abstractions over compatibility layers.

This design aims for clear separation of concerns between tournament orchestration, strategy execution, and checkpoint management.

The implementation successfully demonstrates these principles through clean abstractions, comprehensive error handling, and maintainable code structure. The remaining work focuses on integrating with existing game execution infrastructure while maintaining the established architectural patterns.
