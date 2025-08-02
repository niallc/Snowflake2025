# Value Network Debugging Plan

## Problem Statement

The neural network has been trained on self-play data from a 2018 network, with the policy network performing well but the value network underperforming significantly. Key observations:

1. **Policy-only play performs as well as value-augmented play**
2. **Using value network for 2-4 ply search makes performance worse**
3. **Need better instrumentation to understand why value network is failing**

## Current State Analysis

### Existing Infrastructure
- **Web UI** (`hex_ai/web/app.py`): Basic game interface with model inference
- **Tree Search** (`hex_ai/inference/fixed_tree_search.py`): Minimax search with policy/value integration
- **Manual Verification** (`scripts/deprecated/manual_verification_script.py`): Debugging utilities for tree search

### Current Web App Capabilities
- Basic game board display
- Model selection (2 different checkpoints)
- Search width and temperature controls
- Computer move generation
- Policy and value display (basic)

### Missing Debugging Features
- Detailed tree search visualization
- Move-by-move analysis
- Position construction tools
- Game sequence replay
- Undo functionality
- Verbose output controls

## Proposed Debugging Strategy

### Phase 1: Enhanced Instrumentation (High Priority)

#### 1.1 Verbose Output System
- **Goal**: Add comprehensive debugging output to web UI
- **Implementation**:
  - Add `verbose` flag to API endpoints
  - Create detailed HTML output section at bottom of page
  - Display tree search details similar to `print_all_terminal_nodes()`
  - Show top-k policy moves with probabilities
  - Display value network assessments for current position

#### 1.2 Tree Search Visualization
- **Goal**: Understand how value network influences decisions
- **Implementation**:
  - Integrate `print_all_terminal_nodes()` logic into web UI
  - Display search tree structure with values
  - Show backup values at each node
  - Highlight chosen moves and reasoning

#### 1.3 Policy vs Value Analysis
- **Goal**: Compare policy-only vs value-augmented decisions
- **Implementation**:
  - Side-by-side comparison of move choices
  - Display policy probabilities for all legal moves
  - Show value network's position assessment
  - Track decision conflicts between policy and value

### Phase 2: Position Control Tools (Medium Priority)

#### 2.1 Manual Position Construction
- **Goal**: Test network on specific positions
- **Implementation**:
  - Toggle computer moves on/off
  - Allow manual move placement
  - Position validation and error handling
  - Save/load position functionality

#### 2.2 Move History and Undo
- **Goal**: Analyze game progression
- **Implementation**:
  - Track move history
  - Undo last move functionality
  - Display move sequence
  - Export game in TRMPH format

#### 2.3 Game Sequence Import
- **Goal**: Test network on known game sequences
- **Implementation**:
  - Paste TRMPH sequence input
  - Step through imported games
  - Compare network predictions with actual outcomes
  - Batch analysis of multiple games

### Phase 3: Advanced Analysis Tools (Lower Priority)

#### 3.1 Value Network Diagnostics
- **Goal**: Understand value network behavior patterns
- **Implementation**:
  - Value distribution analysis across positions
  - Correlation between value predictions and game outcomes
  - Value network confidence metrics
  - Position difficulty assessment

#### 3.2 Comparative Analysis
- **Goal**: Compare different model checkpoints
- **Implementation**:
  - Side-by-side model comparison
  - A/B testing interface
  - Performance metrics tracking
  - Model ensemble analysis

## Implementation Plan

### Immediate Next Steps (Week 1)

1. **Add Verbose Output to Web App**
   - Modify `/api/move` and `/api/computer_move` endpoints to accept `verbose` parameter
   - Create HTML output section in web UI
   - Integrate tree search debugging output
   - Display top-k policy moves and value assessments

2. **Enhanced Tree Search Display**
   - Port `print_all_terminal_nodes()` logic to web format
   - Add search tree visualization
   - Show backup values and decision reasoning
   - Color-code positive/negative values

3. **Basic Position Control**
   - Add "Computer Off" toggle
   - Implement manual move placement
   - Add undo functionality
   - Basic position validation

### Medium Term (Week 2-3)

1. **Game Sequence Import**
   - TRMPH sequence parser
   - Step-through interface
   - Move-by-move analysis
   - Export functionality

2. **Advanced Diagnostics**
   - Value network behavior analysis
   - Policy-value conflict detection
   - Performance metrics dashboard
   - Model comparison tools

### Long Term (Week 4+)

1. **Comprehensive Testing Framework**
   - Automated position testing
   - Performance regression detection
   - Model validation suite
   - Training data analysis

## Technical Implementation Details

### API Enhancements

```python
# New endpoint parameters
{
    "verbose": true,  # Enable detailed output
    "computer_off": false,  # Disable computer moves
    "show_tree": true,  # Display search tree
    "max_tree_depth": 3  # Limit tree display depth
}
```

### Frontend Enhancements

```html
<!-- New UI sections -->
<div id="debug-output" class="debug-panel">
    <h3>Debug Information</h3>
    <div id="tree-visualization"></div>
    <div id="policy-analysis"></div>
    <div id="value-analysis"></div>
</div>

<div id="position-controls" class="control-panel">
    <button id="computer-toggle">Computer: ON</button>
    <button id="undo-btn">Undo</button>
    <button id="clear-btn">Clear Board</button>
</div>
```

### Data Structures

```python
# Enhanced response format
{
    "success": True,
    "new_trmph": "...",
    "board": [...],
    "debug_info": {
        "tree_search": {
            "nodes": [...],
            "terminal_values": [...],
            "backup_values": [...]
        },
        "policy_analysis": {
            "top_moves": [...],
            "probabilities": [...]
        },
        "value_analysis": {
            "position_value": 0.75,
            "confidence": 0.8
        }
    }
}
```

## Success Metrics

### Short Term
- [ ] Verbose output working in web UI
- [ ] Tree search visualization functional
- [ ] Manual position construction possible
- [ ] Undo functionality implemented

### Medium Term
- [ ] Game sequence import working
- [ ] Policy vs value analysis clear
- [ ] Value network behavior patterns identified
- [ ] Performance regression detection

### Long Term
- [ ] Automated testing framework
- [ ] Value network performance improved
- [ ] Training data quality validated
- [ ] Model comparison tools complete

## Risk Assessment

### Technical Risks
- **Performance**: Verbose output may slow down web UI
- **Complexity**: Tree visualization may be difficult to implement
- **Compatibility**: Changes may break existing functionality

### Mitigation Strategies
- **Progressive Enhancement**: Add features incrementally
- **Performance Monitoring**: Track response times
- **Backward Compatibility**: Maintain existing API endpoints
- **Testing**: Comprehensive testing at each stage

## Recommended Next Steps (Immediate Action Items)

Based on the analysis, here are the **simplest and most promising** next steps to implement:

### 1. Add Verbose Output to Existing Endpoints (Easiest - 2-3 hours)
**Why**: Provides immediate visibility into network behavior without major UI changes
**Implementation**:
- Add `verbose` parameter to `/api/move` and `/api/computer_move` endpoints
- Return additional debug information in JSON response
- Display this information in a simple text area at bottom of web UI

**Expected Value**: Immediate insights into policy vs value decisions

### 2. Add "Computer Off" Toggle (Easy - 1-2 hours)
**Why**: Enables manual position construction for testing specific scenarios
**Implementation**:
- Add checkbox/button to disable computer moves
- Modify move logic to skip computer move when disabled
- Allow manual move placement without automatic response

**Expected Value**: Ability to test network on specific positions

### 3. Display Top-K Policy Moves (Easy - 1-2 hours)
**Why**: Shows what the policy network thinks are good moves vs what value network chooses
**Implementation**:
- Extract top-k moves from policy probabilities
- Display in simple list format
- Compare with actual chosen move

**Expected Value**: Clear comparison between policy and value-based decisions

### 4. Integrate Tree Search Debug Output (Medium - 3-4 hours)
**Why**: Shows exactly how value network influences decisions in tree search
**Implementation**:
- Port `print_all_terminal_nodes()` logic to web format
- Display terminal node values and backup reasoning
- Show search tree structure when verbose mode enabled

**Expected Value**: Deep understanding of value network's role in decision making

### 5. Add Undo Functionality (Medium - 2-3 hours)
**Why**: Enables iterative testing and analysis of move sequences
**Implementation**:
- Track move history in frontend
- Add undo button that reverts to previous state
- Maintain game state consistency

**Expected Value**: Better debugging workflow for analyzing move sequences

## Priority Order for Implementation

1. **Verbose Output** - Highest impact, lowest effort
2. **Computer Off Toggle** - Enables manual testing
3. **Top-K Policy Display** - Provides immediate policy vs value comparison
4. **Tree Search Debug** - Deep insights into decision making
5. **Undo Functionality** - Improves debugging workflow

## Conclusion

This debugging plan provides a systematic approach to understanding and improving the value network performance. The immediate focus should be on enhanced instrumentation to gain insights into why the value network is underperforming, followed by tools to test and validate improvements.

The plan prioritizes actionable insights over complex analysis, with the goal of quickly identifying the root causes of value network issues and providing the tools needed to validate potential solutions.

**Recommended starting point**: Implement the verbose output system first, as it provides the most value with the least effort and will immediately help identify whether the issue is with the value network itself or how it's being used in the search algorithm. 