# Data Pipeline Cleanup Plan

## Executive Summary

The Hex AI data pipeline has evolved through multiple iterations, resulting in a complex system with overlapping functionality, inconsistent interfaces, and brittle error handling. This document analyzes the current state and proposes a comprehensive refactoring plan to simplify and modernize the architecture.

## Current State Analysis

### 🏗️ **Dataset Class Hierarchy**

The codebase contains **4 different dataset classes** with overlapping functionality:

```
torch.utils.data.Dataset
├── NewProcessedDataset (DEPRECATED)
│   └── AugmentedProcessedDataset (DEPRECATED)
└── StreamingProcessedDataset
    └── StreamingAugmentedProcessedDataset
```

**Problems:**
- **Deprecated classes still in use**: `NewProcessedDataset` and `AugmentedProcessedDataset` are marked as deprecated but still used in some code paths
- **Inconsistent inheritance**: Some classes inherit from deprecated parents
- **Duplicate functionality**: Both streaming and non-streaming versions implement similar logic
- **Complex inheritance chains**: `StreamingAugmentedProcessedDataset` inherits from `StreamingProcessedDataset` but overrides many methods

### 🔄 **Data Flow Complexity**

#### Current Data Flow
```
Raw .trmph files
    ↓
Processed .pkl.gz files (2-channel format)
    ↓
Dataset.__getitem__()
    ↓
get_player_to_move_from_board() + error handling
    ↓
3-channel board creation (blue, red, player-to-move)
    ↓
Data augmentation (4x examples)
    ↓
augmented_collate_fn() (flattens 4 examples)
    ↓
DataLoader batching
    ↓
Training loop
```

**Problems:**
- **Multiple transformation steps**: Data goes through 4+ transformations before reaching the model
- **Error handling scattered**: Error tracking logic is duplicated across multiple classes
- **Inconsistent interfaces**: Some functions expect 2-channel, others expect 3-channel format
- **Complex augmentation**: Augmentation logic is embedded in dataset classes rather than being composable

### 🚨 **Error Handling Complexity**

#### Current Error Handling System
- **`BoardStateErrorTracker`**: Global singleton for tracking board validation errors
- **Error context management**: Manual setting of `_current_file` and `_current_sample` attributes
- **Graceful degradation**: Invalid boards return default values instead of failing
- **Error logging**: Complex logging system with file output and console indicators

**Problems:**
- **Global state**: Error tracker is a global singleton, making testing difficult
- **Manual context management**: Error tracking requires manual setting of context
- **Inconsistent error handling**: Some code paths use error tracker, others don't
- **Complex error recovery**: Error handling logic is scattered across multiple classes

### 📊 **Data Format Inconsistencies**

#### Multiple Board Formats
1. **2-channel format**: `(2, 13, 13)` - blue and red channels
2. **3-channel format**: `(3, 13, 13)` - blue, red, and player-to-move channels
3. **N×N format**: `(13, 13)` - single channel with values 0/1/2

**Problems:**
- **Format conversion overhead**: Constant conversion between formats
- **Inconsistent APIs**: Different functions expect different formats
- **Player-to-move computation**: Computed on-the-fly instead of being stored
- **Memory inefficiency**: Multiple format conversions waste memory

### 🔧 **Configuration and Parameter Management**

#### Current Configuration Issues
- **Hardcoded parameters**: Many parameters are hardcoded in dataset constructors
- **Inconsistent parameter names**: `chunk_size` vs `max_examples` vs `max_examples_per_split`
- **Parameter inheritance**: Parameters passed through multiple inheritance levels
- **Configuration scattering**: Configuration logic spread across multiple files

## 🎯 **Proposed Refactoring Plan**

### Phase 1: Consolidate Dataset Classes

#### 1.1 Create Unified Dataset Interface
```python
class HexDataset(torch.utils.data.Dataset):
    """Unified dataset interface for Hex AI training data."""
    
    def __init__(self, 
                 data_files: List[Path],
                 config: DatasetConfig):
        """
        Initialize dataset with configuration.
        
        Args:
            data_files: List of data file paths
            config: Dataset configuration object
        """
        self.config = config
        self.data_files = data_files
        self._setup_data_loading()
    
    def _setup_data_loading(self):
        """Setup data loading strategy based on configuration."""
        if self.config.streaming:
            self._setup_streaming()
        else:
            self._setup_in_memory()
    
    def __getitem__(self, idx):
        """Get training example with optional augmentation."""
        example = self._load_example(idx)
        
        if self.config.augmentation.enabled:
            example = self._apply_augmentation(example)
        
        return self._format_for_training(example)
```

#### 1.2 Configuration-Driven Design
```python
@dataclass
class DatasetConfig:
    """Configuration for dataset behavior."""
    
    # Data loading
    streaming: bool = True
    chunk_size: int = 100000
    max_examples: Optional[int] = None
    
    # Augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Error handling
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    
    # Format
    output_format: BoardFormat = BoardFormat.THREE_CHANNEL

@dataclass
class AugmentationConfig:
    enabled: bool = True
    probability: float = 1.0
    skip_empty_boards: bool = True

@dataclass
class ErrorHandlingConfig:
    max_errors: int = 5
    max_error_rate: float = 0.05
    fail_fast: bool = False
```

### Phase 2: Simplify Data Flow

#### 2.1 Create Data Transformation Pipeline
```python
class DataTransformPipeline:
    """Composable data transformation pipeline."""
    
    def __init__(self, transforms: List[DataTransform]):
        self.transforms = transforms
    
    def __call__(self, data: TrainingExample) -> TrainingExample:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            data = transform(data)
        return data

class DataTransform(ABC):
    """Base class for data transformations."""
    
    @abstractmethod
    def __call__(self, data: TrainingExample) -> TrainingExample:
        pass

class PlayerToMoveTransform(DataTransform):
    """Add player-to-move channel to board."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler
    
    def __call__(self, data: TrainingExample) -> TrainingExample:
        board_2ch = data.board
        player_to_move = self._compute_player_to_move(board_2ch)
        board_3ch = self._add_player_channel(board_2ch, player_to_move)
        return data.with_board(board_3ch)

class AugmentationTransform(DataTransform):
    """Apply data augmentation."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def __call__(self, data: TrainingExample) -> List[TrainingExample]:
        if not self.config.enabled:
            return [data]
        
        if self.config.skip_empty_boards and self._is_empty_board(data.board):
            return [data]
        
        return self._create_augmented_examples(data)
```

#### 2.2 Unified Training Example Format
```python
@dataclass
class TrainingExample:
    """Unified training example format."""
    
    board: np.ndarray  # Always 3-channel format
    policy: np.ndarray  # (169,) policy target
    value: float  # Value target
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def with_board(self, board: np.ndarray) -> 'TrainingExample':
        """Create new example with different board."""
        return TrainingExample(
            board=board,
            policy=self.policy,
            value=self.value,
            metadata=self.metadata
        )
```

### Phase 3: Improve Error Handling

#### 3.1 Context-Aware Error Handling
```python
class ErrorHandler:
    """Context-aware error handler."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.error_count = 0
        self.total_count = 0
    
    @contextmanager
    def context(self, file_info: str, sample_info: str):
        """Context manager for error tracking."""
        self._current_context = {
            'file': file_info,
            'sample': sample_info,
            'timestamp': datetime.now()
        }
        try:
            yield self
        finally:
            self._current_context = None
    
    def handle_error(self, error: Exception, data: Optional[Any] = None):
        """Handle an error with current context."""
        self.error_count += 1
        self.total_count += 1
        
        error_record = {
            'error': error,
            'context': self._current_context,
            'data': data,
            'count': self.error_count
        }
        
        self._log_error(error_record)
        self._check_thresholds()
        
        if self.config.fail_fast:
            raise error
```

#### 3.2 Simplified Error Recovery
```python
class GracefulErrorHandler(ErrorHandler):
    """Error handler that returns default values instead of failing."""
    
    def __init__(self, config: ErrorHandlingConfig, defaults: Dict[str, Any]):
        super().__init__(config)
        self.defaults = defaults
    
    def handle_error(self, error: Exception, data: Optional[Any] = None):
        """Handle error by returning default values."""
        super().handle_error(error, data)
        
        # Return appropriate default based on error type
        if isinstance(error, BoardStateError):
            return self.defaults['board_state']
        elif isinstance(error, PolicyError):
            return self.defaults['policy']
        else:
            return self.defaults['general']
```

### Phase 4: Standardize Data Formats

#### 4.1 Single Source of Truth
```python
class BoardFormat(Enum):
    """Supported board formats."""
    TWO_CHANNEL = "2ch"  # (2, N, N) - blue, red
    THREE_CHANNEL = "3ch"  # (3, N, N) - blue, red, player-to-move
    SINGLE_CHANNEL = "1ch"  # (N, N) - 0=empty, 1=blue, 2=red

class BoardConverter:
    """Convert between board formats."""
    
    @staticmethod
    def to_three_channel(board: np.ndarray, player_to_move: int) -> np.ndarray:
        """Convert any format to 3-channel."""
        if board.shape[0] == 3:
            return board
        elif board.shape[0] == 2:
            player_channel = np.full((board.shape[1], board.shape[2]), 
                                   float(player_to_move), dtype=np.float32)
            return np.concatenate([board, player_channel[None, ...]], axis=0)
        else:
            # Single channel format
            board_2ch = np.zeros((2, board.shape[0], board.shape[1]), dtype=np.float32)
            board_2ch[0] = (board == 1).astype(np.float32)  # Blue
            board_2ch[1] = (board == 2).astype(np.float32)  # Red
            return BoardConverter.to_three_channel(board_2ch, player_to_move)
```

#### 4.2 Store Player-to-Move in Data
```python
# Modify data processing to include player-to-move
def process_game_to_examples(trmph_text: str, winner: str) -> List[TrainingExample]:
    """Process a game into training examples with player-to-move included."""
    examples = []
    moves = parse_trmph_moves(trmph_text)
    
    for i in range(len(moves) + 1):
        board_2ch = create_board_from_moves(moves[:i])
        player_to_move = i % 2  # Blue starts, alternating
        board_3ch = BoardConverter.to_three_channel(board_2ch, player_to_move)
        
        policy = create_policy_target(moves[i]) if i < len(moves) else None
        value = 1.0 if winner == "1" else 0.0
        
        examples.append(TrainingExample(
            board=board_3ch,
            policy=policy or np.zeros(169, dtype=np.float32),
            value=value
        ))
    
    return examples
```

### Phase 5: Configuration Management

#### 5.1 Centralized Configuration
```python
@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Training configuration
    training: TrainingParams = field(default_factory=TrainingParams)
    
    # Augmentation configuration
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Error handling configuration
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)

def load_config(config_path: Path) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)
```

#### 5.2 Configuration Validation
```python
def validate_config(config: TrainingConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Validate data configuration
    if config.data.max_examples and config.data.max_examples <= 0:
        issues.append("max_examples must be positive")
    
    # Validate augmentation configuration
    if not 0 <= config.augmentation.probability <= 1:
        issues.append("augmentation probability must be between 0 and 1")
    
    # Validate error handling configuration
    if not 0 <= config.error_handling.max_error_rate <= 1:
        issues.append("max_error_rate must be between 0 and 1")
    
    return issues
```

## 🚀 **Implementation Strategy**

### Phase 1: Foundation (Week 1-2)
1. **Create new dataset interface** (`HexDataset`)
2. **Implement configuration classes**
3. **Create basic data transformation pipeline**
4. **Write comprehensive tests**

### Phase 2: Migration (Week 3-4)
1. **Migrate existing code to use new interface**
2. **Update training scripts**
3. **Test with existing data**
4. **Performance benchmarking**

### Phase 3: Optimization (Week 5-6)
1. **Optimize data loading performance**
2. **Implement caching where beneficial**
3. **Add monitoring and metrics**
4. **Documentation and examples**

### Phase 4: Cleanup (Week 7-8)
1. **Remove deprecated classes**
2. **Clean up unused code**
3. **Update documentation**
4. **Final testing and validation**

## 📈 **Expected Benefits**

### Immediate Benefits
- **Reduced complexity**: Single dataset class instead of 4
- **Better error handling**: Centralized, context-aware error management
- **Consistent interfaces**: All code uses same data format
- **Easier testing**: Composable components with clear interfaces

### Long-term Benefits
- **Easier maintenance**: Clear separation of concerns
- **Better performance**: Optimized data flow and caching
- **Extensibility**: Easy to add new transformations or formats
- **Reliability**: Better error handling and validation

### Developer Experience
- **Clearer APIs**: Consistent method signatures and return types
- **Better debugging**: Context-aware error messages
- **Easier configuration**: Single configuration file for all settings
- **Comprehensive testing**: Unit tests for all components

## 🎯 **Success Metrics**

### Code Quality
- **Reduced complexity**: 50% reduction in cyclomatic complexity
- **Better test coverage**: >90% test coverage for new components
- **Fewer bugs**: Reduced number of data-related issues

### Performance
- **Faster training**: 20% improvement in data loading speed
- **Lower memory usage**: 30% reduction in peak memory usage
- **Better scalability**: Support for larger datasets

### Maintainability
- **Easier onboarding**: New developers can understand data flow in <1 day
- **Faster debugging**: Error resolution time reduced by 50%
- **Easier feature addition**: New augmentations or formats added in <1 day

## 🚨 **Risks and Mitigation**

### Risks
1. **Breaking changes**: Migration might break existing code
2. **Performance regression**: New abstraction layers might slow things down
3. **Data format changes**: Changes to stored data format

### Mitigation
1. **Gradual migration**: Keep old interfaces working during transition
2. **Performance testing**: Benchmark at each step
3. **Data compatibility**: Maintain backward compatibility for stored data
4. **Comprehensive testing**: Test all code paths before deployment

## 📋 **Next Steps**

1. **Review and approve this plan**
2. **Set up development environment for new components**
3. **Create detailed implementation plan for Phase 1**
4. **Begin implementation with foundation components**
5. **Regular progress reviews and adjustments**

---

*This document represents a comprehensive plan to modernize and simplify the Hex AI data pipeline. The proposed changes will result in a more maintainable, performant, and reliable system while preserving all existing functionality.* 