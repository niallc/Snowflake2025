# Batch Optimization Plan for Self-Play Performance

## 🎯 **Goal: Reduce 500k games from 24 days to <4 days**

### **Current Performance Analysis**
- **Single game**: 4.97s (GPU)
- **500k games**: 24.1 days
- **Target**: <4 days (6x speedup needed)

### **🚨 Major Bottlenecks Identified**

#### **1. Individual Policy Calls (Primary Bottleneck)**
- **Problem**: Each tree node requires individual policy inference
- **Current**: 18 inference calls per move (1 + 5 + 12)
- **Solution**: Batch all policy calls into single inference

#### **2. Small Batch Sizes**
- **Current average**: 57.3 boards per batch
- **Optimal**: 400+ boards per batch
- **Throughput potential**: 917 → 6,492 boards/s

#### **3. No Cross-Game Batching**
- **Problem**: Each game processes independently
- **Solution**: Collect positions from multiple games (Phase 2)

## 🚀 **Optimization Strategy**

### **Phase 1: Single-Game Position Collection (Immediate Priority)**

#### **Current Flow:**
```
Move Generation:
1. Policy inference for current position (1 call)
2. Value inference for top 5 policy moves (5 individual calls)
3. Minimax search:
   - Policy calls for tree building (12 individual calls)
   - Value calls for leaf evaluation (already batched)
Total: 18 calls per move
```

#### **Optimized Flow:**
```
Move Generation:
1. Collect all positions needed for this move
2. Batch policy inference (1 call)
3. Batch value inference (1 call)
4. Build tree with cached results
Total: 2 calls per move
```

#### **Implementation Plan:**

**Step 1: Position Collection During Tree Building**
```python
class PositionCollector:
    def __init__(self, model):
        self.model = model
        self.policy_requests = []  # (board, callback) tuples
        self.value_requests = []   # (board, callback) tuples
    
    def request_policy(self, board, callback):
        """Request policy inference for a board."""
        self.policy_requests.append((board, callback))
    
    def request_value(self, board, callback):
        """Request value inference for a board."""
        self.value_requests.append((board, callback))
    
    def process_batches(self):
        """Process all collected requests in batches."""
        # Process policy requests
        if self.policy_requests:
            boards = [req[0] for req in self.policy_requests]
            policies, _ = self.model.batch_infer(boards)
            for (board, callback), policy in zip(self.policy_requests, policies):
                callback(policy)
            self.policy_requests.clear()
        
        # Process value requests
        if self.value_requests:
            boards = [req[0] for req in self.value_requests]
            _, values = self.model.batch_infer(boards)
            for (board, callback), value in zip(self.value_requests, values):
                callback(value)
            self.value_requests.clear()
```

**Step 2: Modify Tree Building**
```python
def build_search_tree_with_collection(root_state, model, widths, temperature, collector):
    """Build search tree while collecting positions for batch processing."""
    
    def build_node(state, depth, path):
        node = MinimaxSearchNode(state, depth, path)
        
        if state.game_over or depth >= len(widths):
            return node
        
        # Collect policy request instead of immediate inference
        k = widths[depth] if depth < len(widths) else 1
        
        def policy_callback(policy_logits):
            # Use policy to get top moves
            moves = get_topk_moves_from_policy(policy_logits, state, k, temperature)
            # Build children
            for move in moves:
                child_state = state.make_move(*move)
                child_path = path + [move]
                child_node = build_node(child_state, depth + 1, child_path)
                node.children[move] = child_node
        
        collector.request_policy(state.board, policy_callback)
        return node
    
    return build_node(root_state, 0, [])
```

**Step 3: Modify Move Generation**
```python
def generate_move_with_batching(state, model, search_widths, temperature):
    """Generate move using batched inference."""
    collector = PositionCollector(model)
    
    # Collect current position policy
    current_policy = None
    def current_policy_callback(policy):
        nonlocal current_policy
        current_policy = policy
    
    collector.request_policy(state.board, current_policy_callback)
    
    # Collect value requests for top 5 policy moves
    policy_move_values = []
    def value_callback_factory(index):
        def callback(value):
            policy_move_values[index] = value
        return callback
    
    # Get top 5 moves from current policy
    legal_moves = state.get_legal_moves()
    policy_top_moves = get_top_k_moves_with_probs(
        current_policy, legal_moves, state.board.shape[0], k=5, temperature=temperature
    )
    
    # Collect value requests
    for i, (move, prob) in enumerate(policy_top_moves):
        temp_state = state.make_move(*move)
        collector.request_value(temp_state.board, value_callback_factory(i))
    
    # Build search tree (collects more policy requests)
    root = build_search_tree_with_collection(state, model, search_widths, temperature, collector)
    
    # Process all batches
    collector.process_batches()
    
    # Now all callbacks have been called, proceed with minimax
    evaluate_leaf_nodes([root], model)  # Already batched
    minimax_backup(root)
    
    return root.best_move, root.value
```

#### **Bookkeeping Requirements:**
- **Minimal complexity**: Simple callback-based system
- **Thread safety**: Single-threaded within game, no synchronization needed
- **Error handling**: If callback fails, game continues with fallback
- **Memory management**: Automatic cleanup after each move

### **Phase 2: Cross-Game Batching (Future)**

#### **Architecture:**
```
Position Collector: Gather positions from all workers
Batch Processor: Process large batches (1000+ positions)
Result Distributor: Send results back to workers
```

#### **Implementation Plan:**
1. **Shared position queue** across all workers
2. **Batch processor thread** that waits for sufficient positions
3. **Result distribution** back to waiting workers
4. **Asynchronous processing** to overlap computation

## 📊 **Expected Performance Improvements**

### **Conservative Estimates:**
- **Single-game batching**: 6x speedup (18 → 3 calls)
- **Cross-game batching**: 2x additional speedup
- **Reduced transfers**: 1.5x speedup
- **Total**: 18x speedup

### **Aggressive Estimates:**
- **Single-game batching**: 9x speedup
- **Cross-game batching**: 3x additional speedup
- **Reduced transfers**: 2x speedup
- **Total**: 54x speedup

### **Target Timeline:**
- **Current**: 24.1 days for 500k games
- **Phase 1**: 4.0 days for 500k games
- **Phase 2**: 1.3 days for 500k games

## 🛠 **Implementation Steps**

### **Step 1: Position Collection (Week 1)**
1. Create `PositionCollector` class
2. Modify `build_search_tree` to collect positions
3. Modify move generation to use batched inference
4. Test with single game

### **Step 2: Cross-Game Batching (Week 2)**
1. Create shared position queue
2. Implement batch processor thread
3. Add result distribution system
4. Test with multiple games

### **Step 3: Advanced Optimizations (Week 3)**
1. Implement leaf node batching
2. Optimize memory management
3. Add pipeline processing
4. Performance testing and tuning

## 🎯 **Success Metrics**

### **Performance Targets:**
- **Throughput**: >5,000 boards/s
- **Batch size**: >1,000 positions per batch
- **Transfer overhead**: <10% of total time
- **500k games**: <4 days

### **Quality Targets:**
- **Game quality**: No degradation
- **Memory usage**: <16GB system RAM
- **Stability**: No crashes during long runs
- **Monitoring**: Real-time progress tracking

## 🚀 **Next Actions**

1. **Start with Phase 1** (single-game position collection)
2. **Profile each step** to measure improvements
3. **Iterate quickly** based on results
4. **Aim for 6x speedup** in first iteration

This plan should get us from 24 days to <4 days for 500k games! 🎯