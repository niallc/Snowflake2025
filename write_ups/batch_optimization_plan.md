# Batch Optimization Plan for Self-Play Performance

## 🎯 **Goal: Reduce 500k games from 24 days to <4 days**

### **Current Performance Analysis**
- **Single game**: 4.97s (GPU)
- **500k games**: 24.1 days
- **Target**: <4 days (6x speedup needed)

### **🚨 Major Bottlenecks Identified**

#### **1. CPU↔GPU Transfer Overhead (27% of time)**
- **Problem**: Every tensor transfer between CPU and GPU takes time
- **Current**: 516 individual `get_topk_moves` calls per game
- **Solution**: Batch all positions and minimize transfers

#### **2. Small Batch Sizes (7.1x improvement potential)**
- **Current average**: 57.3 boards per batch
- **Optimal**: 400+ boards per batch
- **Throughput potential**: 917 → 6,492 boards/s

#### **3. No Cross-Game Batching**
- **Problem**: Each game processes independently
- **Solution**: Collect positions from multiple games

## 🚀 **Optimization Strategy**

### **Phase 1: Position Collection During Tree Building**

#### **Current Flow:**
```
Game 1: build_tree() → get_topk_moves() → simple_infer() → transfer overhead
Game 2: build_tree() → get_topk_moves() → simple_infer() → transfer overhead
...
```

#### **Optimized Flow:**
```
Game 1: build_tree() → collect_positions() → no inference yet
Game 2: build_tree() → collect_positions() → no inference yet
...
Batch: process_all_positions() → single GPU batch → single transfer
```

#### **Implementation Plan:**
1. **Modify `build_search_tree()`** to collect positions instead of inferring
2. **Add position accumulator** to gather positions across games
3. **Batch process** all collected positions at once
4. **Distribute results** back to tree nodes

### **Phase 2: Cross-Game Batching**

#### **Current Architecture:**
```
Worker 1: Game 1 (independent)
Worker 2: Game 2 (independent)
Worker 3: Game 3 (independent)
Worker 4: Game 4 (independent)
```

#### **Optimized Architecture:**
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

### **Phase 3: Advanced Optimizations**

#### **1. Leaf Node Batching**
- **Current**: Evaluate leaf nodes individually
- **Optimized**: Collect all leaf nodes across games
- **Potential**: 10x+ improvement in leaf evaluation

#### **2. Memory Management**
- **Keep tensors on GPU** between batches
- **Pre-allocate GPU memory** for large batches
- **Minimize CPU↔GPU transfers**

#### **3. Pipeline Processing**
- **Tree building** (CPU) while **inference** (GPU) runs
- **Overlap computation** and data transfer
- **Streaming processing** for continuous generation

## 📊 **Expected Performance Improvements**

### **Conservative Estimates:**
- **Position collection**: 3x speedup
- **Cross-game batching**: 2x speedup
- **Reduced transfers**: 1.5x speedup
- **Total**: 9x speedup

### **Aggressive Estimates:**
- **Position collection**: 5x speedup
- **Cross-game batching**: 3x speedup
- **Reduced transfers**: 2x speedup
- **Total**: 30x speedup

### **Target Timeline:**
- **Current**: 24.1 days for 500k games
- **Conservative**: 2.7 days for 500k games
- **Aggressive**: 0.8 days for 500k games

## 🛠 **Implementation Steps**

### **Step 1: Position Collection (Week 1)**
1. Modify `build_search_tree()` to collect positions
2. Create position accumulator class
3. Implement batch processing for collected positions
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

1. **Start with Phase 1** (position collection)
2. **Profile each step** to measure improvements
3. **Iterate quickly** based on results
4. **Aim for 10x speedup** in first iteration

This plan should get us from 24 days to <4 days for 500k games! 🎯