# Data Processing and Model Training Plan

**Date:** 2024-12-19  
**Status:** Ready for execution

## 🎯 Current Status

### ✅ Completed
1. **TRMPH Processing Pipeline**: Fully tested and working
2. **Self-Play Data Preprocessing**: Script created and tested on real data
3. **Data Analysis**: Found 493,617 total games with 308,619 duplicates (62.5% duplicate rate)

### 📊 Data Summary
- **Input**: 45 .trmph files in `data/sf25/jul29/`
- **Total games**: 493,617
- **Unique games**: 184,998 (after deduplication)
- **Duplicates removed**: 308,619
- **Chunk size**: 20,000 games per file (recommended)

## 🚀 Next Steps

### Step 1: Process All Self-Play Data
```bash
# Process all data from jul29 directory
python scripts/preprocess_selfplay_data.py \
  --input-dir data/sf25/jul29 \
  --output-dir data/cleaned/jul29 \
  --chunk-size 20000

# Process other directories if they exist
python scripts/preprocess_selfplay_data.py \
  --input-dir data/sf25/jul29_2 \
  --output-dir data/cleaned/jul29_2 \
  --chunk-size 20000

python scripts/preprocess_selfplay_data.py \
  --input-dir data/sf25/jul29_3 \
  --output-dir data/cleaned/jul29_3 \
  --chunk-size 20000
```

**Expected Output**: ~9-10 chunks of 20,000 games each from jul29 directory

### Step 2: Convert to Training Format
```bash
# Process the cleaned chunks into training-ready format
python scripts/process_all_trmph.py \
  --data-dir data/cleaned/jul29 \
  --output-dir data/processed/jul29 \
  --position-selector all
```

**Expected Output**: 
- Individual processed files for each chunk
- Processing statistics
- NO combined dataset (removed to prevent memory crashes)

### Step 3: Use Individual Processed Files
```bash
# Individual processed files are ready for training
# No need to combine - this prevents memory crashes
# Training scripts should be updated to handle multiple files
```

### Step 4: Training with Multiple Files
```bash
# Training scripts need to be updated to handle multiple files
# Instead of one large file, use multiple smaller files
# This prevents memory crashes and allows for better memory management
```

## 📁 Directory Structure

```
data/
├── sf25/                          # Raw self-play data
│   ├── jul29/                     # Original files (45 files)
│   ├── jul29_2/                   # Additional data (if exists)
│   └── jul29_3/                   # Additional data (if exists)
├── cleaned/                       # Deduplicated and chunked data
│   ├── jul29/                     # ~10 chunks of 20k games each
│   ├── jul29_2/                   # (if exists)
│   └── jul29_3/                   # (if exists)
├── processed/                     # Training-ready format
│   ├── jul29/                     # Processed chunks
│   └── final_processed/           # Combined processed data
└── training_data/                 # Final training dataset
    └── shuffled_dataset.pkl.gz    # Ready for training
```

## 🔧 Training Pipeline

### Step 5: Train New Model
```bash
# Use the existing hyperparameter sweep script
python scripts/hyperparam_sweep.py \
  --data-file data/training_data/shuffled_dataset.pkl.gz \
  --output-dir checkpoints/new_model_training
```

## 📈 Expected Results

### Data Processing
- **Input**: ~493k games (with duplicates)
- **After deduplication**: ~185k unique games
- **After processing**: ~185k training examples
- **Final dataset**: ~185k shuffled training examples

### Training
- **Dataset size**: ~185k examples
- **Training time**: Depends on hyperparameters
- **Expected improvement**: Better model due to larger, cleaner dataset

## ⚠️ Important Notes

1. **Duplicate Rate**: 62.5% duplicates suggests temperature was too low in self-play
2. **File Sizes**: 20k games per chunk is manageable for processing
3. **Memory Usage**: Monitor memory during processing of large files
4. **Backup**: Keep original data files as backup

## 🧪 Testing

All scripts have comprehensive test suites:
```bash
# Test preprocessing script
python -m pytest tests/test_preprocess_selfplay_data.py -v

# Test TRMPH processing
python -m pytest tests/test_trmph_processor.py -v

# Test overall pipeline
python -m pytest tests/ -v
```

## 📝 Monitoring

- Check logs in `logs/` directory
- Monitor processing statistics in summary files
- Verify file sizes and counts at each step
- Test small samples before processing full dataset

## 🎯 Success Criteria

1. ✅ All data successfully deduplicated
2. ✅ All data converted to training format
3. ✅ Combined dataset created and shuffled
4. ✅ New model training started
5. ✅ Training pipeline running smoothly

---

**Next Action**: Run Step 1 to process the jul29 data directory 