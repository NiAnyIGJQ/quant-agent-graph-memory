# LSTM - Vector Database Component

This folder contains the independent Vector Database component for the BTC_ACE trading system.

## Overview

The Vector Database provides a complete solution for:
- Converting 30-period market snapshots into fixed-dimension embeddings (128-D)
- Storing and retrieving historical market patterns
- Finding similar historical market conditions via cosine similarity
- Supporting both single and batch operations
- Checkpoint save/load for persistence

## Files in This Folder

### Core Implementation
- **vector_database.py** - Main VectorDatabase class and supporting models
  - `VectorDatabase`: Primary interface
  - `CompositeModel`: LSTM-based encoder with attention
  - `Attention`: Attention mechanism module

### Documentation
- **VECTOR_DATABASE_DOCS.md** - Comprehensive API documentation
  - Full API reference
  - Data format specifications
  - Usage examples
  - Performance characteristics
  - Integration notes

- **QUICK_START.md** - Quick reference guide
  - 5-minute introduction
  - Common use cases
  - API cheat sheet
  - Example workflows

### Examples & Testing
- **examples_usage.py** - Practical usage scenarios
  - Loading historical data from CSV
  - Real-time pattern matching
  - Threshold-based filtering
  - Comparative analysis
  - Database updates and checkpoints

- **test_vector_database.py** - Comprehensive test suite
  - Initialization tests
  - Single/batch vector generation
  - Similarity search
  - Threshold filtering
  - Database clearing
  - Checkpoint save/load

### Models & Data
- **models/** - Folder containing pre-trained models
  - btc_lstm_attn_model_2024-06-01.pth - Pre-trained LSTM attention model

## Quick Start

### Installation
No installation needed. Import directly:
```python
from ace_trading.LSTM.vector_database import VectorDatabase
```

### Basic Usage
```python
import numpy as np
import pandas as pd
from vector_database import VectorDatabase

# Initialize
db = VectorDatabase(
    model_path=r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth"
)

# Add market data (30 periods of OHLCV)
ohlcv = np.random.randn(30, 5) * 1000 + 40000
timestamp = pd.Timestamp('2024-06-01 10:00:00')
db.add_vector(ohlcv, timestamp)

# Find similar patterns
matches = db.query_similar(ohlcv, top_k=5)
for match in matches:
    print(f"{match['timestamp']}: {match['similarity']:.4f}")
```

## Key Features

### 1. Efficient Vectorization
- LSTM + Attention mechanism for semantic market pattern encoding
- 128-dimensional embeddings capture market structure
- Batch processing for high throughput

### 2. Flexible Querying
```python
# Top-k similarity search
matches = db.query_similar(market_data, top_k=10)

# Threshold-based filtering
high_conf = db.query_similar(market_data, threshold=0.95)

# All with timestamps for historical lookup
```

### 3. Data Persistence
```python
# Save
db.save_checkpoint('my_database')

# Load
db_restored = VectorDatabase(MODEL_PATH)
db_restored.load_checkpoint('my_database')
```

### 4. GPU Acceleration
- Automatic CUDA detection and fallback to CPU
- Batch inference for optimal throughput
- configurable device selection

## Data Format

### Input (OHLCV)
- **Shape**: (30, 5) for single, (N, 30, 5) for batch
- **Columns**: [Open, High, Low, Close, Volume]
- **Timestamp**: pd.Timestamp marking end of 30-period window

### Output (Embedding)
- **Shape**: (128,) for single, (N, 128) for batch
- **Type**: float32
- **Normalized**: Z-score normalized per window

### Query Results
```python
[
    {
        'timestamp': pd.Timestamp('2024-05-15 10:00:00'),
        'similarity': 0.9234,  # 0-1 cosine similarity
        'rank': 1
    },
    ...
]
```

## API Reference (Condensed)

| Method | Purpose |
|--------|---------|
| `add_vector(ohlcv, timestamp)` | Add single market embedding |
| `add_vectors_batch(batch, timestamps, batch_size)` | Add multiple embeddings |
| `query_similar(ohlcv, top_k, threshold)` | Find similar patterns |
| `clear_vectors()` | Reset database |
| `get_statistics()` | Get database info |
| `save_checkpoint(path)` | Persist to disk |
| `load_checkpoint(path)` | Load from disk |

See **VECTOR_DATABASE_DOCS.md** for full documentation.

## Performance

### Memory
- ~512 bytes per vector (128 float32 values)
- ~32 bytes per timestamp
- 50k patterns: ~26 MB total

### Speed (GPU)
- Single vector: 5-10 ms
- Batch (1k vectors): 100-150 ms
- Query (50k DB): 2-3 ms

## Testing

Run the complete test suite:
```bash
python test_vector_database.py
```

Expected output:
```
Test Suite: Vector Database Component

TEST 1: Initialization          [PASS]
TEST 2: Single Vector Generation [PASS]
TEST 3: Batch Vector Generation  [PASS]
TEST 4: Similarity Search        [PASS]
TEST 5: Query with Threshold     [PASS]
TEST 6: Clear Database           [PASS]
TEST 7: Checkpoint Save/Load     [PASS]

ALL TESTS PASSED
```

## Example Scenarios

Run practical examples:
```bash
python examples_usage.py
```

Covers:
1. Loading historical CSV data
2. Real-time pattern matching
3. Threshold-based filtering
4. Comparative analysis
5. Incremental database updates
6. Checkpoint management

## Integration Status

**Current Status**: Independent component (not yet integrated with backtest)

**Planned Integration**:
- Load historical market data during backtest initialization
- Add new market windows during backtest execution
- Query similar patterns to inform trading decisions
- Optional: Use historical pattern outcomes for strategy refinement

**To Integrate Later**:
```python
# In backtest engine
from ace_trading.LSTM.vector_database import VectorDatabase

db = VectorDatabase(MODEL_PATH)
# Load historical data...
# Query during trading...
```

## Dependencies

```
torch >= 1.9.0
numpy >= 1.19.0
pandas >= 1.2.0
scikit-learn >= 0.24.0
```

## Troubleshooting

### Error: "Model file not found"
Check MODEL_PATH points to: `models/btc_lstm_attn_model_2024-06-01.pth`

### Error: "Expected 30 periods, got X"
Ensure OHLCV input has exactly 30 rows of market data

### Warning: "Vector database is empty"
Add vectors before querying. Use `add_vector()` or `add_vectors_batch()`

### CUDA out of memory
Reduce `batch_size` in `add_vectors_batch()` or use CPU device

## Next Steps

1. Read full docs: `VECTOR_DATABASE_DOCS.md`
2. Try quick start: `QUICK_START.md`
3. Run examples: `python examples_usage.py`
4. Run tests: `python test_vector_database.py`
5. Integrate when ready (see integration guide in main docs)

## Questions?

- Check VECTOR_DATABASE_DOCS.md for detailed explanations
- Review examples_usage.py for practical patterns
- Run test_vector_database.py to validate your setup

## Version

- Version: 1.0
- Created: 2025-12-21
- Status: Production Ready
- Integration: Pending (planned for future backtest enhancement)
