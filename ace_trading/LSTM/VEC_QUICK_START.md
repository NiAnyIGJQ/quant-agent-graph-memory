# Vector Database - Quick Start Guide

## Installation

No special installation needed. Just import the module:

```python
from ace_trading.LSTM.vector_database import VectorDatabase
```

## 5-Minute Quick Start

### Step 1: Initialize
```python
import numpy as np
import pandas as pd
from ace_trading.LSTM.vector_database import VectorDatabase

# Create database instance
db = VectorDatabase(
    model_path=r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth"
)
```

### Step 2: Add Data
```python
# Example: Add a single 30-period market snapshot
ohlcv = np.random.randn(30, 5) * 1000 + 40000  # Mock data
timestamp = pd.Timestamp('2024-06-01 10:00:00')
embedding = db.add_vector(ohlcv, timestamp)

print(f"Vector generated: shape {embedding.shape}")
```

### Step 3: Query Similar Patterns
```python
# Find similar historical patterns
current_market = np.random.randn(30, 5) * 1000 + 40000
matches = db.query_similar(current_market, top_k=3)

for match in matches:
    print(f"{match['timestamp']}: {match['similarity']:.4f}")
```

### Step 4: Save/Load
```python
# Save database
db.save_checkpoint('my_database')

# Load later
db_restored = VectorDatabase(model_path=MODEL_PATH)
db_restored.load_checkpoint('my_database')
```

## Common Use Cases

### Use Case 1: Find Historical Precedents
```python
# What happened last time the market looked like this?
current_pattern = get_last_30_periods()
similar_days = db.query_similar(current_pattern, top_k=5)

print("Similar historical days:")
for match in similar_days:
    print(f"  - {match['timestamp'].date()}")
    # Then analyze what happened after each date
```

### Use Case 2: Pre-load Historical Database
```python
# Build database from historical CSV
df = pd.read_csv('history.csv')
windows = create_30period_windows(df)
timestamps = extract_timestamps(df)

db.add_vectors_batch(windows, timestamps, batch_size=256)
print(f"Database ready with {len(db.vectors)} historical patterns")
```

### Use Case 3: Real-time Pattern Matching
```python
# During live trading
while trading_active:
    latest_30periods = fetch_latest_30periods()
    current_time = pd.Timestamp.now()

    # Add current pattern for future reference
    db.add_vector(latest_30periods, current_time)

    # Check for similar historical patterns
    matches = db.query_similar(latest_30periods, top_k=3, threshold=0.80)

    if matches:
        print(f"Found {len(matches)} similar historical patterns")
```

### Use Case 4: Data Validation
```python
# Check if database loaded correctly
stats = db.get_statistics()
print(f"Vectors loaded: {stats['num_vectors']}")
print(f"Device: {stats['device']}")
print(f"Time range: {stats['timestamp_range']}")
```

## API Cheat Sheet

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `add_vector(ohlcv, ts)` | Add single embedding | (30,5) array + timestamp | (128,) vector |
| `add_vectors_batch(batch, ts_list, bs)` | Add multiple embeddings | (N,30,5) array + N timestamps | (N,128) vectors |
| `query_similar(ohlcv, k, thresh)` | Find similar patterns | (30,5) array + parameters | List of dicts |
| `clear_vectors()` | Reset database | None | None |
| `get_statistics()` | Get DB info | None | Dict with stats |
| `save_checkpoint(path)` | Save to disk | File path string | None |
| `load_checkpoint(path)` | Load from disk | File path string | None |

## Parameter Guide

### OHLCV Input
- **Shape**: Must be exactly (30, 5)
- **Columns**: [Open, High, Low, Close, Volume] in this order
- **Data type**: float32 or float64
- **Price scale**: Any positive values work (auto-normalized)

### Timestamp
- **Type**: `pd.Timestamp`
- **Meaning**: Marks the END of the 30-period window
- **Uniqueness**: Should be unique for each vector

### Query Parameters
- **ohlcv_current**: (30, 5) array of current market data
- **top_k**: Number of results to return (default 5)
- **threshold**: Minimum similarity (0-1), returns only scores >= threshold

## Output Format

### Single Vector (add_vector)
```python
array([-0.178, -0.172, 0.005, ..., 0.234], dtype=float32)  # 128 values
```

### Batch Vectors (add_vectors_batch)
```python
array([
    [-0.178, -0.172, 0.005, ..., 0.234],
    [-0.215, -0.089, 0.123, ..., 0.456],
    ...
])  # Shape: (N, 128)
```

### Query Results (query_similar)
```python
[
    {
        'timestamp': Timestamp('2024-05-15 10:00:00'),
        'similarity': 0.9456,
        'rank': 1
    },
    {
        'timestamp': Timestamp('2024-04-20 14:30:00'),
        'similarity': 0.8921,
        'rank': 2
    },
    ...
]
```

## Error Handling

```python
try:
    db.add_vector(wrong_shape, timestamp)
except ValueError as e:
    print(f"Shape error: {e}")

if len(db.vectors) == 0:
    print("Database is empty, add data first")

results = db.query_similar(data, top_k=10)
if not results:
    print("No results found (empty database or high threshold)")
```

## Performance Tips

1. **Batch operations are faster**
   - Use `add_vectors_batch()` instead of loop of `add_vector()`
   - Faster inference and GPU memory efficiency

2. **Choose appropriate batch_size**
   - Larger batches = faster but more memory
   - Start with 256-512, adjust based on GPU

3. **Use threshold wisely**
   - threshold=0.9 finds very similar patterns (strict)
   - threshold=0.5 finds somewhat similar (loose)
   - No threshold returns all results sorted by similarity

4. **Save checkpoints**
   - After loading large datasets, save checkpoint
   - Faster reload than re-vectorizing data

## Example: Complete Workflow

```python
import numpy as np
import pandas as pd
from ace_trading.LSTM.vector_database import VectorDatabase

# 1. Initialize
db = VectorDatabase(
    model_path=r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth",
    device='cuda'
)

# 2. Load historical data
df = pd.read_csv('market_data.csv')
windows = []
timestamps = []

for i in range(30, len(df)):
    window = df.iloc[i-29:i+1][['Open','High','Low','Close','Volume']].values
    windows.append(window)
    timestamps.append(pd.Timestamp(df.iloc[i]['datetime']))

# 3. Add to database
print("Loading historical data...")
db.add_vectors_batch(np.array(windows), timestamps, batch_size=512)

# 4. Save checkpoint
db.save_checkpoint('historical_patterns')

# 5. Query
print("\nFinding similar patterns...")
current = windows[-1]  # Last 30 periods
matches = db.query_similar(current, top_k=5)

for m in matches:
    print(f"{m['timestamp'].date()}: {m['similarity']:.4f}")
```

## Next Steps

- Read full documentation: `VECTOR_DATABASE_DOCS.md`
- Run tests: `python test_vector_database.py`
- Integrate with backtest system (coming soon)
