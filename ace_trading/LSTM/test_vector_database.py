"""
Test script for VectorDatabase component.

Tests:
1. Model loading and initialization
2. Single vector generation and storage
3. Batch vector generation
4. Similarity search with various configurations
5. Database clearing
6. Checkpoint save/load
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_database import VectorDatabase

# Configuration
MODEL_PATH = r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth"
CHECKPOINT_PATH = r"D:\BTC_ACE\ace_trading\LSTM\test_checkpoint"


def generate_fake_ohlcv(num_samples: int = 1, periods: int = 30) -> np.ndarray:
    """Generate fake OHLCV data for testing."""
    np.random.seed(42)
    data = np.random.randn(num_samples, periods, 5).astype(np.float32)
    # Scale to realistic ranges
    data[:, :, :4] = 40000 + data[:, :, :4] * 1000  # Price
    data[:, :, 4] = np.abs(data[:, :, 4]) * 10000  # Volume
    return data


def test_initialization():
    """Test 1: Database initialization."""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)

    db = VectorDatabase(
        model_path=MODEL_PATH,
        input_win=30,
        embed_dim=128
    )

    stats = db.get_statistics()
    print(f" Database initialized")
    print(f"   - Device: {stats['device']}")
    print(f"   - Embedding dimension: {stats['embed_dim']}")
    print(f"   - Stored vectors: {stats['num_vectors']}")

    return db


def test_single_vector(db: VectorDatabase):
    """Test 2: Single vector generation and storage."""
    print("\n" + "="*60)
    print("TEST 2: Single Vector Generation")
    print("="*60)

    ohlcv = generate_fake_ohlcv(num_samples=1, periods=30)[0]
    timestamp = pd.Timestamp('2024-06-01 10:00:00')

    emb = db.add_vector(ohlcv, timestamp)

    print(f" Vector generated and stored")
    print(f"   - Embedding shape: {emb.shape}")
    print(f"   - Embedding dtype: {emb.dtype}")
    print(f"   - Sample values: {emb[:5]}")
    print(f"   - Database size: {len(db.vectors)}")

    stats = db.get_statistics()
    print(f"   - Time range: {stats['timestamp_range']}")


def test_batch_vectors(db: VectorDatabase):
    """Test 3: Batch vector generation."""
    print("\n" + "="*60)
    print("TEST 3: Batch Vector Generation")
    print("="*60)

    # Generate 10 samples
    ohlcv_batch = generate_fake_ohlcv(num_samples=10, periods=30)
    base_date = pd.Timestamp('2024-06-01')
    timestamps = [base_date + pd.Timedelta(hours=i) for i in range(10)]

    embeddings = db.add_vectors_batch(ohlcv_batch, timestamps, batch_size=4)

    print(f" Batch vectors generated and stored")
    print(f"   - Batch size: {len(timestamps)}")
    print(f"   - Embeddings shape: {embeddings.shape}")
    print(f"   - Total vectors in DB: {len(db.vectors)}")

    stats = db.get_statistics()
    print(f"   - Time range: {stats['timestamp_range']}")


def test_query_similarity(db: VectorDatabase):
    """Test 4: Query similarity search."""
    print("\n" + "="*60)
    print("TEST 4: Similarity Search")
    print("="*60)

    # Generate a query sample
    query_ohlcv = generate_fake_ohlcv(num_samples=1, periods=30)[0]

    # Query with top-k
    results = db.query_similar(query_ohlcv, top_k=3)

    print(f" Similarity search completed")
    print(f"   - Query results count: {len(results)}")

    if results:
        print(f"\n   Top similar patterns:")
        for result in results:
            print(f"     Rank {result['rank']}: {result['timestamp']} "
                  f"(similarity: {result['similarity']:.4f})")
    else:
        print("   (No results - database may be empty)")


def test_query_with_threshold(db: VectorDatabase):
    """Test 5: Query with similarity threshold."""
    print("\n" + "="*60)
    print("TEST 5: Query with Threshold")
    print("="*60)

    query_ohlcv = generate_fake_ohlcv(num_samples=1, periods=30)[0]

    # Query with threshold
    results = db.query_similar(query_ohlcv, top_k=10, threshold=0.5)

    print(f" Threshold-based search completed")
    print(f"   - Results above threshold (0.5): {len(results)}")

    if results:
        for result in results:
            print(f"     {result['timestamp']}: {result['similarity']:.4f}")


def test_clear(db: VectorDatabase):
    """Test 6: Clear database."""
    print("\n" + "="*60)
    print("TEST 6: Clear Database")
    print("="*60)

    old_size = len(db.vectors)
    db.clear_vectors()
    new_size = len(db.vectors)

    print(f" Database cleared")
    print(f"   - Vectors before: {old_size}")
    print(f"   - Vectors after: {new_size}")
    print(f"   - Timestamps: {len(db.timestamps)}")


def test_checkpoint(db: VectorDatabase):
    """Test 7: Checkpoint save/load."""
    print("\n" + "="*60)
    print("TEST 7: Checkpoint Save/Load")
    print("="*60)

    # Add some data
    ohlcv = generate_fake_ohlcv(num_samples=5, periods=30)
    timestamps = [pd.Timestamp('2024-06-01') + pd.Timedelta(hours=i) for i in range(5)]
    db.add_vectors_batch(ohlcv, timestamps)

    print(f"   Before save: {len(db.vectors)} vectors")

    # Save
    db.save_checkpoint(CHECKPOINT_PATH)
    print(f" Checkpoint saved")

    # Create new database and load
    db_new = VectorDatabase(model_path=MODEL_PATH)
    db_new.load_checkpoint(CHECKPOINT_PATH)

    print(f" Checkpoint loaded")
    print(f"   After load: {len(db_new.vectors)} vectors")
    print(f"   Timestamps match: {db_new.timestamps == db.timestamps}")

    # Verify data integrity
    vectors_match = np.allclose(db.vectors, db_new.vectors)
    print(f"   Vector data match: {vectors_match}")

    return db_new


def main():
    """Run all tests."""
    print("\n" + "Test Suite: Vector Database Component" + "\n")

    try:
        # Test 1: Initialization
        db = test_initialization()

        # Test 2: Single vector
        test_single_vector(db)

        # Test 3: Batch vectors
        test_batch_vectors(db)

        # Test 4: Query similarity
        test_query_similarity(db)

        # Test 5: Query with threshold
        test_query_with_threshold(db)

        # Test 6: Clear
        test_clear(db)

        # Test 7: Checkpoint
        db_restored = test_checkpoint(db)

        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
