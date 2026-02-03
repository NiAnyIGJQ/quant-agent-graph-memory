"""
Example: Vector Database Usage Scenarios

Shows practical examples of how to use the VectorDatabase component
in different trading workflow scenarios.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_database import VectorDatabase

# Configuration
MODEL_PATH = r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth"


# ==============================================================================
# SCENARIO 1: Build Database from Historical CSV
# ==============================================================================

def scenario_1_load_from_csv():
    """
    Load historical market data from CSV and build vector database.
    Useful for: Initial setup, backtesting preparation
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Load Historical Data from CSV")
    print("="*70)

    # Initialize database
    db = VectorDatabase(MODEL_PATH)

    # Simulate reading market data (in real scenario, load from CSV)
    print("Loading historical market data...")

    # Create fake historical data (replace with real CSV load)
    np.random.seed(42)
    num_periods = 5000  # Example: ~3 months of hourly data
    dates = pd.date_range('2024-01-01', periods=num_periods, freq='1H')

    ohlcv_data = np.random.randn(num_periods, 5) * 500 + 40000
    ohlcv_data[:, 4] = np.abs(ohlcv_data[:, 4]) * 1000  # Ensure positive volume

    # Create 30-period windows
    windows = []
    timestamps = []

    for i in range(30, len(dates)):
        window = ohlcv_data[i-29:i+1]
        windows.append(window)
        timestamps.append(pd.Timestamp(dates[i]))

    windows_array = np.array(windows)

    # Add to database in batches for efficiency
    print(f"Adding {len(windows)} vectors to database...")
    db.add_vectors_batch(windows_array, timestamps, batch_size=256)

    # Save for future use
    db.save_checkpoint('historical_db')

    # Print statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total vectors: {stats['num_vectors']}")
    print(f"  Embedding dimension: {stats['embed_dim']}")
    print(f"  Time range: {stats['timestamp_range']}")
    print(f"  Device: {stats['device']}")

    return db


# ==============================================================================
# SCENARIO 2: Real-time Pattern Matching
# ==============================================================================

def scenario_2_realtime_matching(db):
    """
    Find similar historical patterns for current market conditions.
    Useful for: Trading decisions, pattern analysis, risk assessment
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Real-time Pattern Matching")
    print("="*70)

    # Simulate current market data (last 30 periods)
    current_ohlcv = np.random.randn(30, 5) * 500 + 40000
    current_ohlcv[:, 4] = np.abs(current_ohlcv[:, 4]) * 1000

    print(f"\nCurrent market window: {pd.Timestamp.now()}")
    print(f"  Open: {current_ohlcv[0, 0]:.2f}")
    print(f"  High: {current_ohlcv[-1, 1]:.2f}")
    print(f"  Low: {current_ohlcv[-1, 2]:.2f}")
    print(f"  Close: {current_ohlcv[-1, 3]:.2f}")

    # Find similar patterns
    print("\nSearching for similar historical patterns...")
    matches = db.query_similar(current_ohlcv, top_k=5)

    if matches:
        print(f"\nFound {len(matches)} similar patterns:")
        print(f"{'Rank':<5} {'Timestamp':<25} {'Similarity':<12}")
        print("-" * 42)

        for match in matches:
            ts = match['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            sim = match['similarity']
            rank = match['rank']
            print(f"{rank:<5} {ts:<25} {sim:.4f}")

        # Use highest match for analysis
        best_match = matches[0]
        print(f"\nBest match: {best_match['timestamp']} with similarity {best_match['similarity']:.4f}")
        print("Action: Could analyze what happened after this historical date...")

    return matches


# ==============================================================================
# SCENARIO 3: High-Confidence Pattern Filtering
# ==============================================================================

def scenario_3_threshold_filtering(db):
    """
    Use similarity threshold to find only high-confidence matches.
    Useful for: Conservative trading, strong signal generation
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Threshold-based Filtering")
    print("="*70)

    current_ohlcv = np.random.randn(30, 5) * 500 + 40000
    current_ohlcv[:, 4] = np.abs(current_ohlcv[:, 4]) * 1000

    # Different threshold levels
    thresholds = [0.99, 0.95, 0.90, 0.80]

    print("\nMatches at different similarity thresholds:")
    print(f"{'Threshold':<12} {'Matches Found':<15} {'Interpretation':<30}")
    print("-" * 57)

    for threshold in thresholds:
        matches = db.query_similar(current_ohlcv, top_k=1000, threshold=threshold)
        if matches:
            interpretation = "Very high confidence" if threshold > 0.95 else \
                            "High confidence" if threshold > 0.85 else \
                            "Moderate confidence"
        else:
            interpretation = "No matches"

        print(f"{threshold:<12.2f} {len(matches):<15} {interpretation:<30}")

    # Use strict threshold
    high_conf = db.query_similar(current_ohlcv, top_k=100, threshold=0.95)
    print(f"\nHigh-confidence matches (>0.95): {len(high_conf)}")
    if high_conf:
        print(f"  Most similar: {high_conf[0]['timestamp']} ({high_conf[0]['similarity']:.4f})")


# ==============================================================================
# SCENARIO 4: Comparative Analysis (Similar vs Dissimilar)
# ==============================================================================

def scenario_4_comparative_analysis(db):
    """
    Compare current market with both similar and dissimilar patterns.
    Useful for: Contrast analysis, understanding market dynamics
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Comparative Analysis")
    print("="*70)

    current_ohlcv = np.random.randn(30, 5) * 500 + 40000
    current_ohlcv[:, 4] = np.abs(current_ohlcv[:, 4]) * 1000

    # Get all matches sorted by similarity
    all_matches = db.query_similar(current_ohlcv, top_k=1000)

    # Split into similar and dissimilar
    similar = all_matches[:5]
    dissimilar = all_matches[-5:] if len(all_matches) > 5 else []

    print("\nMost SIMILAR patterns:")
    print(f"{'Rank':<5} {'Similarity':<12} {'Date':<20}")
    print("-" * 37)
    for m in similar:
        sim = m['similarity']
        ts = m['timestamp'].strftime('%Y-%m-%d %H:%M')
        rank = m['rank']
        print(f"{rank:<5} {sim:.4f}       {ts:<20}")

    if dissimilar:
        print("\nMost DISSIMILAR patterns (for contrast):")
        print(f"{'Similarity':<12} {'Date':<20}")
        print("-" * 32)
        for m in dissimilar:
            sim = m['similarity']
            ts = m['timestamp'].strftime('%Y-%m-%d %H:%M')
            print(f"{sim:.4f}       {ts:<20}")

        print("\nInsight: When market patterns are opposite, outcomes may differ")


# ==============================================================================
# SCENARIO 5: Incremental Database Updates
# ==============================================================================

def scenario_5_incremental_update(db):
    """
    Add new data points to existing database (e.g., new market data).
    Useful for: Live trading, continuous learning
    """
    print("\n" + "="*70)
    print("SCENARIO 5: Incremental Database Updates")
    print("="*70)

    initial_size = len(db.vectors)
    print(f"Initial database size: {initial_size} vectors")

    # Add new data (simulating new market periods)
    print("\nAdding new market data...")

    new_windows = []
    new_timestamps = []

    for i in range(10):
        new_window = np.random.randn(30, 5) * 500 + 40000
        new_window[:, 4] = np.abs(new_window[:, 4]) * 1000
        new_windows.append(new_window)

        timestamp = pd.Timestamp.now() - pd.Timedelta(hours=10-i)
        new_timestamps.append(timestamp)

    db.add_vectors_batch(np.array(new_windows), new_timestamps)

    final_size = len(db.vectors)
    print(f"Added: {final_size - initial_size} vectors")
    print(f"Final database size: {final_size} vectors")

    # Re-save checkpoint with new data
    print("\nSaving updated checkpoint...")
    db.save_checkpoint('updated_historical_db')


# ==============================================================================
# SCENARIO 6: Database Checkpoint Management
# ==============================================================================

def scenario_6_checkpoint_workflow():
    """
    Demonstrate checkpoint save and restore workflow.
    Useful for: Persistence, resuming runs, version control
    """
    print("\n" + "="*70)
    print("SCENARIO 6: Checkpoint Workflow")
    print("="*70)

    # Create and populate a database
    db_original = VectorDatabase(MODEL_PATH)

    print("Creating database with sample data...")
    sample_windows = np.random.randn(100, 30, 5) * 500 + 40000
    sample_timestamps = [
        pd.Timestamp('2024-06-01') + pd.Timedelta(hours=i)
        for i in range(100)
    ]

    db_original.add_vectors_batch(sample_windows, sample_timestamps)
    print(f"Original database: {len(db_original.vectors)} vectors")

    # Save checkpoint
    checkpoint_path = 'checkpoint_demo'
    print(f"\nSaving checkpoint to '{checkpoint_path}'...")
    db_original.save_checkpoint(checkpoint_path)

    # Restore from checkpoint
    print("Restoring from checkpoint...")
    db_restored = VectorDatabase(MODEL_PATH)
    db_restored.load_checkpoint(checkpoint_path)

    print(f"Restored database: {len(db_restored.vectors)} vectors")

    # Verify integrity
    match = np.allclose(db_original.vectors, db_restored.vectors)
    print(f"Data integrity check: {'PASSED' if match else 'FAILED'}")

    # Continue using restored database
    print("\nUsing restored database for queries...")
    test_query = sample_windows[0]
    results = db_restored.query_similar(test_query, top_k=3)
    print(f"Found {len(results)} similar patterns")


# ==============================================================================
# MAIN: Run All Scenarios
# ==============================================================================

def main():
    """Run all example scenarios."""

    print("\n" + "="*70)
    print("VECTOR DATABASE USAGE EXAMPLES")
    print("="*70)

    # Scenario 1: Load from CSV
    db = scenario_1_load_from_csv()

    # Scenario 2: Real-time matching
    scenario_2_realtime_matching(db)

    # Scenario 3: Threshold filtering
    scenario_3_threshold_filtering(db)

    # Scenario 4: Comparative analysis
    scenario_4_comparative_analysis(db)

    # Scenario 5: Incremental updates
    scenario_5_incremental_update(db)

    # Scenario 6: Checkpoint workflow
    scenario_6_checkpoint_workflow()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
