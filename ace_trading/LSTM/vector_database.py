"""
VectorDatabase Component for BTC_ACE Trading System

Handles:
1. Loading LSTM attention model for embedding generation
2. Converting market data (30-period OHLCV) to vectors
   - 【重要】INPUT_WIN=30 现在代表 30 个 1H K线（即 30 小时历史）
   - 不再是 30 个 1M K线，这样可以获得更稳定的市场模式识别
3. Similarity search via cosine similarity
4. Timestamp-based historical pattern retrieval

Data Flow:
- 输入: 30期 1H OHLCV 数据 [30, 5] 矩阵
- 处理: Z-score 标准化 → LSTM编码 → 注意力加权 → 128维嵌入
- 输出: 128维向量 + 时间戳，用于相似度搜索
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Attention mechanism for LSTM output."""

    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


class CompositeModel(nn.Module):
    """LSTM-based model for generating market embeddings."""

    def __init__(self, input_dim: int = 5, embed_dim: int = 128,
                 hidden_dim: int = 512, num_layers: int = 3,
                 input_win: int = 30, pred_win: int = 5):
        super(CompositeModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_win = input_win
        self.pred_win = pred_win

        # Encoder: LSTM + Attention
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                    batch_first=True, dropout=0.0)
        self.attention = Attention(hidden_dim)
        self.encoder_fc = nn.Linear(hidden_dim, embed_dim)

        # Reconstruction decoder (for compatibility)
        self.recon_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.recon_fc = nn.Linear(hidden_dim, input_dim)

        # Prediction decoder (for compatibility)
        self.pred_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.pred_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_win, input_dim)

        Returns:
            recon: Reconstructed historical sequence
            pred: Predicted future sequence
            emb: Embedding vector (batch_size, embed_dim)
        """
        # Encoder
        lstm_out, _ = self.encoder_lstm(x)
        context = self.attention(lstm_out)
        emb = self.encoder_fc(context)

        # Reconstruction
        emb_expanded_recon = emb.unsqueeze(1).repeat(1, self.input_win, 1)
        out_recon, _ = self.recon_lstm(emb_expanded_recon)
        recon = self.recon_fc(out_recon)

        # Prediction
        emb_expanded_pred = emb.unsqueeze(1).repeat(1, self.pred_win, 1)
        out_pred, _ = self.pred_lstm(emb_expanded_pred)
        pred = self.pred_fc(out_pred)

        return recon, pred, emb


class VectorDatabase:
    """
    Vector database for storing and retrieving market embeddings.

    Workflow:
    1. Initialize with model path and raw market data
    2. Use add_vector() to generate and store embeddings from market data
    3. Use query_similar() to find similar historical market patterns
    """

    def __init__(self, model_path: str, device: Optional[str] = None,
                 input_win: int = 30, pred_win: int = 5,
                 embed_dim: int = 128, hidden_dim: int = 512, num_layers: int = 3):
        """
        Initialize vector database.

        Args:
            model_path: Path to LSTM model (.pth file)
            device: torch device ('cuda' or 'cpu'). Auto-detects if None.
            input_win: Historical window size (default 30 periods)
            pred_win: Future prediction window (default 5 periods)
            embed_dim: Embedding dimension (default 128)
            hidden_dim: LSTM hidden dimension (default 512)
            num_layers: LSTM layers (default 3)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        self.input_win = input_win
        self.pred_win = pred_win
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Storage
        self.vectors: np.ndarray = np.empty((0, embed_dim), dtype=np.float32)
        self.timestamps: List[pd.Timestamp] = []

        logger.info(f" VectorDatabase initialized on {self.device}")

    def _load_model(self, model_path: str) -> CompositeModel:
        """Load LSTM model from checkpoint."""
        model = CompositeModel(
            input_dim=5,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            input_win=self.input_win,
            pred_win=self.pred_win
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        logger.info(f" Model loaded from {model_path}")
        return model

    def clear_vectors(self) -> None:
        """Clear all stored vectors and timestamps."""
        self.vectors = np.empty((0, self.embed_dim), dtype=np.float32)
        self.timestamps = []
        logger.info(" Vector database cleared")

    def _normalize_ohlcv(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Normalize OHLCV data using Z-score normalization.

        Args:
            ohlcv: Array of shape (seq_len, 5) with columns [O, H, L, C, V]

        Returns:
            Normalized array of same shape
        """
        normalized = ohlcv.copy().astype(np.float32)

        # Price normalization (OHLC) - first 4 columns
        price_data = normalized[:, :4]
        p_mean = np.mean(price_data, axis=0, keepdims=True)
        p_std = np.std(price_data, axis=0, keepdims=True) + 1e-8
        normalized[:, :4] = (price_data - p_mean) / p_std

        # Volume normalization - last column (log-normalized)
        vol_data = np.log1p(normalized[:, 4:5])
        v_mean = np.mean(vol_data, axis=0, keepdims=True)
        v_std = np.std(vol_data, axis=0, keepdims=True) + 1e-8
        normalized[:, 4:5] = (vol_data - v_mean) / v_std

        return normalized

    def add_vector(self, ohlcv_data: np.ndarray, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Generate embedding from OHLCV data and add to database.

        Args:
            ohlcv_data: Array of shape (input_win, 5) with [Open, High, Low, Close, Volume]
                       representing exactly 30 periods of market data
            timestamp: Timestamp marking the end of this 30-period window

        Returns:
            Generated embedding vector of shape (embed_dim,)
        """
        if ohlcv_data.shape[0] != self.input_win:
            raise ValueError(f"Expected {self.input_win} periods, got {ohlcv_data.shape[0]}")

        # Normalize
        norm_data = self._normalize_ohlcv(ohlcv_data)

        # Convert to tensor
        x = torch.from_numpy(norm_data).unsqueeze(0).to(self.device)  # (1, 30, 5)

        # Generate embedding
        with torch.no_grad():
            _, _, emb = self.model(x)
            emb_np = emb.cpu().numpy()[0]  # (embed_dim,)

        # Store
        self.vectors = np.vstack([self.vectors, emb_np.reshape(1, -1)])
        self.timestamps.append(timestamp)

        return emb_np

    def add_vectors_batch(self, ohlcv_batch: np.ndarray,
                         timestamps: List[pd.Timestamp],
                         batch_size: int = 1024) -> np.ndarray:
        """
        Generate embeddings from a batch of OHLCV data.

        Args:
            ohlcv_batch: Array of shape (num_samples, input_win, 5)
            timestamps: List of timestamps (one per sample)
            batch_size: Inference batch size

        Returns:
            All embeddings of shape (num_samples, embed_dim)
        """
        if ohlcv_batch.shape[1] != self.input_win:
            raise ValueError(f"Expected {self.input_win} periods, got {ohlcv_batch.shape[1]}")

        if len(timestamps) != ohlcv_batch.shape[0]:
            raise ValueError("Number of timestamps must match number of samples")

        # Normalize all samples
        norm_batch = np.array([self._normalize_ohlcv(sample) for sample in ohlcv_batch])

        # Batch inference
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(norm_batch), batch_size):
                batch_data = norm_batch[i:i+batch_size]
                x = torch.from_numpy(batch_data).to(self.device)
                _, _, emb = self.model(x)
                embeddings.append(emb.cpu().numpy())

        all_embeddings = np.vstack(embeddings)

        # Store
        self.vectors = np.vstack([self.vectors, all_embeddings])
        self.timestamps.extend(timestamps)

        return all_embeddings

    def query_similar(self, ohlcv_current: np.ndarray, top_k: int = 5,
                     threshold: Optional[float] = None) -> List[Dict]:
        """
        Find similar historical patterns in the vector database.

        Args:
            ohlcv_current: Current market data, shape (input_win, 5)
            top_k: Number of top similar results to return
            threshold: Minimum similarity score (0-1). If None, all results returned.

        Returns:
            List of dicts with keys:
            - 'timestamp': pd.Timestamp of matching historical pattern
            - 'similarity': Cosine similarity score (0-1)
            - 'rank': Ranking (1-indexed)
        """
        if len(self.vectors) == 0:
            logger.warning(" Vector database is empty")
            return []

        if ohlcv_current.shape[0] != self.input_win:
            raise ValueError(f"Expected {self.input_win} periods, got {ohlcv_current.shape[0]}")

        # Generate embedding for current data
        norm_current = self._normalize_ohlcv(ohlcv_current)
        x = torch.from_numpy(norm_current).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, query_emb = self.model(x)
            query_vec = query_emb.cpu().numpy()  # (1, embed_dim)

        # Similarity search
        similarities = cosine_similarity(query_vec, self.vectors).flatten()

        # Get top-k
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(similarities[idx])

            # Apply threshold if specified
            if threshold is not None and score < threshold:
                continue

            results.append({
                'timestamp': self.timestamps[idx],
                'similarity': score,
                'rank': rank
            })

        return results

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        return {
            'num_vectors': len(self.vectors),
            'embed_dim': self.embed_dim,
            'timestamp_range': f"{self.timestamps[0]} - {self.timestamps[-1]}" if self.timestamps else "empty",
            'device': str(self.device)
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save vectors and timestamps to disk."""
        np.save(filepath + '_vectors.npy', self.vectors)
        np.save(filepath + '_timestamps.npy',
               np.array([str(ts) for ts in self.timestamps], dtype=object))
        logger.info(f" Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load vectors and timestamps from disk."""
        self.vectors = np.load(filepath + '_vectors.npy')
        timestamp_strs = np.load(filepath + '_timestamps.npy', allow_pickle=True)
        self.timestamps = [pd.Timestamp(ts) for ts in timestamp_strs]
        logger.info(f" Checkpoint loaded from {filepath}")
