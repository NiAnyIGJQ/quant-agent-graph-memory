"""
VectorDatabaseManager - 向量数据库持久化管理器

功能：
1. 将向量数据库保存到磁盘（NPZ格式）
2. 加载已保存的向量数据库
3. 导出向量库为可交互的格式（JSON、CSV）
4. 生成统计报告
5. 支持向量数据查询接口（不需要重新计算）
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorDatabaseManager:
    """
    向量数据库持久化管理器

    负责处理向量数据的保存、加载和交互查询，
    不涉及向量的生成（由VectorDatabase类处理）
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化管理器

        Args:
            base_dir: 向量库保存的基础目录，默认为 snapshots/latest
        """
        if base_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            base_dir = os.path.join(repo_root, 'snapshots', 'latest')

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self.vectors: np.ndarray = np.empty((0, 128), dtype=np.float32)  # 默认128维
        self.timestamps: List[pd.Timestamp] = []
        self.metadata: Dict = {}

        logger.info(f"VectorDatabaseManager initialized at {self.base_dir}")

    def save_from_vec_db(self, vec_db, output_name: str = "vector_db_checkpoint") -> str:
        """
        从VectorDatabase对象保存向量数据

        Args:
            vec_db: VectorDatabase实例
            output_name: 输出文件名前缀

        Returns:
            保存的目录路径
        """
        try:
            # 复制数据
            self.vectors = vec_db.vectors.copy()
            self.timestamps = vec_db.timestamps.copy()
            self.metadata = {
                'embed_dim': vec_db.embed_dim,
                'hidden_dim': vec_db.hidden_dim,
                'num_layers': vec_db.num_layers,
                'input_win': vec_db.input_win,
                'pred_win': vec_db.pred_win,
                'device': str(vec_db.device),
                'saved_at': datetime.now().isoformat()
            }

            # 保存数据
            save_path = self._save_checkpoint(output_name)

            print(f"[VECTOR_DB_MANAGER] 向量库已保存: {save_path}")
            print(f"  - 向量数: {len(self.vectors)}")
            print(f"  - 向量维度: {self.vectors.shape[1] if len(self.vectors) > 0 else 0}")
            print(f"  - 时间范围: {self.timestamps[0]} - {self.timestamps[-1] if self.timestamps else 'N/A'}")

            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to save from vec_db: {e}")
            raise

    def _save_checkpoint(self, name: str = "vector_db_checkpoint") -> Path:
        """
        内部方法：保存检查点

        Args:
            name: 文件名前缀

        Returns:
            保存目录路径
        """
        output_dir = self.base_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存向量和时间戳为NPZ（二进制，高效）
        npz_path = output_dir / f"{name}.npz"
        np.savez_compressed(
            npz_path,
            vectors=self.vectors,
            timestamps=np.array([str(ts) for ts in self.timestamps], dtype=object),
            metadata=json.dumps(self.metadata)
        )
        print(f"  ✓ NPZ文件: {npz_path}")

        # 2. 保存元数据为JSON（可读性强）
        metadata_path = output_dir / f"{name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                **self.metadata,
                'total_vectors': int(len(self.vectors)),
                'vector_dimension': int(self.vectors.shape[1]) if len(self.vectors) > 0 else 0,
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 元数据: {metadata_path}")

        # 3. 保存时间戳索引为CSV（用于查询）
        if self.timestamps:
            index_path = output_dir / f"{name}_index.csv"
            df_index = pd.DataFrame({
                'index': range(len(self.timestamps)),
                'timestamp': [str(ts) for ts in self.timestamps],
                'vector_id': [f"vec_{i:06d}" for i in range(len(self.timestamps))]
            })
            df_index.to_csv(index_path, index=False)
            print(f"  ✓ 时间戳索引: {index_path}")

        # 4. 保存统计信息
        stats_path = output_dir / f"{name}_statistics.json"
        stats = self._compute_statistics()
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 统计信息: {stats_path}")

        return output_dir

    def load_checkpoint(self, checkpoint_dir: Optional[str] = None) -> bool:
        """
        加载之前保存的向量库

        Args:
            checkpoint_dir: 检查点目录，默认使用base_dir

        Returns:
            加载成功返回True，失败返回False
        """
        try:
            if checkpoint_dir is None:
                checkpoint_dir = self.base_dir
            else:
                checkpoint_dir = Path(checkpoint_dir)

            # 查找NPZ文件
            npz_files = list(checkpoint_dir.glob("vector_db_checkpoint.npz"))
            if not npz_files:
                # 尝试找任何.npz文件
                npz_files = list(checkpoint_dir.glob("*.npz"))

            if not npz_files:
                logger.error(f"No .npz checkpoint found in {checkpoint_dir}")
                return False

            npz_path = npz_files[0]

            # 加载数据
            data = np.load(npz_path, allow_pickle=True)
            self.vectors = data['vectors']

            # 恢复时间戳
            timestamp_strs = data['timestamps']
            self.timestamps = [pd.Timestamp(ts) for ts in timestamp_strs]

            # 加载元数据
            if 'metadata' in data:
                self.metadata = json.loads(data['metadata'].item())

            logger.info(f"Checkpoint loaded from {npz_path}")
            print(f"[VECTOR_DB_MANAGER] 向量库已加载:")
            print(f"  - 向量数: {len(self.vectors)}")
            print(f"  - 向量维度: {self.vectors.shape[1]}")
            print(f"  - 时间范围: {self.timestamps[0]} - {self.timestamps[-1]}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def export_as_dataframe(self) -> pd.DataFrame:
        """
        导出为pandas DataFrame，便于数据分析

        Returns:
            DataFrame，列为 ['timestamp', 'vector_id', 'vector_data']
        """
        if len(self.vectors) == 0:
            return pd.DataFrame()

        data = {
            'timestamp': [str(ts) for ts in self.timestamps],
            'vector_id': [f"vec_{i:06d}" for i in range(len(self.vectors))],
        }

        # 将向量添加为单独的列（每列一个维度）
        for i in range(self.vectors.shape[1]):
            data[f'dim_{i:03d}'] = self.vectors[:, i]

        return pd.DataFrame(data)

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        导出为CSV文件（包含向量维度）

        Args:
            output_path: 输出路径，默认为 {base_dir}/vector_db_export.csv

        Returns:
            保存的文件路径
        """
        if output_path is None:
            output_path = self.base_dir / "vector_db_export.csv"
        else:
            output_path = Path(output_path)

        df = self.export_as_dataframe()
        df.to_csv(output_path, index=False)

        logger.info(f"Exported to CSV: {output_path}")
        print(f"[EXPORT] 向量库已导出为CSV: {output_path}")

        return str(output_path)

    def export_to_json(self, output_path: Optional[str] = None,
                      include_vectors: bool = True) -> str:
        """
        导出为JSON格式（用于Web应用或API）

        Args:
            output_path: 输出路径
            include_vectors: 是否包含完整向量数据

        Returns:
            保存的文件路径
        """
        if output_path is None:
            output_path = self.base_dir / "vector_db_export.json"
        else:
            output_path = Path(output_path)

        export_data = {
            'metadata': self.metadata,
            'statistics': self._compute_statistics(),
            'total_vectors': len(self.vectors),
            'timestamp_range': {
                'start': str(self.timestamps[0]) if self.timestamps else None,
                'end': str(self.timestamps[-1]) if self.timestamps else None
            }
        }

        if include_vectors:
            export_data['vectors'] = [
                {
                    'id': f"vec_{i:06d}",
                    'timestamp': str(ts),
                    'vector': self.vectors[i].tolist()
                }
                for i, ts in enumerate(self.timestamps)
            ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported to JSON: {output_path}")
        print(f"[EXPORT] 向量库已导出为JSON: {output_path}")

        return str(output_path)

    def query_by_timestamp(self, timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """
        按时间戳查询向量（不需要重新计算）

        Args:
            timestamp: 查询时间戳

        Returns:
            对应的向量，未找到返回None
        """
        try:
            idx = self.timestamps.index(timestamp)
            return self.vectors[idx]
        except ValueError:
            return None

    def query_by_timerange(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[List[pd.Timestamp], np.ndarray]:
        """
        按时间范围查询向量

        Args:
            start: 起始时间
            end: 结束时间

        Returns:
            (时间戳列表, 向量数组)
        """
        mask = np.array([(start <= ts <= end) for ts in self.timestamps])
        indices = np.where(mask)[0]

        selected_timestamps = [self.timestamps[i] for i in indices]
        selected_vectors = self.vectors[indices]

        return selected_timestamps, selected_vectors

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self._compute_statistics()

    def _compute_statistics(self) -> Dict:
        """计算统计信息"""
        if len(self.vectors) == 0:
            return {'status': 'empty', 'vector_count': 0}

        return {
            'vector_count': int(len(self.vectors)),
            'vector_dimension': int(self.vectors.shape[1]),
            'timestamp_range': {
                'start': str(self.timestamps[0]),
                'end': str(self.timestamps[-1]),
                'duration_days': (self.timestamps[-1] - self.timestamps[0]).days
            },
            'vector_stats': {
                'mean': float(np.mean(self.vectors)),
                'std': float(np.std(self.vectors)),
                'min': float(np.min(self.vectors)),
                'max': float(np.max(self.vectors))
            },
            'memory_usage_mb': float(self.vectors.nbytes / 1024 / 1024)
        }

    def create_summary_report(self, output_path: Optional[str] = None) -> str:
        """
        创建向量库总结报告

        Args:
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        if output_path is None:
            output_path = self.base_dir / "vector_db_summary.txt"
        else:
            output_path = Path(output_path)

        stats = self.get_statistics()

        report = f"""
{'='*60}
向量数据库总结报告
{'='*60}

生成时间: {datetime.now().isoformat()}

基本信息:
  - 向量总数: {stats.get('vector_count', 0)}
  - 向量维度: {stats.get('vector_dimension', 0)}
  - 存储大小: {stats.get('memory_usage_mb', 0):.2f} MB

时间范围:
  - 起始: {stats.get('timestamp_range', {}).get('start', 'N/A')}
  - 结束: {stats.get('timestamp_range', {}).get('end', 'N/A')}
  - 时间跨度: {stats.get('timestamp_range', {}).get('duration_days', 0)} 天

向量统计:
  - 平均值: {stats.get('vector_stats', {}).get('mean', 0):.6f}
  - 标准差: {stats.get('vector_stats', {}).get('std', 0):.6f}
  - 最小值: {stats.get('vector_stats', {}).get('min', 0):.6f}
  - 最大值: {stats.get('vector_stats', {}).get('max', 0):.6f}

模型配置:
  - 嵌入维度: {self.metadata.get('embed_dim', 'N/A')}
  - 隐藏维度: {self.metadata.get('hidden_dim', 'N/A')}
  - LSTM层数: {self.metadata.get('num_layers', 'N/A')}
  - 输入窗口: {self.metadata.get('input_win', 'N/A')}
  - 预测窗口: {self.metadata.get('pred_win', 'N/A')}

{'='*60}
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[REPORT] 总结报告已生成: {output_path}")

        return str(output_path)
