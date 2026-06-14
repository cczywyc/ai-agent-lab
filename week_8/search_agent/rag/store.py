"""
最小向量存储 — numpy + 余弦相似度

为什么不直接上 FAISS/Chroma：
  - 学习阶段语料小（几十~几百 chunks），numpy 完全够用
  - 全在内存，启动即载入，无外部服务依赖
  - 持久化 = .npy + .json，零运维

持久化结构（VECTOR_STORE_DIR 下）：
  {namespace}_vectors.npy   形状 (N, D)，已 L2 归一化
  {namespace}_metadata.json [{doc, section, chunk_id, text, ...}, ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import VECTOR_STORE_DIR, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    """
    单 namespace 的向量库。
    多 namespace 共用一个目录（不同前缀文件名），为后续记忆系统预留扩展。
    """

    def __init__(self, namespace: str = "docs", persist_dir: Optional[Path] = None):
        self.namespace = namespace
        self.persist_dir = Path(persist_dir or VECTOR_STORE_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.vectors: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self.metadata: list[dict] = []

    # ---------- 路径 ----------
    @property
    def _vec_path(self) -> Path:
        return self.persist_dir / f"{self.namespace}_vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.persist_dir / f"{self.namespace}_metadata.json"

    # ---------- 写入 ----------
    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """追加向量与元数据。vectors 必须已 L2 归一化。"""
        if len(vectors) != len(metadata):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata ({len(metadata)}) length mismatch"
            )
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != EMBEDDING_DIM {EMBEDDING_DIM}"
            )

        if self.vectors.shape[0] == 0:
            self.vectors = vectors.astype(np.float32, copy=False)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32, copy=False)])
        self.metadata.extend(metadata)

    def reset(self) -> None:
        """清空内存中数据（不删持久化文件，需要手动 save 覆盖）。"""
        self.vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self.metadata = []

    # ---------- 持久化 ----------
    def save(self) -> None:
        np.save(self._vec_path, self.vectors)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(
            f"VectorStore[{self.namespace}] saved: "
            f"{len(self.metadata)} chunks → {self.persist_dir}"
        )

    def load(self) -> bool:
        """从磁盘加载。返回 True 表示成功，False 表示文件不存在。"""
        if not self._vec_path.exists() or not self._meta_path.exists():
            return False
        self.vectors = np.load(self._vec_path).astype(np.float32, copy=False)
        with open(self._meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True

    # ---------- 检索 ----------
    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """
        余弦 top-k 检索。

        Args:
            query_vector: shape (D,)，必须已 L2 归一化
            top_k: 返回前 k 条
        Returns:
            [(metadata, score), ...]，score 范围 [-1, 1]，已按降序排列
        """
        if self.vectors.shape[0] == 0:
            return []

        # 余弦相似度 = 点积（双方已归一化）
        scores = self.vectors @ query_vector  # shape (N,)

        k = min(top_k, len(scores))
        # argpartition 取 top-k 索引（无序），再对这 k 个排序
        if k < len(scores):
            top_idx = np.argpartition(scores, -k)[-k:]
        else:
            top_idx = np.arange(len(scores))
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(self.metadata[i], float(scores[i])) for i in top_idx]

    # ---------- 状态 ----------
    def __len__(self) -> int:
        return len(self.metadata)

    def info(self) -> dict:
        return {
            "namespace": self.namespace,
            "count": len(self.metadata),
            "dim": int(self.vectors.shape[1]) if self.vectors.size else EMBEDDING_DIM,
            "persist_dir": str(self.persist_dir),
        }
