"""
Retriever — 拼装 embed + store 的最终检索接口

对外暴露：
  get_retriever() → 全局单例（懒加载向量库）
  Retriever.retrieve(query, top_k) → list[dict]，每条带 doc/section/chunk_id/score/text

对调用方友好：返回 dict 列表（而非 dataclass），方便直接 json 序列化进 tool result。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    VECTOR_STORE_DIR, RETRIEVE_TOP_K, RETRIEVE_MIN_SCORE,
)
from .embed import embed_query, embed_texts
from .store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """对 store + embed 的封装。"""

    def __init__(self, namespace: str = "docs"):
        self.namespace = namespace
        self.store = VectorStore(namespace=namespace)
        self._loaded = False

    def load(self) -> bool:
        """从磁盘加载向量库。返回 False 表示库不存在（需要 ingest）。"""
        ok = self.store.load()
        self._loaded = ok
        return ok

    def is_ready(self) -> bool:
        return self._loaded and len(self.store) > 0

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVE_TOP_K,
        min_score: float = RETRIEVE_MIN_SCORE,
    ) -> list[dict]:
        """
        语义检索。返回元素：
            {doc, section, chunk_id, text, score, path}
        分数低于 min_score 的会被过滤。
        """
        if not self.is_ready():
            raise RuntimeError(
                "Vector store not loaded. Run `python main.py --ingest` first."
            )

        q_vec = embed_query(query)
        hits = self.store.search(q_vec, top_k=top_k)

        results: list[dict] = []
        for meta, score in hits:
            if score < min_score:
                continue
            results.append({
                "doc": meta.get("doc"),
                "section": meta.get("section"),
                "chunk_id": meta.get("chunk_id"),
                "text": meta.get("text"),
                "path": meta.get("path"),
                "score": round(float(score), 4),
            })
        return results

    # ---------- 写入侧（供 ingest 命令调用） ----------
    def rebuild_from_chunks(
        self, chunks: list[dict], *, show_progress: bool = True
    ) -> None:
        """
        给定 chunks（来自 rag.ingest.ingest_all），重新构建向量库。
        会覆盖旧数据。
        """
        if not chunks:
            logger.warning("rebuild_from_chunks called with 0 chunks; skip")
            return

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        vectors = embed_texts(texts, show_progress=show_progress)

        self.store.reset()
        self.store.add(vectors, chunks)
        self.store.save()
        self._loaded = True


# ============================================================
# 多 namespace 缓存（docs / memory_facts 可并存）
# ============================================================

_RETRIEVERS: dict[str, Retriever] = {}


def get_retriever(namespace: str = "docs", *, autoload: bool = True) -> Retriever:
    """
    按 namespace 获取 retriever。同进程内同 namespace 复用。
    """
    if namespace not in _RETRIEVERS:
        r = Retriever(namespace=namespace)
        if autoload:
            r.load()
        _RETRIEVERS[namespace] = r
    return _RETRIEVERS[namespace]
