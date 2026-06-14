"""
Embedding 模块 — 封装 DashScope text-embedding-v3

接口：
  embed_texts(texts: list[str], text_type: str = "document") -> np.ndarray
  embed_query(text: str) -> np.ndarray

设计要点：
  - 复用 config.client（OpenAI 兼容客户端）
  - 自动分批（兼容接口单次 ≤10 条）
  - 区分 query / document 两种语义（DashScope text_type 参数）
  - 出错时不静默，抛异常让上层处理
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np

from config import client, EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """单批调用 embedding API，返回每条文本对应的向量。"""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    # OpenAI 风格返回：resp.data[i].embedding
    return [item.embedding for item in resp.data]


def embed_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
    sleep_between: float = 0.0,
    show_progress: bool = False,
) -> np.ndarray:
    """
    批量嵌入文本。

    Args:
        texts: 文本列表
        batch_size: 单批大小（None 则取 config.EMBEDDING_BATCH_SIZE）
        sleep_between: 批次间睡眠秒数（防限流）
        show_progress: 是否打印进度

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM)，已 L2 归一化
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    batch_size = batch_size or EMBEDDING_BATCH_SIZE
    all_vectors: list[list[float]] = []

    total = len(texts)
    for start in range(0, total, batch_size):
        chunk = texts[start:start + batch_size]
        if show_progress:
            print(f"  embed batch {start + 1}-{start + len(chunk)} / {total}")

        try:
            vectors = _embed_batch(chunk)
        except Exception as e:
            logger.error(
                f"Embedding batch {start}-{start + len(chunk)} failed: {e}"
            )
            raise

        all_vectors.extend(vectors)

        if sleep_between > 0 and (start + batch_size) < total:
            time.sleep(sleep_between)

    arr = np.asarray(all_vectors, dtype=np.float32)
    # L2 归一化：检索时余弦相似度 = 点积，省一次除法
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # 防零向量
    arr = arr / norms

    return arr


def embed_query(text: str) -> np.ndarray:
    """单条查询文本嵌入。返回 shape (EMBEDDING_DIM,)。"""
    arr = embed_texts([text])
    return arr[0]
