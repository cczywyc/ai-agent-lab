"""
长期记忆 — 三类记忆

对应设计文档"设计二"：

| 类型     | 数据结构                    | 写入时机        | 进 prompt 方式 |
|----------|----------------------------|-----------------|----------------|
| 偏好     | key-value                  | 显式信号        | 每轮全量       |
| 已确认事实 | list[{fact,source,turn,..}] | 抽取自带引用回答 | 语义召回 top-k |
| 主题兴趣 | counter                    | 每轮规则抽取     | 召回加权（暂未实装，留接口）|

实现要点：
  - 事实存两份：json 文件存元数据（fact/source/turn/timestamp），
    向量库（namespace="memory_facts"）存对应向量
  - 通过 list 下标关联两者：metadata.json 第 i 项 ↔ vector 第 i 行
  - 偏好/主题只 json，无向量
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    MEMORY_DIR, MEMORY_PREFS_FILE, MEMORY_FACTS_FILE, MEMORY_TOPICS_FILE,
    MEMORY_FACTS_TOP_K, MEMORY_FACTS_MIN_SCORE, EMBEDDING_DIM,
)
from rag.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """单条已确认事实。"""
    fact: str
    source: str           # "doc#section[#chunk_id]" 或 "turn:N"
    turn: int             # 写入时的对话轮次
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class LongTermMemory:
    """三类长期记忆 + 持久化 + 事实语义召回。"""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or MEMORY_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 偏好：key-value，全量进 prompt
        self.preferences: dict[str, str] = {}

        # 已确认事实：dataclass 列表，与向量库下标对齐
        self.facts: list[Fact] = []

        # 主题计数：调用方可根据 count 排序做兴趣建模
        self.topics: dict[str, int] = {}

        # 事实向量库（懒加载）
        self._facts_store: Optional[VectorStore] = None

    # ============================================================
    # 偏好
    # ============================================================
    def set_preference(self, key: str, value: str) -> None:
        """显式信号触发；同 key 覆盖。"""
        key = key.strip()
        if not key:
            return
        self.preferences[key] = value.strip()
        logger.info(f"Preference set: {key} = {value}")

    def forget_preference(self, key: str) -> bool:
        return self.preferences.pop(key, None) is not None

    # ============================================================
    # 主题
    # ============================================================
    def bump_topics(self, topics: list[str]) -> None:
        for t in topics:
            t = t.strip()
            if not t:
                continue
            self.topics[t] = self.topics.get(t, 0) + 1

    def top_topics(self, n: int = 5) -> list[tuple[str, int]]:
        return sorted(self.topics.items(), key=lambda x: -x[1])[:n]

    # ============================================================
    # 已确认事实（带语义召回）
    # ============================================================
    def _ensure_facts_store(self) -> VectorStore:
        if self._facts_store is None:
            self._facts_store = VectorStore(
                namespace="memory_facts",
                persist_dir=self.persist_dir,
            )
            self._facts_store.load()  # 静默允许首次空
        return self._facts_store

    def add_fact(self, fact: Fact, vector: Optional[np.ndarray] = None) -> None:
        """
        追加事实。向量可由调用方预先算好传入，避免重复 embed。
        若 vector=None，只入 json 不入向量库（不可召回）。
        """
        self.facts.append(fact)
        if vector is not None:
            store = self._ensure_facts_store()
            # 形状统一为 (1, D)
            v = vector.reshape(1, -1) if vector.ndim == 1 else vector
            # 同步元数据（与 self.facts 下标对齐）
            store.add(v, [fact.to_dict()])

    def recall_facts(
        self,
        query_vector: np.ndarray,
        top_k: int = MEMORY_FACTS_TOP_K,
        min_score: float = MEMORY_FACTS_MIN_SCORE,
    ) -> list[dict]:
        """语义召回 top-k 相关事实。返回 [{fact, source, turn, score}, ...]"""
        store = self._ensure_facts_store()
        if len(store) == 0:
            return []
        hits = store.search(query_vector, top_k=top_k)
        return [
            {**meta, "score": round(float(score), 4)}
            for meta, score in hits
            if score >= min_score
        ]

    # ============================================================
    # 持久化
    # ============================================================
    @property
    def _prefs_path(self) -> Path:
        return self.persist_dir / MEMORY_PREFS_FILE

    @property
    def _facts_path(self) -> Path:
        return self.persist_dir / MEMORY_FACTS_FILE

    @property
    def _topics_path(self) -> Path:
        return self.persist_dir / MEMORY_TOPICS_FILE

    def save(self) -> None:
        with open(self._prefs_path, "w", encoding="utf-8") as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=2)
        with open(self._facts_path, "w", encoding="utf-8") as f:
            json.dump([fa.to_dict() for fa in self.facts], f,
                      ensure_ascii=False, indent=2)
        with open(self._topics_path, "w", encoding="utf-8") as f:
            json.dump(self.topics, f, ensure_ascii=False, indent=2)
        # 向量库（如果有）
        if self._facts_store is not None and len(self._facts_store) > 0:
            self._facts_store.save()

    def load(self) -> None:
        if self._prefs_path.exists():
            with open(self._prefs_path, "r", encoding="utf-8") as f:
                self.preferences = json.load(f)
        if self._facts_path.exists():
            with open(self._facts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.facts = [Fact(**d) for d in data]
        if self._topics_path.exists():
            with open(self._topics_path, "r", encoding="utf-8") as f:
                self.topics = json.load(f)
        # 触发向量库懒加载
        self._ensure_facts_store()

        logger.info(
            f"LongTermMemory loaded: {len(self.preferences)} prefs, "
            f"{len(self.facts)} facts, {len(self.topics)} topics"
        )

    def clear(self) -> None:
        """清空内存中数据并删除持久化文件（含向量库）。"""
        self.preferences = {}
        self.facts = []
        self.topics = {}
        for p in (self._prefs_path, self._facts_path, self._topics_path):
            if p.exists():
                p.unlink()
        # 清向量库
        store = self._ensure_facts_store()
        store.reset()
        for p in (store._vec_path, store._meta_path):
            if p.exists():
                p.unlink()
        self._facts_store = None
