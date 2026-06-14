"""
长期记忆 — 三类记忆 × LangGraph Store（v4.1）

对应设计文档"设计二"，迁移后的分工：

| 类型     | namespace             | 写入时机        | 进 prompt 方式 |
|----------|-----------------------|-----------------|----------------|
| 偏好     | ("ltm","preferences") | 显式信号        | 每轮全量       |
| 已确认事实 | ("ltm","facts")       | 抽取自带引用回答 | 语义召回 top-k |
| 主题兴趣 | ("ltm","topics")      | 每轮规则抽取     | 召回加权（暂未实装，留接口）|

v4.0 → v4.1 的变化（计划 §二）：
  - "存储 + 召回 + 持久化"整体交给 store：json 三件套与
    namespace="memory_facts" 向量库退役，Fact↔向量"下标对齐"的手工同步消失
  - 事实写入不再手动 embed（store.batch 批量 put，索引由 store 内部建）
  - 召回不再传 query_vector（store.search 原生 query 入参，内部自己 embed）
  - "挑选"常量（top_k / min_score）留在调用方这侧过滤，不交给 store

本模块瘦身为：Fact schema 约定 + 三个 namespace 的读写函数（store 由调用方传入，
来源是 LangGraph 节点注入或 main 入口的 get_ltm_store()）。
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field, asdict

from langgraph.store.base import BaseStore, PutOp

from config import MEMORY_FACTS_TOP_K, MEMORY_FACTS_MIN_SCORE
from .ltm_store import NS_PREFS, NS_FACTS, NS_TOPICS

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """单条已确认事实（store value 的 schema 约定）。"""
    fact: str
    source: str           # "doc#section[#chunk_id]" 或 "turn:N"
    turn: int             # 写入时的对话轮次
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def fact_key(fact_text: str) -> str:
    """事实的 store key：原文哈希——同文重写幂等（迁移脚本重跑不产生重复）。"""
    return hashlib.sha1(fact_text.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 内部：翻页列全（BaseStore.search 单页有 limit，计数/清空要翻完）
# ============================================================
_PAGE = 100


def _list_all(store: BaseStore, namespace: tuple) -> list:
    out, offset = [], 0
    while True:
        page = store.search(namespace, limit=_PAGE, offset=offset)
        out.extend(page)
        if len(page) < _PAGE:
            return out
        offset += _PAGE


# ============================================================
# 偏好
# ============================================================
def set_preference(store: BaseStore, key: str, value: str) -> None:
    """显式信号触发；同 key 覆盖。index=False：纯 kv，不进语义索引。"""
    key = key.strip()
    if not key:
        return
    store.put(NS_PREFS, key, {"value": value.strip()}, index=False)
    logger.info(f"Preference set: {key} = {value}")


def forget_preference(store: BaseStore, key: str) -> bool:
    if store.get(NS_PREFS, key) is None:
        return False
    store.delete(NS_PREFS, key)
    return True


def list_preferences(store: BaseStore) -> dict[str, str]:
    """全量偏好（段 2 每轮全进 prompt，量小）。"""
    return {it.key: it.value.get("value", "") for it in _list_all(store, NS_PREFS)}


# ============================================================
# 主题
# ============================================================
def bump_topics(store: BaseStore, topics: list[str]) -> None:
    """get 现值 +1 后 put。index=False：纯计数，不进语义索引。"""
    for t in topics:
        t = t.strip()
        if not t:
            continue
        cur = store.get(NS_TOPICS, t)
        n = (cur.value.get("count", 0) if cur else 0) + 1
        store.put(NS_TOPICS, t, {"count": n}, index=False)


def top_topics(store: BaseStore, n: int = 5) -> list[tuple[str, int]]:
    items = [(it.key, it.value.get("count", 0)) for it in _list_all(store, NS_TOPICS)]
    return sorted(items, key=lambda x: -x[1])[:n]


# ============================================================
# 已确认事实（语义召回交给 store）
# ============================================================
def add_facts(store: BaseStore, facts: list[Fact]) -> None:
    """
    批量写入事实。store.batch 把整批的 embed 合并成一次 API 调用
    （与 v4.0 手动 embed_texts 批量等价的经济性）。
    嵌入失败时降级为 index=False 写入：数据不丢，只是不可语义召回
    （与 v4.0 "embed 失败只入 json 不入向量库"的行为一致）。
    """
    if not facts:
        return
    ops = [PutOp(namespace=NS_FACTS, key=fact_key(f.fact), value=f.to_dict())
           for f in facts]
    try:
        store.batch(ops)
    except Exception as e:  # noqa: BLE001 — embed/后端失败都不该丢数据
        logger.warning(
            f"Indexed batch put failed: {e}. "
            f"Storing facts without index (not semantically recallable)."
        )
        for f in facts:
            store.put(NS_FACTS, fact_key(f.fact), f.to_dict(), index=False)


def recall_facts(
    store: BaseStore,
    query: str,
    top_k: int = MEMORY_FACTS_TOP_K,
    min_score: float = MEMORY_FACTS_MIN_SCORE,
) -> list[dict]:
    """
    语义召回 top-k 相关事实，返回 [{fact, source, turn, ..., score}, ...]。
    native search 内部 embed query；min_score 阈值留在这侧过滤（与 v4.0 语义一致）。
    score 为 None 的条目（降级写入的未索引事实）不参与召回。
    """
    hits = store.search(NS_FACTS, query=query, limit=top_k)
    return [
        {**h.value, "score": round(float(h.score), 4)}
        for h in hits
        if h.score is not None and h.score >= min_score
    ]


# ============================================================
# 计数 / 清空（info、--reset-memory 用）
# ============================================================
def counts(store: BaseStore) -> dict[str, int]:
    return {
        "preferences": len(_list_all(store, NS_PREFS)),
        "facts": len(_list_all(store, NS_FACTS)),
        "topics": len(_list_all(store, NS_TOPICS)),
    }


def clear_all(store: BaseStore) -> None:
    """清空三个 namespace（BaseStore 无整 namespace 删除接口，逐条 delete）。"""
    for ns in (NS_PREFS, NS_FACTS, NS_TOPICS):
        for it in _list_all(store, ns):
            store.delete(ns, it.key)
