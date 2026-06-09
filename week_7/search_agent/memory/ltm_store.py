"""
ltm_store.py — 长期记忆的 LangGraph Store 工厂（v4.1，落地决策 E）

三个 namespace 的约定（计划 §二）：
  ("ltm", "preferences") — 偏好，纯 kv，put/get/list，无语义索引
  ("ltm", "facts")       — 已确认事实，配语义索引（store 内部自管 embed）
  ("ltm", "topics")      — 主题计数，纯 kv，get+1+put

语义索引只给 facts：index 配置 fields=["fact"]，且 prefs/topics 写入时
显式 index=False（否则 store 级 index 对所有 put 生效，主题计数也会白调 embedding API）。

E2c 的取舍：失去"共用向量库"（facts 向量从 rag/store.py 挪进 store 自管），
保留"共用 embedding 模型"（embed_for_store 仍包装 rag.embed 的 text-embedding-v3）。

后端：
  - make_inmemory_store —— 开发/离线测试（可注入 stub embed）
  - make_sqlite_store  —— 本地文件级持久化（MEMORY_DIR/ltm.db）
  - get_ltm_store —— 进程级单例，默认 SqliteStore（计划 §三选 C-SqliteStore）

SqliteStore 连接注意：必须 isolation_level=None（autocommit）——
原生 sqlite3 连接默认隐式开事务，会和 store 内部的 BEGIN 冲突
（from_conn_string 是 contextmanager，进程级长连接要自己建）。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Callable, Optional

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

from config import EMBEDDING_DIM, MEMORY_DIR

logger = logging.getLogger(__name__)

# 长期记忆持久化文件（与 json/向量库同目录，便于对照与备份）
LTM_DB_PATH = Path(MEMORY_DIR) / "ltm.db"

# ============================================================
# Namespace 约定
# ============================================================
NS_PREFS = ("ltm", "preferences")
NS_FACTS = ("ltm", "facts")
NS_TOPICS = ("ltm", "topics")


# ============================================================
# embed 适配：rag.embed → LangGraph index 要的 (list[str]) -> list[list[float]]
# ============================================================
def embed_for_store(texts: list[str]) -> list[list[float]]:
    """复用 text-embedding-v3（已 L2 归一化，store 的余弦搜索直接可用）。"""
    from rag.embed import embed_texts  # 延迟导入：测试可完全离线
    return embed_texts(list(texts)).tolist()


def make_index_config(
    embed_fn: Optional[Callable] = None, dims: Optional[int] = None,
) -> dict:
    """语义索引配置——只对 fact 字段建索引。"""
    return {
        "embed": embed_fn or embed_for_store,
        "dims": dims or EMBEDDING_DIM,
        "fields": ["fact"],
    }


def make_inmemory_store(
    embed_fn: Optional[Callable] = None, dims: Optional[int] = None,
) -> InMemoryStore:
    """内存版 store：开发期接口跑通 / 离线测试（传 stub embed 即可零 API）。"""
    return InMemoryStore(index=make_index_config(embed_fn, dims))


def make_sqlite_store(
    db_path: Optional[Path] = None,
    embed_fn: Optional[Callable] = None,
) -> SqliteStore:
    """
    持久化版 store。连接进程级长活（不走 from_conn_string 的 contextmanager），
    isolation_level=None 让 store 自管事务。
    """
    path = Path(db_path or LTM_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,  # autocommit：避免与 store 内部 BEGIN 冲突
    )
    store = SqliteStore(conn, index=make_index_config(embed_fn))
    store.setup()  # 建表（幂等）
    return store


# ============================================================
# 进程级单例（与 get_memory 同款模式）
# ============================================================
_STORE_SINGLETON: Optional[BaseStore] = None


def get_ltm_store() -> BaseStore:
    """
    长期记忆 store 单例。graph.compile(store=...) 与 main.py 的记忆工具共用。
    默认 SqliteStore（本地 ltm.db）：跨进程持久化 + facts 语义索引。
    """
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        _STORE_SINGLETON = make_sqlite_store()
        logger.info(f"LTM store initialized: SqliteStore @ {LTM_DB_PATH}")
    return _STORE_SINGLETON
