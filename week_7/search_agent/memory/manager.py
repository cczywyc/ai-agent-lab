"""
MemoryManager — 顶层协调

对节点层的暴露面只有两个方法（v4.1 起都多收一个 store 入参）：
  - assemble_context(user_message, system_prompt, store) → (messages, AssemblyReport)
  - update_from_turn(user_message, assistant_message, trace, store) → None

内部职责：
  1. 协调 short_term / long_term(=store 操作) / summarizer / assembler
  2. 每轮规则抽取主题、偏好、事实候选
  3. 在 needs_eviction 时调摘要、做事实晋升、evict 旧轮
  4. 持久化（每轮结束；v4.1 起只剩短期 + 摘要——长期记忆由 store 后端自管）

v4.0 → v4.1：
  - 长期记忆三类的读写经 LangGraph Store（store 由 LangGraph 节点注入，
    或 main 入口用 get_ltm_store() 取单例传入）
  - 删除长期记忆对 embed_query/embed_texts 的用法（rag.embed 仍被 rag 文档检索用）
  - 短期 + 摘要 + extractor 不动
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langgraph.store.base import BaseStore

from config import (
    SHORT_TERM_K, SHORT_TERM_CHAR_BUDGET,
    SUMMARY_TRIGGER_TURNS, MEMORY_DIR,
)
from . import long_term
from .assembler import AssemblyReport, ContextAssembler
from .extractor import (
    extract_topics, extract_preference, extract_fact_candidates,
)
from .long_term import Fact
from .short_term import ConversationTurn, ShortTermMemory
from .summarizer import MemorySummarizer

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆系统的对外门面。"""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        *,
        autoload: bool = True,
    ):
        self.persist_dir = Path(persist_dir or MEMORY_DIR)
        self.short = ShortTermMemory(
            k=SHORT_TERM_K, char_budget=SHORT_TERM_CHAR_BUDGET,
        )
        self.summarizer = MemorySummarizer(persist_dir=self.persist_dir)
        self.assembler = ContextAssembler(
            short_term=self.short,
            summary_text_getter=lambda: self.summarizer.summary_text,
        )

        if autoload:
            self.load()

    # ============================================================
    # 装配：每轮 LLM 调用前
    # ============================================================
    def assemble_context(
        self, user_message: str, system_prompt: str, store: BaseStore,
    ) -> tuple[list[dict], AssemblyReport]:
        return self.assembler.assemble(user_message, system_prompt, store)

    # ============================================================
    # 更新：每轮 LLM 调用后
    # ============================================================
    def update_from_turn(
        self,
        user_message: str,
        assistant_message: str,
        trace,  # AgentTrace duck typing（v4.0 起为节点构造的 shim）
        store: BaseStore,
    ) -> None:
        """
        每轮结束时调用。完成：
          1. 主题计数（user_message + assistant_message）→ store
          2. 偏好显式信号识别 → store
          3. 事实候选抽取并写入长期记忆（store 自建语义索引）
          4. 把本轮加进短期记忆
          5. 必要时触发摘要 + evict
          6. 持久化（短期 + 摘要；长期由 store 后端自管）
        """
        turn_no = self.short.next_turn_number()

        # 1. 主题
        topics = extract_topics(user_message) + extract_topics(assistant_message)
        if topics:
            long_term.bump_topics(store, topics)

        # 2. 偏好
        pref = extract_preference(user_message)
        if pref:
            long_term.set_preference(store, pref["key"], pref["value"])

        # 3. 已确认事实（仅当回答带 [doc#section] 引用；不再手动 embed）。
        #    v4.2 踩坑 #3 收紧：引用还须对上本轮真实检索来源（trace 携带
        #    retrieved_chunks）。不带该字段的旧调用方 → None = 不校验，向后兼容。
        chunks = getattr(trace, "retrieved_chunks", None)
        allowed_sources = (
            {(str(c.get("doc", "")).strip(), str(c.get("section", "")).strip())
             for c in chunks}
            if chunks is not None else None
        )
        self._extract_and_store_facts(assistant_message, turn_no, store, allowed_sources)

        # 4. 加入短期记忆
        tool_summaries = self._summarize_tools(trace)
        turn = ConversationTurn(
            turn_number=turn_no,
            user_message=user_message,
            assistant_message=assistant_message,
            tool_summaries=tool_summaries,
            used_retrieve=getattr(trace, "retrieved", False),
            used_search=getattr(trace, "searched", False),
        )
        self.short.add_turn(turn)

        # 5. 触发摘要 + evict
        self._maybe_evict_and_summarize(turn_no)

        # 6. 持久化
        self.save()

    # ============================================================
    # 私有：事实抽取（向量化由 store 接管）
    # ============================================================
    def _extract_and_store_facts(
        self, answer: str, turn_no: int, store: BaseStore,
        allowed_sources: Optional[set] = None,
    ) -> None:
        candidates = extract_fact_candidates(answer, allowed_sources=allowed_sources)
        if not candidates:
            return

        facts = [Fact(fact=fact_text, source=source, turn=turn_no)
                 for fact_text, source in candidates]
        long_term.add_facts(store, facts)
        logger.info(f"Promoted {len(facts)} facts to long-term (turn {turn_no})")

    # ============================================================
    # 私有：工具摘要
    # ============================================================
    @staticmethod
    def _summarize_tools(trace) -> list[str]:
        """提取本轮所有工具调用的简要摘要。"""
        out: list[str] = []
        for t in getattr(trace, "turns", []):
            for tc in getattr(t, "tool_calls", []):
                marker = "✓" if tc.result_success else "✗"
                out.append(f"{tc.tool_name}({marker})")
        return out

    # ============================================================
    # 私有：摘要触发与 evict
    # ============================================================
    def _maybe_evict_and_summarize(self, turn_no: int) -> None:
        """
        触发条件（任一）：
          - 短期记忆需要 evict（超 K 轮或超字符预算）
          - 每 SUMMARY_TRIGGER_TURNS 轮兜底
        """
        # 没超 K 就不动
        if not self.short.needs_eviction():
            return

        overflow = self.short.overflow()
        if not overflow:
            # 边界：超字符预算但只有 K 轮——这种情况下我们不动，
            # 等下一轮来临时再判
            return

        logger.info(
            f"Evicting {len(overflow)} overflow turn(s) at turn {turn_no}. "
            f"Promoting facts + producing summary."
        )

        # （事实已经在 update_from_turn 当轮就抽过了；这里只做低保真摘要）
        new_summary = self.summarizer.summarize_low_fidelity(overflow)
        self.summarizer.append_summary(new_summary)

        # evict：保留最近 K 轮，其余丢弃
        self.short.turns = self.short.turns[-self.short.k:]

    # ============================================================
    # 持久化（v4.1：只剩短期 + 摘要；长期记忆的持久化是 store 后端的事）
    # ============================================================
    def save(self) -> None:
        self.summarizer.save()
        # 短期记忆也落盘，允许跨会话恢复对话
        self._save_short_term()

    def load(self) -> None:
        self.summarizer.load()
        self._load_short_term()

    @property
    def _short_term_path(self) -> Path:
        return self.persist_dir / "memory_short_term.json"

    def _save_short_term(self) -> None:
        import json
        with open(self._short_term_path, "w", encoding="utf-8") as f:
            json.dump(self.short.snapshot(), f, ensure_ascii=False, indent=2)

    def _load_short_term(self) -> None:
        import json
        p = self._short_term_path
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            self.short.load_snapshot(json.load(f))

    # ============================================================
    # 重置（--reset-memory 用）
    # ============================================================
    def reset(self, store: BaseStore) -> None:
        self.short.clear()
        long_term.clear_all(store)
        self.summarizer.clear()
        if self._short_term_path.exists():
            self._short_term_path.unlink()
        logger.info("Memory has been reset.")

    # ============================================================
    # 自省（调试用）
    # ============================================================
    def info(self, store: BaseStore) -> dict:
        c = long_term.counts(store)
        return {
            "short_term_turns": len(self.short.turns),
            "preferences": long_term.list_preferences(store),
            "facts": c["facts"],
            "topics_top5": long_term.top_topics(store, 5),
            "summary_chars": len(self.summarizer.summary_text),
        }


# ============================================================
# 单例
# ============================================================

_MEMORY_SINGLETON: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    global _MEMORY_SINGLETON
    if _MEMORY_SINGLETON is None:
        _MEMORY_SINGLETON = MemoryManager(autoload=True)
    return _MEMORY_SINGLETON
