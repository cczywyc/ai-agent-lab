"""
MemoryManager — 顶层协调

对 agent.py 的暴露面只有两个方法：
  - assemble_context(user_message, system_prompt) → (messages, AssemblyReport)
  - update_from_turn(user_message, assistant_message, trace) → None

内部职责：
  1. 协调 short_term / long_term / summarizer / assembler
  2. 每轮规则抽取主题、偏好、事实候选
  3. 在 needs_eviction 时调摘要、做事实晋升、evict 旧轮
  4. 持久化（每轮结束）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import (
    SHORT_TERM_K, SHORT_TERM_CHAR_BUDGET,
    SUMMARY_TRIGGER_TURNS, MEMORY_DIR,
)
from rag.embed import embed_query
from .assembler import AssemblyReport, ContextAssembler
from .extractor import (
    extract_topics, extract_preference, extract_fact_candidates,
)
from .long_term import Fact, LongTermMemory
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
        self.long = LongTermMemory(persist_dir=self.persist_dir)
        self.summarizer = MemorySummarizer(persist_dir=self.persist_dir)
        self.assembler = ContextAssembler(
            long_term=self.long,
            short_term=self.short,
            summary_text_getter=lambda: self.summarizer.summary_text,
            embed_query_fn=embed_query,
        )

        if autoload:
            self.load()

    # ============================================================
    # 装配：每轮 LLM 调用前
    # ============================================================
    def assemble_context(
        self, user_message: str, system_prompt: str,
    ) -> tuple[list[dict], AssemblyReport]:
        return self.assembler.assemble(user_message, system_prompt)

    # ============================================================
    # 更新：每轮 LLM 调用后
    # ============================================================
    def update_from_turn(
        self,
        user_message: str,
        assistant_message: str,
        trace,  # AgentTrace（避免引入硬依赖，用 duck typing）
    ) -> None:
        """
        每轮结束时调用。完成：
          1. 主题计数（user_message + assistant_message）
          2. 偏好显式信号识别
          3. 事实候选抽取并写入长期记忆（带向量）
          4. 把本轮加进短期记忆
          5. 必要时触发摘要 + evict
          6. 持久化
        """
        turn_no = self.short.next_turn_number()

        # 1. 主题
        topics = extract_topics(user_message) + extract_topics(assistant_message)
        if topics:
            self.long.bump_topics(topics)

        # 2. 偏好
        pref = extract_preference(user_message)
        if pref:
            self.long.set_preference(pref["key"], pref["value"])

        # 3. 已确认事实（仅当回答带 [doc#section] 引用）
        self._extract_and_store_facts(assistant_message, turn_no)

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
    # 私有：事实抽取 + 向量化
    # ============================================================
    def _extract_and_store_facts(self, answer: str, turn_no: int) -> None:
        candidates = extract_fact_candidates(answer)
        if not candidates:
            return

        # 批量 embed 一次，避免每条 fact 单独调 API
        texts = [fact for fact, _src in candidates]
        try:
            from rag.embed import embed_texts
            vectors = embed_texts(texts)
        except Exception as e:
            logger.warning(
                f"Embed fact candidates failed: {e}. "
                f"Storing facts without vectors (not semantically recallable)."
            )
            vectors = None

        for i, (fact_text, source) in enumerate(candidates):
            f = Fact(fact=fact_text, source=source, turn=turn_no)
            v = vectors[i] if vectors is not None else None
            self.long.add_fact(f, vector=v)

        if candidates:
            logger.info(f"Promoted {len(candidates)} facts to long-term (turn {turn_no})")

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
    # 持久化
    # ============================================================
    def save(self) -> None:
        self.long.save()
        self.summarizer.save()
        # 短期记忆也落盘，允许跨会话恢复对话
        self._save_short_term()

    def load(self) -> None:
        self.long.load()
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
    def reset(self) -> None:
        self.short.clear()
        self.long.clear()
        self.summarizer.clear()
        if self._short_term_path.exists():
            self._short_term_path.unlink()
        logger.info("Memory has been reset.")

    # ============================================================
    # 自省（调试用）
    # ============================================================
    def info(self) -> dict:
        return {
            "short_term_turns": len(self.short.turns),
            "preferences": dict(self.long.preferences),
            "facts": len(self.long.facts),
            "topics_top5": self.long.top_topics(5),
            "summary_chars": len(self.summarizer.summary_text),
        }


# ============================================================
# 单例
# ============================================================

_MEMORY_SINGLETON: Optional[MemoryManager] = None


def get_memory(reset: bool = False) -> MemoryManager:
    global _MEMORY_SINGLETON
    if _MEMORY_SINGLETON is None:
        _MEMORY_SINGLETON = MemoryManager(autoload=True)
    if reset:
        _MEMORY_SINGLETON.reset()
    return _MEMORY_SINGLETON
