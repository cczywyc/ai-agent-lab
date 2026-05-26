"""
上下文装配器 — 设计四的实现

固定装配顺序 + 分段 token 预算 + 超预算决策流：

| 段 | 内容                             | 进出规则             |
|:--:|----------------------------------|----------------------|
| 1  | System prompt                    | 永远在               |
| 2  | 用户偏好（key-value 拼成短文本） | 永远在（很小）       |
| 3  | 历史摘要                         | 有则在               |
| 4  | 召回的长期事实                   | 语义 top-k           |
| 5  | 最近 K 轮对话                    | 双闸门裁剪           |
| 6  | 当前问题                         | 永远在               |

排序逻辑：越稳定/强约束越靠前。超预算时段 5 先裁 → 段 3/4 次之 → 段 1/2/6 不可裁。

输出是标准 OpenAI messages 列表，直接给 client.chat.completions.create。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import SEGMENT_BUDGETS, TOTAL_CONTEXT_BUDGET
from .long_term import LongTermMemory
from .short_term import ShortTermMemory, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class AssemblyReport:
    """装配产物的元信息（trace 用）。"""
    segments_present: list[str]
    segments_trimmed: list[str]
    total_chars: int
    facts_recalled: int


class ContextAssembler:
    """六段装配。"""

    def __init__(
        self,
        long_term: LongTermMemory,
        short_term: ShortTermMemory,
        summary_text_getter,  # callable: () -> str
        embed_query_fn,       # callable: (str) -> np.ndarray
    ):
        self.long = long_term
        self.short = short_term
        self.get_summary = summary_text_getter
        self.embed_query = embed_query_fn

    # ============================================================
    # 主入口
    # ============================================================
    def assemble(
        self,
        user_message: str,
        system_prompt: str,
    ) -> tuple[list[dict], AssemblyReport]:
        """
        返回 (messages, report)。messages 即可直接传给 LLM。
        """
        segments_present: list[str] = ["system"]
        segments_trimmed: list[str] = []
        facts_recalled_count = 0

        # ---- 段 1：system prompt ----
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # ---- 段 2：用户偏好（永远在）----
        pref_text = self._format_preferences()
        if pref_text:
            messages.append({"role": "system", "content": pref_text})
            segments_present.append("preferences")

        # ---- 段 3：历史摘要 ----
        summary = (self.get_summary() or "").strip()
        if summary:
            budget = SEGMENT_BUDGETS["summary"]
            if len(summary) > budget:
                summary = summary[-budget:]   # 保留最近的摘要
                segments_trimmed.append("summary")
            messages.append({
                "role": "system",
                "content": f"[历史对话摘要]\n{summary}",
            })
            segments_present.append("summary")

        # ---- 段 4：召回长期事实 ----
        facts = self._recall_facts(user_message)
        if facts:
            facts_recalled_count = len(facts)
            facts_text = self._format_facts(facts)
            budget = SEGMENT_BUDGETS["facts"]
            if len(facts_text) > budget:
                # 字符级截断，但保证不切坏整条事实——直接削尾
                facts_text = facts_text[:budget].rsplit("\n", 1)[0]
                segments_trimmed.append("facts")
            messages.append({"role": "system", "content": facts_text})
            segments_present.append("facts")

        # ---- 段 5：最近 K 轮（双闸门裁剪）----
        recent_msgs, recent_trimmed = self._format_recent_turns()
        if recent_msgs:
            messages.extend(recent_msgs)
            segments_present.append("recent")
            if recent_trimmed:
                segments_trimmed.append("recent")

        # ---- 段 6：当前问题（永远在）----
        messages.append({"role": "user", "content": user_message})
        segments_present.append("current")

        # ---- 总预算审计 ----
        total = sum(len(m.get("content") or "") for m in messages)
        if total > TOTAL_CONTEXT_BUDGET:
            logger.warning(
                f"Assembled context {total} chars exceeds total budget "
                f"{TOTAL_CONTEXT_BUDGET}. Will rely on LLM truncation."
            )

        report = AssemblyReport(
            segments_present=segments_present,
            segments_trimmed=segments_trimmed,
            total_chars=total,
            facts_recalled=facts_recalled_count,
        )
        return messages, report

    # ============================================================
    # 各段格式化
    # ============================================================
    def _format_preferences(self) -> str:
        prefs = self.long.preferences
        if not prefs:
            return ""
        lines = [f"- {k}: {v}" for k, v in prefs.items()]
        return "[用户偏好]\n" + "\n".join(lines)

    def _recall_facts(self, query: str) -> list[dict]:
        """调 retriever 语义召回。失败/库空返回 []。"""
        if not self.long.facts:
            return []
        try:
            q_vec = self.embed_query(query)
        except Exception as e:
            logger.warning(f"Embedding query for facts recall failed: {e}")
            return []
        return self.long.recall_facts(q_vec)

    @staticmethod
    def _format_facts(facts: list[dict]) -> str:
        lines = ["[相关长期事实（来自历史对话）]"]
        for f in facts:
            src = f.get("source", "")
            turn = f.get("turn", "?")
            lines.append(f"- {f['fact']} (来源: {src}, 第{turn}轮)")
        return "\n".join(lines)

    def _format_recent_turns(self) -> tuple[list[dict], bool]:
        """
        把最近 K 轮 ConversationTurn 还原成 user/assistant 消息序列。

        裁剪策略（设计文档"裁剪优先级"的简化版）：
          - 第一遍：原样还原
          - 第二遍：若总字符 > 段预算，从最旧一轮开始把 assistant 截断到 200 字
          - 工具结果不存到 short_term（设计上"可重新获取的原材料"），所以这里
            不需要处理工具历史
        """
        recent = self.short.recent()
        if not recent:
            return [], False

        msgs: list[dict] = []
        for t in recent:
            msgs.append({"role": "user", "content": t.user_message})
            msgs.append({"role": "assistant", "content": t.assistant_message})

        total = sum(len(m["content"]) for m in msgs)
        budget = SEGMENT_BUDGETS["recent"]
        if total <= budget:
            return msgs, False

        # 第二遍裁剪：旧轮的 assistant 内容截断
        # （保留 user 全文，因为 user 通常更短且不可重生）
        trimmed = False
        for i, m in enumerate(msgs):
            if total <= budget:
                break
            if m["role"] == "assistant" and len(m["content"]) > 200:
                saved = len(m["content"]) - 200
                m["content"] = m["content"][:200] + " ...[已裁剪]"
                total -= saved
                trimmed = True
        return msgs, trimmed
