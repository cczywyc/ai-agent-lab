"""
短期记忆 — 对话历史裁剪

设计要点（对应设计文档"设计一"）：
  - 保留窗口用「轮次 + token 双闸门」，超出走摘要而非硬丢弃
  - 默认保留 system + 最近 K 轮，同时设字符预算
  - 裁剪优先级（v3.0 适配，最该丢 → 最不该丢）：
    1. 旧的检索 chunk 正文（可重新检索）
    2. 旧的网页正文
    3. 旧的工具调用记录
    4. 旧的模型回答
    5. 最近 K 轮（不丢）
  - 超限不硬丢，先送摘要再丢

实现简化：
  - 每个 Turn = 一次 user → assistant 完整轮次（不存中间工具结果，
    那些都是"可重新获取的原材料"——设计文档明确说丢这部分最划算）
  - 短期记忆只持有 turns 列表；超 K 轮的部分由 manager 调摘要后 evict
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import time


@dataclass
class ConversationTurn:
    """单轮对话（user + assistant final answer + 元信息）。"""
    turn_number: int           # 全局轮次序号（从 1 开始）
    user_message: str
    assistant_message: str
    # 工具调用摘要：["retrieve_documents(query=...) → 5 chunks", ...]
    tool_summaries: list[str] = field(default_factory=list)
    # 本轮是否触发了任何工具
    used_retrieve: bool = False
    used_search: bool = False
    timestamp: float = field(default_factory=time.time)

    def char_count(self) -> int:
        return (
            len(self.user_message)
            + len(self.assistant_message)
            + sum(len(s) for s in self.tool_summaries)
        )

    def to_dict(self) -> dict:
        return asdict(self)


class ShortTermMemory:
    """
    持有完整轮次列表，对外提供"保留哪些"的判断。
    实际 evict（送摘要再删）由 MemoryManager 协调。
    """

    def __init__(self, k: int, char_budget: int):
        self.k = k
        self.char_budget = char_budget
        self.turns: list[ConversationTurn] = []

    # ---------- 写入 ----------
    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def next_turn_number(self) -> int:
        return (self.turns[-1].turn_number if self.turns else 0) + 1

    # ---------- 查询 ----------
    def recent(self, k: Optional[int] = None) -> list[ConversationTurn]:
        """返回最近 k 轮（默认 self.k）。"""
        n = self.k if k is None else k
        return self.turns[-n:] if n > 0 else []

    def overflow(self) -> list[ConversationTurn]:
        """返回超出 K 窗口的旧轮次（候选摘要对象）。"""
        if len(self.turns) <= self.k:
            return []
        return self.turns[:-self.k]

    def total_chars_recent(self) -> int:
        return sum(t.char_count() for t in self.recent())

    def needs_eviction(self) -> bool:
        """
        是否需要 evict（送摘要）：
          - 轮次闸门：超 K 轮
          - 字符闸门：最近 K 轮的字符数 > 预算（说明单轮过大，
            也提示摘要应该介入压缩）
        """
        if len(self.turns) > self.k:
            return True
        if self.total_chars_recent() > self.char_budget:
            return True
        return False

    # ---------- 持久化辅助 ----------
    def snapshot(self) -> list[dict]:
        return [t.to_dict() for t in self.turns]

    def load_snapshot(self, data: list[dict]) -> None:
        self.turns = [ConversationTurn(**d) for d in data]

    def clear(self) -> None:
        self.turns = []
