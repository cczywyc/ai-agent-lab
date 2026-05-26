"""
Agent Trace 数据结构 — v3.0
三层结构：AgentTrace → TurnTrace → ToolCallTrace

v3.0 变更：
  - ToolCallTrace 增加 retrieved_chunks 字段（仅 retrieve_documents 调用时填充）
  - TurnTrace 增加 retrieval_correction_injected
  - AgentTrace 增加 retrieved / retrieval_correction_triggered
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


@dataclass
class ToolCallTrace:
    """单次工具调用的追踪记录"""
    tool_name: str
    tool_args: dict
    result_success: bool
    result_summary: str
    error_type: Optional[str] = None
    duration_ms: int = 0
    # v3.0：retrieve_documents 调用时，记录召回的 chunk 简要信息（不存全文）
    retrieved_chunks: list = field(default_factory=list)
    # 每个元素结构：{doc, section, chunk_id, score}


@dataclass
class TurnTrace:
    """单轮 Agent 决策的追踪记录"""
    turn_number: int
    finish_reason: str  # "stop" | "tool_calls"
    tool_calls: list = field(default_factory=list)
    correction_injected: bool = False
    fallback_injected: bool = False
    retrieval_correction_injected: bool = False  # v3.0


@dataclass
class AgentTrace:
    """完整对话的追踪记录"""
    user_question: str
    total_turns: int = 0
    turns: list = field(default_factory=list)
    final_answer: Optional[str] = None
    searched: bool = False               # 是否调用了 web_search
    retrieved: bool = False              # v3.0：是否调用了 retrieve_documents
    correction_triggered: bool = False
    fallback_triggered: bool = False
    retrieval_correction_triggered: bool = False  # v3.0
    total_duration_ms: int = 0

    def add_turn(self, turn: TurnTrace):
        self.turns.append(turn)
        self.total_turns = len(self.turns)

    def finalize(self, answer: str, start_time: float):
        self.final_answer = answer
        self.total_duration_ms = int((time.time() - start_time) * 1000)
        self.searched = any(
            tc.tool_name == "web_search"
            for t in self.turns for tc in t.tool_calls
        )
        self.retrieved = any(
            tc.tool_name == "retrieve_documents"
            for t in self.turns for tc in t.tool_calls
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        tools_used = []
        for t in self.turns:
            for tc in t.tool_calls:
                tools_used.append(
                    f"{tc.tool_name}({'✓' if tc.result_success else '✗'})"
                )
        tools_str = " → ".join(tools_used) if tools_used else "无工具调用"

        flags = []
        if self.correction_triggered:
            flags.append("搜索纠正")
        if self.retrieval_correction_triggered:
            flags.append("检索纠正")
        if self.fallback_triggered:
            flags.append("降级")
        flags_str = f" [{','.join(flags)}]" if flags else ""

        return (
            f"[{self.total_turns}轮 | {self.total_duration_ms}ms] "
            f"{tools_str}{flags_str}"
        )
