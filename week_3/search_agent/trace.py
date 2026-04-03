"""
Agent Trace 数据结构
让 Agent 的每一步决策可追踪、可审计、可评测。

三层结构：
  AgentTrace → TurnTrace → ToolCallTrace

用法：
  trace = AgentTrace(user_question="...")
  trace.add_turn(turn_trace)
  print(trace.to_json())
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
    result_summary: str  # 结果摘要（截断到 200 字符）
    error_type: Optional[str] = None
    duration_ms: int = 0


@dataclass
class TurnTrace:
    """单轮 Agent 决策的追踪记录"""
    turn_number: int
    finish_reason: str  # "stop" | "tool_calls"
    tool_calls: list = field(default_factory=list)  # list[ToolCallTrace]
    correction_injected: bool = False  # 本轮是否注入了纠正指令
    fallback_injected: bool = False  # 本轮是否注入了降级指令


@dataclass
class AgentTrace:
    """完整对话的追踪记录"""
    user_question: str
    total_turns: int = 0
    turns: list = field(default_factory=list)  # list[TurnTrace]
    final_answer: Optional[str] = None
    searched: bool = False  # 是否使用了搜索工具
    correction_triggered: bool = False  # 是否触发了"应该搜索但没搜索"检测
    fallback_triggered: bool = False  # 是否触发了降级
    total_duration_ms: int = 0

    def add_turn(self, turn: TurnTrace):
        """添加一轮追踪记录"""
        self.turns.append(turn)
        self.total_turns = len(self.turns)

    def finalize(self, answer: str, start_time: float):
        """最终化 trace 记录"""
        self.final_answer = answer
        self.total_duration_ms = int((time.time() - start_time) * 1000)
        # 检查是否有任何轮次使用了 web_search
        self.searched = any(
            tc.tool_name == "web_search"
            for t in self.turns
            for tc in t.tool_calls
        )

    def to_dict(self) -> dict:
        """转为字典（递归处理 dataclass）"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """转为格式化 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        """生成人类可读的单行摘要"""
        tools_used = []
        for t in self.turns:
            for tc in t.tool_calls:
                tools_used.append(
                    f"{tc.tool_name}({'✓' if tc.result_success else '✗'})"
                )
        tools_str = " → ".join(tools_used) if tools_used else "无工具调用"

        flags = []
        if self.correction_triggered:
            flags.append("纠正")
        if self.fallback_triggered:
            flags.append("降级")
        flags_str = f" [{','.join(flags)}]" if flags else ""

        return (
            f"[{self.total_turns}轮 | {self.total_duration_ms}ms] "
            f"{tools_str}{flags_str}"
        )
