"""
第九周 v7.0 · researcher 内层最小循环（决策 G：tools 环换接 MCP client，闸门不动）。

这是周三 mcp_stub.run_researcher_loop 的真实化：同一套 agent↔tools↔inject 闸门
（turn_count / empty_retries / synthesis-reserve），唯一变化是 `tools` 这一环从
v6.0 的 execute_tool(dict 派发) 换成 await station.call()（经真实 MCP client + stdio）。

为隔离验"接线机制"，驱动用确定性脚本调用序列（不接 LLM，零 API 那一半，承周三纪律）。
端到端接真实模型驱动是周五真实跑的事。闸门常量与周三 stub 对齐，E6 无回归才可比。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mcp_station import ESCALATE, INNER_RETRY, SUCCESS, MCPToolsStation

MAX_TURNS = 5
MAX_EMPTY_RETRIES = 2
SYNTHESIS_RESERVE_AT = MAX_TURNS - 1   # 最后一轮预留给综合（synthesis-reserve）


@dataclass
class LoopTrace:
    turn_count: int = 0
    empty_retries: int = 0
    synthesis_forced: bool = False
    injected: list = field(default_factory=list)
    last_path: str = ""
    escalated: bool = False


async def run_researcher_loop(scripted_calls, station: MCPToolsStation) -> LoopTrace:
    """跑确定性脚本调用序列，闸门全在 loop 里、与 station 实现无关（决策 G 的设计主张）。
    scripted_calls: [(tool_name, args), ...]。"""
    tr = LoopTrace()
    calls = list(scripted_calls)
    i = 0
    while tr.turn_count < MAX_TURNS:
        # synthesis-reserve：到预留轮且已有结果 → 逼综合早退
        if tr.turn_count >= SYNTHESIS_RESERVE_AT and tr.injected:
            tr.synthesis_forced = True
            break
        if i >= len(calls):
            break
        tr.turn_count += 1
        name, args = calls[i]
        routing = await station.call(name, args)
        tr.last_path = routing.path
        if routing.path == SUCCESS:
            tr.injected.append(routing.payload)
            i += 1
        elif routing.path == INNER_RETRY:
            tr.empty_retries += 1
            if tr.empty_retries > MAX_EMPTY_RETRIES:   # 重试跑满 → 轻量重置、推进
                tr.empty_retries = 0
                i += 1
        elif routing.path == ESCALATE:                 # 协议 error/守卫拒：退出内层、上报
            tr.escalated = True
            break
    return tr
