"""
edges.py — v4.0 条件边（规则：边 = 读 state 做路由决策，不改 state）

与 v0.3 §2.3 一一对应：
  route_after_agent  = route_by_finish + need_correction 的复合
                       （草稿画成两个菱形是概念分层；LangGraph 一个节点
                        只挂一个条件边函数，所以复合成一个 dispatcher，
                        两段逻辑仍是独立函数、可单测）
  need_correction    = v3.0 分支 1 的"只在从未用过工具时才纠正"
  after_tools        = check_failures + 回 agent 的 turn_count 闸门
  gate_to_agent      = 每条回到 agent 的边共用的闸门（E4：自己主动收口，
                       框架 recursion_limit 只兜底）

v0.3 实现约束：读 per-query 标志一律 state.get(k, 默认)，
不依赖 init 一定先跑过（E2：TypedDict 无隐式默认值）。
"""

from config import MAX_TURNS, MAX_CONSECUTIVE_ERRORS
from checks import should_have_retrieved, should_have_searched
from state import AgentState


def route_after_agent(state: AgentState) -> str:
    """agent 之后：tool_calls → tools；stop → 纠正判定；answer 已写 → 短路 finalize。"""
    if state.get("answer"):  # LLM 调用失败短路通道（agent 节点写入）
        return "finalize"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):  # route_by_finish: "tool_calls"
        return "tools"
    return need_correction(state)          # route_by_finish: "stop"


def need_correction(state: AgentState) -> str:
    """
    复刻 v3.0 分支 1：
      - 任何工具被调用过 → 尊重模型的 stop，直接 update_memory
      - 否则按"检索优先、联网其次"判定，各自最多注入一次
      - 都不满足 → update_memory
    """
    if state.get("has_searched", False) or state.get("has_retrieved", False):
        return "update_memory"

    user_message = state.get("user_message", "")
    if (not state.get("retrieval_correction_injected", False)
            and should_have_retrieved(user_message)):
        return "inject_correction_retrieval"
    if (not state.get("search_correction_injected", False)
            and should_have_searched(user_message)):
        return "inject_correction_search"
    return "update_memory"


def after_tools(state: AgentState) -> str:
    """tools 之后：连续失败 ≥2 且未降级过 → inject_fallback；否则过闸门回 agent。"""
    if (state.get("consecutive_failures", 0) >= MAX_CONSECUTIVE_ERRORS
            and not state.get("fallback_injected", False)):
        return "inject_fallback"
    return gate_to_agent(state)


def gate_to_agent(state: AgentState) -> str:
    """所有回到 agent 的边共用的循环上限闸门：超限直接 finalize（v0.3 §2.3）。"""
    return "agent" if state.get("turn_count", 0) < MAX_TURNS else "finalize"
