"""
edges.py — v5.0 条件边（规则：边 = 读 state 做路由决策，不改 state）

v5.0 第七周新增两个外循环 dispatcher（设计草稿 v0.3 §2.3 / E1 / E4）：
  route_after_planner  planner 之后：done / step_index≥MAX_STEPS / replan_count≥MAX_REPLAN → finalize；
                       否则 → step_init（开始下一子任务，触发 §一 per-subtask 重置）
  route_after_critic   critic 之后：accept → planner；retry 且 retry_count<MAX_RETRY → assemble（重做该步）；
                       escalate 或 retry 达上限 → planner（escalate 通道）。
                       判定三路、**物理两出口** {planner, assemble}（E1：别写成三出口）。

v4.2 内层边原样保留，仅把**出口从 finalize 改接 critic**（executor 引擎是黑盒，
入口 assemble(executor)、出口 critic——单步做完/做不下去都交 critic 审）：
  route_after_agent / need_correction / after_tools / gate_to_agent

v0.3 实现约束：读字段一律 state.get(k, 默认)（E2：TypedDict 无隐式默认值）。
need_correction 的"该不该检索/搜索"按**当前子任务 query**判（executor 工作在子任务粒度）。
"""

from config import MAX_TURNS, MAX_CONSECUTIVE_ERRORS, MAX_STEPS, MAX_REPLAN, MAX_RETRY
from checks import should_have_retrieved, should_have_searched
from state import AgentState


def _current_query(state: AgentState) -> str:
    """当前子任务 query（need_correction 据此判该不该检索/搜索）。"""
    plan = state.get("plan", [])
    k = state.get("step_index", 0)
    if 0 <= k < len(plan):
        return plan[k].get("query", "")
    return state.get("user_message", "")


# ============================================================
# 外循环 dispatcher（v5.0 新增）
# ============================================================

def route_after_planner(state: AgentState) -> str:
    """planner 之后：双闸门主动收口（E4），否则进下一子任务（先 step_init 重置）。"""
    if state.get("done", False) or state.get("step_index", 0) >= MAX_STEPS:
        return "finalize"
    if state.get("replan_count", 0) >= MAX_REPLAN:   # 绕圈兜底（E4：escalate 恒压下它先收口）
        return "finalize"
    if state.get("step_index", 0) >= len(state.get("plan", [])):  # 子任务走完
        return "finalize"
    return "step_init"


def route_after_critic(state: AgentState) -> str:
    """critic 之后：三路判定塌缩成两条物理边 {planner, assemble}（E1）。"""
    verdict = state.get("critic_verdict", "accept")
    if verdict == "accept":
        return "planner"                                  # 去判 done / 推进下一步
    if verdict == "retry" and state.get("retry_count", 0) < MAX_RETRY:
        return "assemble"                                 # 回 executor 重做该步（不经 step_init）
    return "planner"                                      # escalate / retry 达上限 → escalate 通道


# ============================================================
# 内层边（v4.2 原样，出口由 finalize 改接 critic）
# ============================================================

def route_after_agent(state: AgentState) -> str:
    """agent 之后：tool_calls → tools；stop → 纠正判定（最终落 critic）。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return need_correction(state)


def need_correction(state: AgentState) -> str:
    """复刻 v3.0 分支 1（v5.0：被接受的 stop 去 critic 而非 finalize）：
      - 本子任务已用过工具 → 尊重 stop，交 critic 审
      - 否则按"检索优先、联网其次"判定（按当前子任务 query），各自最多注入一次
      - 都不满足 → critic
    """
    if state.get("has_searched", False) or state.get("has_retrieved", False):
        return "critic"
    query = _current_query(state)
    if (not state.get("retrieval_correction_injected", False)
            and should_have_retrieved(query)):
        return "inject_correction_retrieval"
    if (not state.get("search_correction_injected", False)
            and should_have_searched(query)):
        return "inject_correction_search"
    return "critic"


def after_tools(state: AgentState) -> str:
    """tools 之后：连续失败 ≥2 且未降级过 → inject_fallback；否则过闸门回 agent。"""
    if (state.get("consecutive_failures", 0) >= MAX_CONSECUTIVE_ERRORS
            and not state.get("fallback_injected", False)):
        return "inject_fallback"
    return gate_to_agent(state)


def gate_to_agent(state: AgentState) -> str:
    """回 agent 的内层闸门：单子任务 tool-use 上限（v5.0：超限交 critic 审而非 finalize）。"""
    return "agent" if state.get("turn_count", 0) < MAX_TURNS else "critic"
