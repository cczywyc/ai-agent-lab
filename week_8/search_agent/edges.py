"""
edges.py — v6.0 条件边（规则：边 = 读 state 做路由决策，不改 state）

v6.0 第八周外循环 dispatcher（设计草稿 v0.2 §七 状态图 / E1 / E3 / E4）：
  route_supervisor      supervisor 之后：读 active_worker（supervisor 节点已按阶段设好）派 worker——
                        researcher → step_init（进 researcher 引擎，触发 per-subtask 重置）/ writer / reviewer / finalize。
                        路由本身是**条件函数、不接 LLM**（E1：supervisor 一个条件函数即可按阶段派活）。
  route_after_critic    researcher 内层 critic 之后：accept → supervisor（收编推进）；
                        retry 且 retry_count<MAX_RETRY → retry_reset（重做该研究子任务）；
                        escalate 或 retry 达上限 → supervisor（skip-and-advance 通道）。物理两出口 {supervisor, retry_reset}（E1）。
  route_after_reviewer  reviewer 之后：accept → finalize（早退）；reject 且 review_count<MAX_REVIEW → writer（返修）；
                        reject 且 review_count≥MAX_REVIEW → finalize（闸门收口走 best-so-far）。物理两出口 {writer, finalize}。
                        **读 review_count**（与 reviewer 节点"reject 时 review_count+=1"写同一个 key——E4 写键≡读键铁律）。

v5.0 内层边原样保留（researcher 引擎是黑盒，入口 assemble(researcher)、出口 critic）：
  route_after_agent / need_correction / after_tools / gate_to_agent

实现约束：读字段一律 state.get(k, 默认)（E2：TypedDict 无隐式默认值）。
need_correction 的"该不该检索/搜索"按**当前研究子任务 query**判（researcher 工作在子任务粒度）。
"""

from config import (
    MAX_TURNS, MAX_CONSECUTIVE_ERRORS, MAX_RETRY, MAX_REVIEW,
    SYNTHESIS_RESERVE_TURNS,
)
from checks import should_have_retrieved, should_have_searched
from state import AgentState


def _current_query(state: AgentState) -> str:
    """当前研究子任务 query（need_correction 据此判该不该检索/搜索）。"""
    plan = state.get("plan", [])
    k = state.get("step_index", 0)
    if 0 <= k < len(plan):
        return plan[k].get("query", "")
    return state.get("user_message", "")


# ============================================================
# 外循环 dispatcher（v6.0 supervisor 多 Agent）
# ============================================================

def route_supervisor(state: AgentState) -> str:
    """supervisor 之后：读 active_worker（supervisor 节点已按阶段设好）派 worker。
    active_worker=researcher → step_init（researcher 引擎入口，先做 per-subtask 重置）；writer/reviewer 直达；
    其余（含 finalize / 空）→ finalize。路由是纯条件函数、不接 LLM（E1）。"""
    aw = state.get("active_worker", "")
    if aw == "researcher":
        return "step_init"          # 进 researcher 引擎：step_init → assemble → agent ↔ tools ↔ inject → critic
    if aw in ("writer", "reviewer"):
        return aw
    return "finalize"


def route_after_critic(state: AgentState) -> str:
    """researcher 内层 critic 之后：三路判定塌缩成两条物理边 {supervisor, retry_reset}（E1）。
    accept/escalate → supervisor（收编推进 / skip-and-advance）；retry 且未达上限 → retry_reset（重做该研究子任务）。"""
    verdict = state.get("critic_verdict", "accept")
    if verdict == "accept":
        return "supervisor"                               # 去收编 finding / 推进
    if verdict == "retry" and state.get("retry_count", 0) < MAX_RETRY:
        return "retry_reset"                              # 轻量重置后回 researcher 重做该子任务（全新 turn 预算）
    return "supervisor"                                   # escalate / retry 达上限 → supervisor skip-and-advance 通道


def route_after_reviewer(state: AgentState) -> str:
    """reviewer 之后：accept 早退；reject 且未达上限 → writer 返修；reject 且达上限 → finalize（best-so-far 收口）。
    **读 review_count**——与 reviewer 节点 reject 时 review_count+=1 写**同一个 key**（E4 写键≡读键，否则死锁撞 recursion）。
    review_count（writer↔reviewer 级）独立于 replan_count（supervisor 级 skip），各兜各的正交维度（E4）。"""
    verdict = state.get("review_verdict", "")
    if verdict == "accept":
        return "finalize"                                 # 早退（主力：reviewer 一 accept 立即出环，正常根本到不了上限）
    if state.get("review_count", 0) >= MAX_REVIEW:
        return "finalize"                                 # 闸门收口（断路器）→ finalize 走 best-so-far；recursion 只兜底
    return "writer"                                       # 返修：带原稿 + review_notes 重写下一稿


# ============================================================
# 内层边（v5.0 researcher 引擎原样；出口落 critic = researcher 内层自检）
# ============================================================

def route_after_agent(state: AgentState) -> str:
    """agent 之后：tool_calls → tools；stop → 纠正判定（最终落 critic）。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return need_correction(state)


def need_correction(state: AgentState) -> str:
    """复刻 v3.0 分支 1（v5.0：被接受的 stop 去 critic 而非 finalize）：
      - 本研究子任务已用过工具 → 尊重 stop，交 critic 审
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
    """tools 之后：连续失败 ≥2 且未降级过 → inject_fallback；临近 turn 上限仍在检索 →
    inject_synthesis（逼综合，最多一次，v0.5）；否则过闸门回 agent。"""
    if (state.get("consecutive_failures", 0) >= MAX_CONSECUTIVE_ERRORS
            and not state.get("fallback_injected", False)):
        return "inject_fallback"
    if (state.get("turn_count", 0) >= MAX_TURNS - SYNTHESIS_RESERVE_TURNS
            and not state.get("synthesis_forced", False)):
        return "inject_synthesis"
    return gate_to_agent(state)


def gate_to_agent(state: AgentState) -> str:
    """回 agent 的内层闸门：单研究子任务 tool-use 上限（v5.0：超限交 critic 审而非 finalize）。"""
    return "agent" if state.get("turn_count", 0) < MAX_TURNS else "critic"
