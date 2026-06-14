"""
graph.py — v6.0 状态图组装（设计草稿 v0.2 §七 状态图 · supervisor 多 Agent）

    START → init → supervisor ─route_supervisor─┬─(researcher)─► step_init ─► assemble(researcher) ─► agent ─┐
                      ▲   ▲   ▲                  │                                                            │
       (accept 收编/  │   │   │ (writer 写完)     │  researcher 引擎（v5.0 内层循环原样，出口落 critic）       │
        skip-advance) │   │   │                  │   agent ┬ tool_calls ► tools ┬ 失败≥2 ► inject_fallback ──┘(闸门)
                      │   │   └──────────────────┤         └ stop ► 纠正判定/critic └(闸门)► agent / critic
   critic ─route_after_critic─┬─ supervisor      │
    (researcher 内层自检)      └─ retry_reset ►assemble (retry 重做该研究子任务)
                              ┌───────────────────┘
                       (writer)─► writer ─► supervisor
                       (reviewer)─► reviewer ─route_after_reviewer─┬─ writer (reject & review_count<MAX_REVIEW，返修)
                                                                   └─ finalize (accept 早退 / reject & ≥MAX_REVIEW best-so-far)
    收口统一：finalize（交付被评审的 draft）→ human_review（审批）→ update_memory → END

升格不重写（决策 G）：planner→supervisor、executor 引擎→researcher（step_init/assemble/agent/tools/inject_*/critic
原样复用）、critic→reviewer。真正新增仅 writer 节点、工具式 handoff（task_description 四要素）、writer↔reviewer 打回循环。

四档闸门正交（E4/决策 H）：researcher 内层 turn_count<MAX_TURNS（单子任务 tool-use）/ retry_count<MAX_RETRY（业务重做）；
supervisor 级 replan_count<MAX_REPLAN（研究子任务 skip）；writer↔reviewer 级 review_count<MAX_REVIEW（打回）。
框架 recursion_limit 只兜底（E7：本周桩 super-step≈10 ≪ 默认 10007，收口靠显式闸门、非抛异常）。
"""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import nodes
from edges import (
    after_tools,
    gate_to_agent,
    route_after_agent,
    route_supervisor,
    route_after_critic,
    route_after_reviewer,
)
from memory.ltm_store import get_ltm_store
from state import AgentState


def build_graph(checkpointer=None, store=None):
    """
    组装并编译 v6.0 supervisor 多 Agent 状态图。
    不传 checkpointer 时用独立的 InMemorySaver；
    不传 store 时用 get_ltm_store() 进程单例（测试可注入 stub embed 的 InMemoryStore）。
    """
    g = StateGraph(AgentState)

    # ===== 节点（干活/改 state）=====
    g.add_node("init", nodes.init)
    g.add_node("supervisor", nodes.supervisor)          # v6.0：planner 升格——拆研究子任务 / 收编推进 / 按阶段派活
    # --- researcher 引擎（v5.0 executor 内层循环原样复用；经模块属性引用以便测试 monkeypatch）---
    g.add_node("step_init", nodes.step_init)            # 新研究子任务 per-subtask 全量重置
    g.add_node("retry_reset", nodes.retry_reset)        # 业务 retry 轻量重置
    g.add_node("assemble", partial(nodes.assemble, role="researcher"))
    g.add_node("agent", nodes.agent)
    g.add_node("tools", nodes.tools)
    g.add_node("inject_correction_retrieval", partial(nodes.inject_correction, kind="retrieval"))
    g.add_node("inject_correction_search", partial(nodes.inject_correction, kind="search"))
    g.add_node("inject_fallback", nodes.inject_fallback)
    g.add_node("inject_synthesis", nodes.inject_synthesis)
    g.add_node("critic", nodes.critic)                  # researcher 内层单步自检 → verdict
    # --- v6.0 新增 worker ---
    g.add_node("writer", nodes.writer)                  # 全新：findings → 初稿
    g.add_node("reviewer", nodes.reviewer)              # critic 升格：审 draft → verdict+notes+score
    # --- 收尾链尾（v4.2 反转时序原样）---
    g.add_node("finalize", nodes.finalize)              # 改：交付被评审的 draft（best_draft/accepted）
    g.add_node("human_review", nodes.human_review)
    g.add_node("update_memory", nodes.update_memory)

    # ===== 顺序边 =====
    g.add_edge(START, "init")
    g.add_edge("init", "supervisor")
    g.add_edge("step_init", "assemble")                 # 新研究子任务全量重置后开始装配
    g.add_edge("retry_reset", "assemble")               # 业务 retry 轻量重置后重做该研究子任务
    g.add_edge("assemble", "agent")
    g.add_edge("writer", "supervisor")                  # writer 写完回 supervisor（它会路由到 reviewer 待审）
    # 收尾三连（v4.2 反转）：交付稿 → 审批 → 记忆
    g.add_edge("finalize", "human_review")
    g.add_edge("human_review", "update_memory")
    g.add_edge("update_memory", END)

    # ===== 条件边（读 state 做决策）=====
    # supervisor 后：读 active_worker 派 worker（researcher 经 step_init 进引擎）
    g.add_conditional_edges("supervisor", route_supervisor, {
        "step_init": "step_init",
        "writer": "writer",
        "reviewer": "reviewer",
        "finalize": "finalize",
    })
    # researcher 引擎内层分流（v5.0 原样，出口落 critic）
    g.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "inject_correction_retrieval": "inject_correction_retrieval",
        "inject_correction_search": "inject_correction_search",
        "critic": "critic",
    })
    g.add_conditional_edges("tools", after_tools, {
        "inject_fallback": "inject_fallback",
        "inject_synthesis": "inject_synthesis",
        "agent": "agent",
        "critic": "critic",
    })
    for cycle_node in ("inject_correction_retrieval", "inject_correction_search",
                       "inject_fallback", "inject_synthesis"):
        g.add_conditional_edges(cycle_node, gate_to_agent, {
            "agent": "agent",
            "critic": "critic",
        })
    # researcher 内层 critic 后：三路判定塌缩成两条物理边 {supervisor, retry_reset}（E1）
    g.add_conditional_edges("critic", route_after_critic, {
        "supervisor": "supervisor",                     # accept 收编 / escalate skip-and-advance（v5.0 是 planner）
        "retry_reset": "retry_reset",                   # retry：轻量重置后回 researcher 重做该子任务
    })
    # reviewer 后：打回循环两条物理边 {writer, finalize}（E3：恒 reject 在 review_count 闸门收口走 best-so-far）
    g.add_conditional_edges("reviewer", route_after_reviewer, {
        "writer": "writer",                             # 返修
        "finalize": "finalize",                         # accept 早退 / 达上限 best-so-far 收口
    })

    return g.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or get_ltm_store(),
    )
