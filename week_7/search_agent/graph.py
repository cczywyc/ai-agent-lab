"""
graph.py — v5.0 状态图组装（设计草稿 v0.3 §五 接线 · planner-executor-critic 外循环）

    START → init → planner ─┬─(done / 超 MAX_STEPS / 超 MAX_REPLAN)──────────────► finalize
                            │
                            └─(下一子任务)─► step_init ─► assemble(role=executor) ─► agent ─┐
                                  ▲  ▲                                                       │
       retry（critic→assemble，  │  │ executor 引擎（v4.2 内层循环原样，出口改接 critic）   │
        不经 step_init）─────────┘  │   agent ┬ tool_calls ► tools ┬ 失败≥2 ► inject_fallback ┘(闸门)
                                    │         └ stop ► 纠正判定/critic └(闸门)► agent / critic
                            ┌───────┘
                       critic ─┬─ accept ──────────────► planner（判 done / 推进）
                               ├─ retry(<上限) ────────► assemble（重做该步）
                               └─ escalate / retry 达上限 ► planner（escalate 通道：re-plan/跳过）
    收口统一：finalize（组装结构化报告）→ human_review（审批）→ update_memory → END

外循环双闸门（E4，各兜一正交维度、非冗余）：step_index<MAX_STEPS（前进步数）+
replan_count<MAX_REPLAN（原地绕圈）。内层 turn_count<MAX_TURNS（单子任务 tool-use）原样。
框架 recursion_limit 只兜底（E6：默认 10007 远超有界任务，收口靠显式闸门）。

收尾时序反转（v4.2 起）：finalize 先组装终稿 → human_review 看到的永远是真实报告、可当场补救
→ update_memory 记录用户实际看到的版本。checkpointer 管 thread 内短期、store 管跨 thread 长期。
"""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import nodes
from edges import (
    after_tools,
    gate_to_agent,
    route_after_agent,
    route_after_planner,
    route_after_critic,
)
from memory.ltm_store import get_ltm_store
from state import AgentState


def build_graph(checkpointer=None, store=None):
    """
    组装并编译 v5.0 状态图。
    不传 checkpointer 时用独立的 InMemorySaver；
    不传 store 时用 get_ltm_store() 进程单例（测试可注入 stub embed 的 InMemoryStore）。
    """
    g = StateGraph(AgentState)

    # ===== 节点（干活/改 state）=====
    g.add_node("init", nodes.init)
    g.add_node("planner", nodes.planner)              # v5.0 新增：拆解 / 推进 / re-plan
    g.add_node("step_init", nodes.step_init)          # v5.0 新增：新子任务 per-subtask 全量重置（设计 ①）
    g.add_node("retry_reset", nodes.retry_reset)      # v5.0 新增：业务 retry 轻量重置（与 step_init 对称）
    # 经由模块属性引用，保证测试 monkeypatch nodes.call_*_model / execute_tool 生效
    g.add_node("assemble", partial(nodes.assemble, role="executor"))  # 决策 C：按 role 参数化
    g.add_node("agent", nodes.agent)
    g.add_node("tools", nodes.tools)
    g.add_node("inject_correction_retrieval", partial(nodes.inject_correction, kind="retrieval"))
    g.add_node("inject_correction_search", partial(nodes.inject_correction, kind="search"))
    g.add_node("inject_fallback", nodes.inject_fallback)
    g.add_node("critic", nodes.critic)                # v5.0 新增：审单步 → verdict
    g.add_node("finalize", nodes.finalize)
    g.add_node("human_review", nodes.human_review)
    g.add_node("update_memory", nodes.update_memory)

    # ===== 顺序边 =====
    g.add_edge(START, "init")
    g.add_edge("init", "planner")
    g.add_edge("step_init", "assemble")               # 新子任务全量重置后开始装配
    g.add_edge("retry_reset", "assemble")             # 业务 retry 轻量重置后重做该步
    g.add_edge("assemble", "agent")
    # 收尾三连（v4.2 反转）：终稿报告 → 审批 → 记忆
    g.add_edge("finalize", "human_review")
    g.add_edge("human_review", "update_memory")
    g.add_edge("update_memory", END)

    # ===== 条件边（读 state 做决策）=====
    # planner 后：双闸门收口 / 进下一子任务（E4）
    g.add_conditional_edges("planner", route_after_planner, {
        "step_init": "step_init",
        "finalize": "finalize",
    })
    # agent 后：executor 引擎内层分流（出口落 critic）
    g.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "inject_correction_retrieval": "inject_correction_retrieval",
        "inject_correction_search": "inject_correction_search",
        "critic": "critic",                           # stop·无需纠正 → 交 critic 审单步
    })
    g.add_conditional_edges("tools", after_tools, {
        "inject_fallback": "inject_fallback",
        "agent": "agent",
        "critic": "critic",                           # turn_count 闸门超限 → critic
    })
    # 所有回到 agent 的内层边都过 turn_count 闸门（超限交 critic）
    for cycle_node in ("inject_correction_retrieval", "inject_correction_search", "inject_fallback"):
        g.add_conditional_edges(cycle_node, gate_to_agent, {
            "agent": "agent",
            "critic": "critic",
        })
    # critic 后：三路判定塌缩成两条物理边 {planner, retry_reset}（E1）
    g.add_conditional_edges("critic", route_after_critic, {
        "planner": "planner",
        "retry_reset": "retry_reset",                 # retry：轻量重置后回 executor 重做该步
    })

    return g.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or get_ltm_store(),
    )
