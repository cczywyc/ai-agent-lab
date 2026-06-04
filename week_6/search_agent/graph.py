"""
graph.py — v4.0 状态图组装（v0.3 §五 状态图原样接线）

    START → init → assemble → agent ─┬─ tool_calls → tools ─┬─ 失败≥2 → inject_fallback ──(闸门)→ agent/finalize
                                     │                      └─(闸门)→ agent / finalize
                                     ├─ stop·需检索纠正 → inject_correction_retrieval ──(闸门)→ agent/finalize
                                     ├─ stop·需联网纠正 → inject_correction_search ──(闸门)→ agent/finalize
                                     └─ stop·无需纠正 → update_memory → human_review → finalize → END

checkpointer 包住整张图（compile(checkpointer=...)），按 thread_id 归档快照。
本周选 InMemorySaver（决策 A：跨会话持久化仍由 memory/ 的 json 承担，
图状态进程内存活；Store/SqliteSaver 迁移是后续单独一步）。
"""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import nodes
from edges import after_tools, gate_to_agent, route_after_agent
from state import AgentState


def build_graph(checkpointer=None):
    """组装并编译 v4.0 状态图。不传 checkpointer 时用独立的 InMemorySaver。"""
    g = StateGraph(AgentState)

    # ===== 节点（干活/改 state）=====
    g.add_node("init", nodes.init)
    g.add_node("assemble", nodes.assemble)
    # 经由模块属性引用 agent/tools，保证测试 monkeypatch nodes.call_model/execute_tool 生效
    g.add_node("agent", nodes.agent)
    g.add_node("tools", nodes.tools)
    # 决策 D：两条可见路径（检索优先、联网其次记在图里），共用同一函数体
    g.add_node("inject_correction_retrieval", partial(nodes.inject_correction, kind="retrieval"))
    g.add_node("inject_correction_search", partial(nodes.inject_correction, kind="search"))
    g.add_node("inject_fallback", nodes.inject_fallback)
    g.add_node("update_memory", nodes.update_memory)
    g.add_node("human_review", nodes.human_review)
    g.add_node("finalize", nodes.finalize)

    # ===== 顺序边 =====
    g.add_edge(START, "init")
    g.add_edge("init", "assemble")
    g.add_edge("assemble", "agent")
    g.add_edge("update_memory", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)

    # ===== 条件边（读 state 做决策）=====
    g.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "inject_correction_retrieval": "inject_correction_retrieval",
        "inject_correction_search": "inject_correction_search",
        "update_memory": "update_memory",
        "finalize": "finalize",  # LLM 错误短路
    })
    g.add_conditional_edges("tools", after_tools, {
        "inject_fallback": "inject_fallback",
        "agent": "agent",
        "finalize": "finalize",  # turn_count 闸门
    })
    # 所有回到 agent 的边都过 turn_count 闸门（E4：主动收口）
    for cycle_node in ("inject_correction_retrieval", "inject_correction_search", "inject_fallback"):
        g.add_conditional_edges(cycle_node, gate_to_agent, {
            "agent": "agent",
            "finalize": "finalize",
        })

    return g.compile(checkpointer=checkpointer or InMemorySaver())
