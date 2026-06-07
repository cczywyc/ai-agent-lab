"""
graph.py — v4.2 状态图组装（v0.3 §五 接线 + 决策 E：store 进 compile + 收尾时序反转）

    START → init → assemble → agent ─┬─ tool_calls → tools ─┬─ 失败≥2 → inject_fallback ──(闸门)→ agent/finalize
                                     │                      └─(闸门)→ agent / finalize
                                     ├─ stop·需检索纠正 → inject_correction_retrieval ──(闸门)→ agent/finalize
                                     ├─ stop·需联网纠正 → inject_correction_search ──(闸门)→ agent/finalize
                                     └─ stop·无需纠正 → finalize
    所有路径统一收口：finalize → human_review → update_memory → END

收尾时序反转（v4.2，修 06-05 复跑 quirk 2 的两个倒挂）：
  先 finalize 组装终稿（含占位符）→ 审批看到的永远是真实终稿、可当场补救；
  再 human_review（可改写）→ 最后 update_memory 记录用户实际看到的版本。
  旧图错误/超轮路径绕过 update_memory 的行为，由其占位符跳过逻辑等价承接。

checkpointer 包住整张图（compile(checkpointer=...)），按 thread_id 归档快照——
管 thread 内短期状态。store 管跨 thread 长期记忆（compile(store=...)，
LangGraph 注入给 assemble / update_memory 节点）——这是 checkpointer / store 的边界。
checkpointer 本周仍选 InMemorySaver（决策 A）。
"""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import nodes
from edges import after_tools, gate_to_agent, route_after_agent
from memory.ltm_store import get_ltm_store
from state import AgentState


def build_graph(checkpointer=None, store=None):
    """
    组装并编译 v4.1 状态图。
    不传 checkpointer 时用独立的 InMemorySaver；
    不传 store 时用 get_ltm_store() 进程单例（测试可注入 stub embed 的 InMemoryStore）。
    """
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
    # 收尾三连（v4.2 反转）：终稿 → 审批 → 记忆
    g.add_edge("finalize", "human_review")
    g.add_edge("human_review", "update_memory")
    g.add_edge("update_memory", END)

    # ===== 条件边（读 state 做决策）=====
    g.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "inject_correction_retrieval": "inject_correction_retrieval",
        "inject_correction_search": "inject_correction_search",
        "finalize": "finalize",  # stop·无需纠正 与 LLM 错误短路同入口
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

    return g.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or get_ltm_store(),
    )
