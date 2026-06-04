"""
E2 — init 重置 × checkpointer 持久化（今天的核心）
验证（决策 C）：同一 thread_id 多次 invoke 时——
  (a) messages 历史跨轮累加（多轮记忆，靠 add_messages + checkpointer）；
  (b) per-query 标志被入口 init 节点打回初值，不会从上一问题串过来。
对照：再建一张没有 init 的图，证明"不重置就会泄漏"——这正是 init 节点存在的理由。
对应草稿：设计问题 ① / 周三验证清单 item 2 / 踩坑 #2 的图上验证。
"""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# agent 节点进入时"看到的 has_retrieved 值"记录在这里，用于证明重置是否生效
observed = {"with_init": [], "no_init": []}


class State(TypedDict):
    messages: Annotated[list, add_messages]  # 累加：跨轮持久化
    has_retrieved: bool                      # per-query 标志：每问题应清零
    retrieved_chunks: list                   # 替换字段


def init(state: State):
    # 入口重置本轮 per-query 标志；注意：不返回 messages（让它保持持久化）
    return {"has_retrieved": False, "retrieved_chunks": []}


def make_agent(tag: str):
    def agent(state: State):
        # 记录进入时看到的 has_retrieved —— 这是判断"是否被重置"的探针
        observed[tag].append(state.get("has_retrieved"))
        return {"messages": [{"role": "assistant", "content": "answer"}],
                "has_retrieved": True}
    return agent


def build(with_init: bool, tag: str):
    g = StateGraph(State)
    g.add_node("agent", make_agent(tag))
    if with_init:
        g.add_node("init", init)
        g.add_edge(START, "init")
        g.add_edge("init", "agent")
    else:
        g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile(checkpointer=InMemorySaver())


def run_two_turns(graph, thread_id):
    cfg = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"messages": [{"role": "user", "content": "Q1"}]}, cfg)
    final = graph.invoke({"messages": [{"role": "user", "content": "Q2"}]}, cfg)
    return final


def main():
    print("=== E2: init 重置 × checkpointer 持久化 ===\n")

    g_with = build(with_init=True, tag="with_init")
    final_with = run_two_turns(g_with, "t_with")

    g_no = build(with_init=False, tag="no_init")
    final_no = run_two_turns(g_no, "t_no")

    print("[有 init 的图]")
    print(f"  两轮后 messages 条数         : {len(final_with['messages'])}  (期望 4: Q1,A1,Q2,A2)")
    print(f"  agent 每轮进入时看到 has_retrieved: {observed['with_init']}  (期望 [False, False])")
    print("\n[无 init 的图 — 对照]")
    print(f"  两轮后 messages 条数         : {len(final_no['messages'])}  (期望 4: 持久化与 init 无关)")
    print(f"  agent 每轮进入时看到 has_retrieved: {observed['no_init']}  (期望 [None, True] ← 1轮无默认, 2轮泄漏)")

    ok_persist = len(final_with["messages"]) == 4                     # 历史跨轮累加
    ok_reset = observed["with_init"] == [False, False]               # init 把标志打回 False
    ok_leak_demo = observed["no_init"] == [None, True]               # 1轮:字段未初始化=None; 2轮:泄漏=True

    print("\n[通过判据]")
    print(f"  messages 跨轮持久化 (4 条)              : {'✓ PASS' if ok_persist else '✗ FAIL'}")
    print(f"  init 重置 per-query 标志 ([F,F])        : {'✓ PASS' if ok_reset else '✗ FAIL'}")
    print(f"  对照:无 init -> [None, True]            : {'✓ PASS' if ok_leak_demo else '✗ FAIL'}")
    print("    └ 意外发现:TypedDict 字段无隐式默认,第一轮是 None(裸取会 KeyError);")
    print("      所以 init 不止'跨轮重置',还负责'首轮建立默认值'——比草稿假设更必要。")
    print("\n  关键不对称:init 对 messages 不返回 = 保留(持久化);")
    print("            对替换字段返回 [] / 标志返回 False = 清空/重置。")
    allok = ok_persist and ok_reset and ok_leak_demo
    print(f"\nE2 {'全部通过 ✓ —— 决策 C 坐实,且证明 init 节点不可省' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
