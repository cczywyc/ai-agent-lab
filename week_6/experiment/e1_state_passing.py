"""
E1 — 状态在节点间传递 + add_messages 累加
验证：State schema 用 add_messages 单 reducer + 其余字段默认替换，
      状态能在 assemble -> agent -> finalize 三个桩节点间正常流动。
对应草稿：设计问题 ① / 周三验证清单 item 1。
桩节点：不调模型、不碰 RAG，只返回固定 dict 更新。
"""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]  # 累加字段
    marker: str                              # 替换字段


def assemble(state: State):
    # A 节点写 marker，并追加一条消息
    return {"marker": "set_by_assemble",
            "messages": [{"role": "user", "content": "msg from assemble"}]}


def agent(state: State):
    # B 节点再追加一条消息（验证累加，不是覆盖）
    return {"messages": [{"role": "assistant", "content": "msg from agent"}]}


def finalize(state: State):
    # C 节点读 A 写的 marker（验证替换字段跨节点可见）
    seen = state["marker"]
    return {"messages": [{"role": "assistant", "content": f"finalize saw marker={seen}"}]}


def build():
    g = StateGraph(State)
    g.add_node("assemble", assemble)
    g.add_node("agent", agent)
    g.add_node("finalize", finalize)
    g.add_edge(START, "assemble")
    g.add_edge("assemble", "agent")
    g.add_edge("agent", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def main():
    graph = build()
    final = graph.invoke({"messages": [], "marker": ""})

    print("=== E1: 状态传递 + add_messages 累加 ===")
    print(f"最终 marker      : {final['marker']!r}")
    print(f"最终 messages 条数: {len(final['messages'])}")
    for m in final["messages"]:
        # 消息可能被转成 langchain 消息对象，统一取 content
        content = m.content if hasattr(m, "content") else m["content"]
        print(f"   - {content}")

    ok_marker = final["marker"] == "set_by_assemble"          # C 看得到 A 写的字段
    ok_accum = len(final["messages"]) == 3                     # 三节点各追加一条 = 累加
    print("\n[通过判据]")
    print(f"  替换字段跨节点可见 (C 读到 A 的 marker): {'✓ PASS' if ok_marker else '✗ FAIL'}")
    print(f"  messages 累加而非覆盖 (3 条)          : {'✓ PASS' if ok_accum else '✗ FAIL'}")
    print(f"\nE1 {'全部通过 ✓' if (ok_marker and ok_accum) else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
