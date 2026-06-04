"""
E4 — 纠正 cycle × turn_count 闸门(item 4)
验证:回连边(inject_* -> agent)配 turn_count 闸门能正常终止,
     在 LangGraph 自带递归上限之前先收口(保留 v3.0 的显式语义)。
对照:把同一张图的 turn_count 闸门拆掉、单靠框架的 recursion_limit,
     看它抛 GraphRecursionError —— 证明"闸门是你这边主动收口的"。
对应草稿:设计问题 ② / 周三验证清单 item 4。
"""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

MAX_TURNS = 6


class State(TypedDict):
    turn_count: int
    log: Annotated[list, add]


def init(state: State):
    return {"turn_count": 0}


def agent(state: State):
    n = state["turn_count"] + 1
    return {"turn_count": n, "log": [f"turn {n}"]}


def route(state: State):
    # 闸门:未到上限就回 agent(模拟纠正/工具后的回连),否则收尾
    return "agent" if state["turn_count"] < MAX_TURNS else "finalize"


def finalize(state: State):
    return {}


def build(guarded: bool):
    g = StateGraph(State)
    g.add_node("init", init)
    g.add_node("agent", agent)
    g.add_node("finalize", finalize)
    g.add_edge(START, "init")
    g.add_edge("init", "agent")
    if guarded:
        g.add_conditional_edges("agent", route, {"agent": "agent", "finalize": "finalize"})
    else:
        g.add_edge("agent", "agent")  # 无闸门:死循环,只能靠框架 recursion_limit 兜底
    g.add_edge("finalize", END)
    return g.compile()


def main():
    print("=== E4: cycle × turn_count 闸门 ===\n")

    graph = build(guarded=True)
    final = graph.invoke({"turn_count": 0, "log": []})
    print("[有 turn_count 闸门]")
    print(f"  跑完轮次: {final['turn_count']}  (期望 {MAX_TURNS})")
    print(f"  log     : {final['log']}")
    ok_stop = final["turn_count"] == MAX_TURNS

    print("\n[无闸门 — 对照:单靠框架 recursion_limit]")
    g2 = build(guarded=False)
    raised = False
    try:
        # 给个较小的 recursion_limit 以便快速看到收口行为(默认 25)
        g2.invoke({"turn_count": 0, "log": []}, {"recursion_limit": 8})
    except GraphRecursionError as e:
        raised = True
        print(f"  抛出 GraphRecursionError(如期):{str(e)[:60]}...")

    print("\n[通过判据]")
    print(f"  闸门在框架上限前先收口,停在 turn {MAX_TURNS}: {'✓ PASS' if ok_stop else '✗ FAIL'}")
    print(f"  对照:无闸门确实撞 recursion_limit 抛错   : {'✓ PASS' if raised else '✗ FAIL'}")
    print("    └ 结论:turn_count 是你主动收口(保留 v3.0 MAX_TURNS 语义),")
    print("      框架 recursion_limit 只当最后兜底,不该靠它来正常终止。")
    allok = ok_stop and raised
    print(f"\nE4 {'全部通过 ✓' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
