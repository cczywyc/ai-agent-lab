"""
E3 — retrieved_chunks 轮内累加 + 跨问题清空（决策 B）
验证：
  (a) tools 节点手动"当前+新"累加(替换语义),一个 invoke 内多次检索能累起来;
  (b) init 返回 [] 能干净清空(替换字段);
对照:同时放一个 bad_chunks: Annotated[list, add](operator.add reducer),
  init 也对它返回 [],证明它【清不掉】—— 这就是草稿里"为何不给 chunks 上 operator.add"的实证。
对应草稿:设计问题 ① / 周三验证清单 item 3。
"""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 记录"第二个 invoke 刚进 agent 时(init 之后)"两个字段的值
snapshot = {}


class State(TypedDict):
    retrieved_chunks: list                 # 替换字段:累加靠 tools 节点手动做
    bad_chunks: Annotated[list, add]       # 对照:operator.add reducer 自动累加
    loop_count: int


def init(state: State):
    # 清空两个字段 + 计数归零
    return {"retrieved_chunks": [], "bad_chunks": [], "loop_count": 0}


def agent(state: State):
    # 第二个 invoke 首次进 agent 时(loop_count==0)拍快照,看 init 后两字段状态
    if state["loop_count"] == 0 and "invoke2" not in snapshot:
        if snapshot.get("_seen_first_invoke"):
            snapshot["invoke2"] = (list(state["retrieved_chunks"]), list(state["bad_chunks"]))
        else:
            snapshot["_seen_first_invoke"] = True
    return {}  # agent 桩不改状态,路由交给条件边


def route(state: State):
    return "tools" if state["loop_count"] < 2 else "finalize"


def tools(state: State):
    n = state["loop_count"]
    new = [f"chunk_{n}a", f"chunk_{n}b"]
    return {
        "retrieved_chunks": state["retrieved_chunks"] + new,  # 手动累加(替换语义)
        "bad_chunks": new,                                    # operator.add 自动累加
        "loop_count": n + 1,
    }


def finalize(state: State):
    return {}


def build():
    g = StateGraph(State)
    g.add_node("init", init)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("finalize", finalize)
    g.add_edge(START, "init")
    g.add_edge("init", "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())


def main():
    print("=== E3: retrieved_chunks 轮内累加 + 跨问题清空 ===\n")
    graph = build()
    cfg = {"configurable": {"thread_id": "t1"}}

    f1 = graph.invoke({"retrieved_chunks": [], "bad_chunks": [], "loop_count": 0}, cfg)
    print("[Invoke 1 — 一个问题内检索 2 次]")
    print(f"  retrieved_chunks: {f1['retrieved_chunks']}")
    print(f"  bad_chunks      : {f1['bad_chunks']}")

    f2 = graph.invoke({"retrieved_chunks": [], "bad_chunks": [], "loop_count": 0}, cfg)
    inv2_rc, inv2_bad = snapshot["invoke2"]
    print("\n[Invoke 2 — init 之后、再次检索之前的快照]")
    print(f"  retrieved_chunks: {inv2_rc}   (期望 []  ← 替换字段被 init 清空)")
    print(f"  bad_chunks      : {inv2_bad}   (期望 非空 ← operator.add 字段 init 清不掉)")

    ok_accum = f1["retrieved_chunks"] == ["chunk_0a", "chunk_0b", "chunk_1a", "chunk_1b"]
    ok_clear = inv2_rc == []
    ok_badnoclear = len(inv2_bad) == 4   # 上一个 invoke 累的 4 条还在,没被 [] 清掉

    print("\n[通过判据]")
    print(f"  轮内累加 (单 invoke 内累成 4 条)        : {'✓ PASS' if ok_accum else '✗ FAIL'}")
    print(f"  跨问题清空 (替换字段 init 后为 [])      : {'✓ PASS' if ok_clear else '✗ FAIL'}")
    print(f"  对照:operator.add 字段 init 清不掉      : {'✓ PASS' if ok_badnoclear else '✗ FAIL'}")
    print("    └ 实证:带累加 reducer 的字段,return [] 是\"追加空列表\"(no-op),无法重置;")
    print("      所以 chunks 选\"替换语义 + 节点内手动累加\",init 才能用 [] 干净清空。")
    allok = ok_accum and ok_clear and ok_badnoclear
    print(f"\nE3 {'全部通过 ✓ —— 决策 B 坐实' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
