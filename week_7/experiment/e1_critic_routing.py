"""
E1 — critic 三路条件边分发（accept / retry / escalate）
验证：单个条件边函数 route_after_critic 按 critic_verdict 把三种裁决正确路由到
      planner / executor / planner 三个目标；且"retry 达上限"塌缩到 escalate 通道。
对应草稿：设计问题 ②（§2.3）/ 周三验证清单 item 1 / 决策 A、B。
桩节点：不调模型、不审质量，critic 桩按注入的 _test_verdict 直接写 verdict。
"""
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

MAX_RETRY = 2


class State(TypedDict):
    messages: Annotated[list, add_messages]
    critic_verdict: Literal["accept", "retry", "escalate"]
    retry_count: int
    landed: str            # 哪个 sink 跑到了（替换字段，验路由落点）
    _test_verdict: str     # 注入：本次想让 critic 发的裁决
    _test_retry: int       # 注入：进 critic 前的 retry_count


def critic(state: State):
    # 桩：直接把注入的裁决写进 state（真实 critic 会审单步输出）
    return {"critic_verdict": state["_test_verdict"],
            "retry_count": state.get("_test_retry", 0)}


def route_after_critic(state: State) -> str:
    """与 edges.py 同款写法：读 state 做路由，state.get 防御取值。"""
    verdict = state.get("critic_verdict", "accept")
    if verdict == "accept":
        return "to_planner"
    if verdict == "retry" and state.get("retry_count", 0) < MAX_RETRY:
        return "to_executor"            # 回 assemble(executor) 重做该步
    return "to_planner"                 # escalate，或 retry 达上限 → 都回 planner


def planner_sink(state: State):
    return {"landed": "planner"}


def executor_sink(state: State):
    return {"landed": "executor"}


def build():
    g = StateGraph(State)
    g.add_node("critic", critic)
    g.add_node("to_planner", planner_sink)
    g.add_node("to_executor", executor_sink)
    g.add_edge(START, "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"to_planner": "to_planner", "to_executor": "to_executor"})
    g.add_edge("to_planner", END)
    g.add_edge("to_executor", END)
    return g.compile()


def run(graph, verdict, retry):
    final = graph.invoke({"messages": [], "critic_verdict": "accept",
                          "retry_count": retry, "landed": "",
                          "_test_verdict": verdict, "_test_retry": retry})
    return final["landed"]


def main():
    print("=== E1: critic 三路分发 ===\n")
    graph = build()

    cases = [
        ("accept",   0, "planner"),    # accept → planner（去判 done）
        ("retry",    0, "executor"),   # retry 未达上限 → 回 executor 重做
        ("retry",    2, "planner"),    # retry 达上限 → 塌缩到 planner（escalate 通道）
        ("escalate", 0, "planner"),    # escalate → planner（换措辞/跳过/re-plan）
    ]
    results = []
    for verdict, retry, expect in cases:
        got = run(graph, verdict, retry)
        ok = got == expect
        results.append(ok)
        print(f"  verdict={verdict:<8} retry_count={retry} → {got:<8} "
              f"(期望 {expect}) {'✓' if ok else '✗'}")

    print("\n[通过判据]")
    print(f"  四种裁决全部路由到期望目标: {'✓ PASS' if all(results) else '✗ FAIL'}")
    print(f"\nE1 {'全部通过 ✓' if all(results) else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
