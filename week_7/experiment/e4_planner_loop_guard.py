"""
E4 — 外循环双闸门：step_index < MAX_STEPS 且 replan_count < MAX_REPLAN
验证：planner → executor → critic → planner 外循环靠两层闸门主动收口，
      在框架 recursion_limit 之前先终止（E4 教训在外循环重演）。
对照：拆掉闸门单靠 recursion_limit，看它抛 GraphRecursionError。
对应草稿：设计问题 ②（§2.3 双层闸门）/ 周三验证清单 item 4 / 决策 E。
桩节点：planner/executor/critic 桩不调模型；critic 一直发 escalate 制造 re-plan 压力。
"""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

MAX_STEPS = 4
MAX_REPLAN = 2


class State(TypedDict):
    step_index: int
    replan_count: int
    log: Annotated[list, add]


def planner(state: State):
    # escalate 进来 = re-plan：replan_count +1（桩里恒被 critic escalate 推着 re-plan）
    n = state.get("replan_count", 0)
    came_from_escalate = state.get("step_index", 0) > 0
    rc = n + 1 if came_from_escalate else n
    return {"replan_count": rc, "log": [f"plan(replan={rc})"]}


def executor(state: State):
    i = state["step_index"] + 1
    return {"step_index": i, "log": [f"exec(step={i})"]}


def critic(state: State):
    return {"log": ["critic→escalate"]}


def route_after_planner(state: State, *, guarded: bool) -> str:
    if not guarded:
        return "executor"   # 无闸门：永远继续，靠 recursion_limit 兜底
    if state.get("step_index", 0) >= MAX_STEPS or state.get("replan_count", 0) >= MAX_REPLAN:
        return "finalize"
    return "executor"


def route_after_critic(state: State) -> str:
    return "planner"        # 桩：恒 escalate 回 planner（制造外循环压力）


def finalize(state: State):
    return {}


def build(guarded: bool):
    from functools import partial
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("critic", critic)
    g.add_node("finalize", finalize)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", partial(route_after_planner, guarded=guarded),
                            {"executor": "executor", "finalize": "finalize"})
    g.add_edge("executor", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"planner": "planner"})
    g.add_edge("finalize", END)
    return g.compile()


def main():
    print("=== E4: 外循环双闸门 ===\n")

    final = build(guarded=True).invoke({"step_index": 0, "replan_count": 0, "log": []})
    # 应被 replan_count 闸门先收口（escalate 每轮推 +1，2 轮即到上限）
    ok_stop = final.get("replan_count", 0) >= MAX_REPLAN or final.get("step_index", 0) >= MAX_STEPS
    print("[有双闸门]")
    print(f"  收口时 step_index={final['step_index']}  replan_count={final['replan_count']}")
    print(f"  log: {final['log']}")

    print("\n[对照 — 无闸门：单靠 recursion_limit]")
    raised = False
    try:
        build(guarded=False).invoke({"step_index": 0, "replan_count": 0, "log": []},
                                    {"recursion_limit": 10})
    except GraphRecursionError as e:
        raised = True
        print(f"  抛出 GraphRecursionError（如期）：{str(e)[:55]}...")

    print("\n[通过判据]")
    print(f"  双闸门在框架上限前先收口        : {'✓ PASS' if ok_stop else '✗ FAIL'}")
    print(f"  对照：无闸门撞 recursion_limit 抛错: {'✓ PASS' if raised else '✗ FAIL'}")
    print("    └ 结论：step_index / replan_count 是外循环主动收口，")
    print("      框架 recursion_limit 只当最后兜底（同内层 turn_count 闸门）。")
    allok = ok_stop and raised
    print(f"\nE4 {'全部通过 ✓' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
