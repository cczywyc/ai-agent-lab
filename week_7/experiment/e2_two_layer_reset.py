"""
E2 — 两层重置并存：per-subtask 标志每步清零，per-task 状态跨步保留
验证：v4.2 的"per-query 重置"在 v5.0 降一层——内层标志（has_retrieved 等）每个子任务
      清零，外层状态（step_results）跨子任务累加；二者都由 init 跨问题清零。
对照：去掉 step 转移重置，看 has_retrieved 从上一子任务泄漏到下一子任务。
对应草稿：设计问题 ①（§一 重置职责升级）/ 周三验证清单 item 2。
桩节点：executor 桩进门先记下看到的 has_retrieved（验是否被上一步泄漏），再置 True。
"""
from functools import partial
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

PER_SUBTASK_DEFAULTS = {"has_retrieved": False, "turn_count": 0}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list                 # per-task：跨步保留
    step_index: int            # per-task
    step_results: list         # per-task：跨步累加（节点内手动）
    has_retrieved: bool        # per-subtask：每步应清零
    leak_log: list             # executor 进门看到的 has_retrieved（验泄漏）


def init(state: State):
    # 跨问题清零（per-task + per-subtask 全套）
    return {"plan": [{"id": 0}, {"id": 1}], "step_index": 0,
            "step_results": [], **PER_SUBTASK_DEFAULTS, "leak_log": []}


def step_reset(state: State, *, enabled: bool):
    """step 转移：开始新子任务时重置 per-subtask 标志（per-task 不动）。"""
    if not enabled:
        return {}                          # 对照：不重置
    return dict(PER_SUBTASK_DEFAULTS)      # 只打回 per-subtask，不碰 step_results


def executor(state: State):
    # 进门先记下看到的 has_retrieved（重置生效则为 False；泄漏则为上一步的 True）
    seen = state.get("has_retrieved", False)
    sr = state.get("step_results", []) + [{"step_id": state["step_index"], "text": "stub"}]
    return {"leak_log": state.get("leak_log", []) + [seen],
            "has_retrieved": True, "step_results": sr,
            "step_index": state["step_index"] + 1}


def route(state: State) -> str:
    return "step_reset" if state["step_index"] < len(state["plan"]) else "finalize"


def finalize(state: State):
    return {}


def build(reset_enabled: bool):
    g = StateGraph(State)
    g.add_node("init", init)
    g.add_node("step_reset", partial(step_reset, enabled=reset_enabled))
    g.add_node("executor", executor)
    g.add_node("finalize", finalize)
    g.add_edge(START, "init")
    g.add_edge("init", "step_reset")
    g.add_edge("step_reset", "executor")
    g.add_conditional_edges("executor", route,
                            {"step_reset": "step_reset", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def main():
    print("=== E2: 两层重置并存 ===\n")

    final = build(reset_enabled=True).invoke({"messages": []})
    ok_subtask = final["leak_log"] == [False, False]   # 每步进门都 False = 重置生效
    ok_task = len(final["step_results"]) == 2           # step_results 跨两步累加

    print("[有 step 转移重置]")
    print(f"  每步进门 has_retrieved : {final['leak_log']}  (期望 [False, False])")
    print(f"  step_results 累加条数   : {len(final['step_results'])}  (期望 2)")

    leaked = build(reset_enabled=False).invoke({"messages": []})
    ok_leak = leaked["leak_log"] == [False, True]       # 对照：第二步进门看到泄漏的 True
    print("\n[对照 — 无 step 重置]")
    print(f"  每步进门 has_retrieved : {leaked['leak_log']}  (泄漏则为 [False, True])")

    print("\n[通过判据]")
    print(f"  per-subtask 标志每步清零        : {'✓ PASS' if ok_subtask else '✗ FAIL'}")
    print(f"  per-task step_results 跨步保留  : {'✓ PASS' if ok_task else '✗ FAIL'}")
    print(f"  对照组确实泄漏（证明重置必要）  : {'✓ PASS' if ok_leak else '✗ FAIL'}")
    allok = ok_subtask and ok_task and ok_leak
    print(f"\nE2 {'全部通过 ✓' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
