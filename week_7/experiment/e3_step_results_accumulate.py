"""
E3 — step_results 节点内手动累加（替换语义，刻意不上 reducer）
验证：step_results 在 executor 节点里"当前 + 新"手动累加跨步攒结论，init 返回 []
      可跨问题清空。这是 v4.2 retrieved_chunks 先例的照搬（决策 G）。
对照：给 step_results 上 operator.add reducer，init 的 return [] 变成"追加空列表"
      的 no-op，清不掉——重演 v4.2 chunks 的 E3 实证。
对应草稿：设计问题 ①（§一 reducer 判断）/ 周三验证清单 item 3。
桩节点：executor 桩只追加一条结论 dict。
"""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


def make_state(use_reducer: bool):
    if use_reducer:
        class State(TypedDict):                    # 对照：带累加 reducer
            step_results: Annotated[list, add]
            step_index: int
    else:
        class State(TypedDict):                    # 设计：替换语义
            step_results: list
            step_index: int
    return State


def build(use_reducer: bool):
    State = make_state(use_reducer)

    def init(state: State):
        # 想清空：替换语义下 return [] 生效；带 reducer 时是"追加空列表"no-op
        return {"step_results": [], "step_index": 0}

    def executor(state: State):
        if use_reducer:
            # 带 reducer：只返回"新增"，框架替你累加
            return {"step_results": [{"step_id": state["step_index"]}],
                    "step_index": state["step_index"] + 1}
        # 替换语义：节点内手动"当前 + 新"
        cur = state.get("step_results", [])
        return {"step_results": cur + [{"step_id": state["step_index"]}],
                "step_index": state["step_index"] + 1}

    def route(state: State) -> str:
        return "executor" if state["step_index"] < 3 else "finalize"

    def finalize(state: State):
        return {}

    g = StateGraph(State)
    g.add_node("init", init)
    g.add_node("executor", executor)
    g.add_node("finalize", finalize)
    g.add_edge(START, "init")
    g.add_edge("init", "executor")
    g.add_conditional_edges("executor", route,
                            {"executor": "executor", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def main():
    print("=== E3: step_results 手动累加 vs reducer ===\n")

    # 设计方案：替换语义 + 节点内手动累加
    g1 = build(use_reducer=False)
    # 预置一条脏数据，验 init 的 [] 能清空它
    final = g1.invoke({"step_results": [{"step_id": 99}], "step_index": 7})
    ok_accum = [r["step_id"] for r in final["step_results"]] == [0, 1, 2]
    print("[替换语义 + 节点内手动累加]")
    print(f"  累加结果 step_id : {[r['step_id'] for r in final['step_results']]}  (期望 [0,1,2])")
    print(f"  init 清空脏数据   : {'是' if 99 not in [r['step_id'] for r in final['step_results']] else '否'}")

    # 对照：带 reducer，init 的 [] 清不掉脏数据
    g2 = build(use_reducer=True)
    leaked = g2.invoke({"step_results": [{"step_id": 99}], "step_index": 7})
    ids = [r["step_id"] for r in leaked["step_results"]]
    ok_noop = 99 in ids   # 脏数据还在 = reducer 下 init 的 [] 是 no-op
    print("\n[对照 — 带 operator.add reducer]")
    print(f"  累加结果 step_id : {ids}")
    print(f"  脏数据 99 是否残留: {'是（init 清不掉）' if ok_noop else '否'}")

    print("\n[通过判据]")
    print(f"  手动累加得 [0,1,2] 且 init 清空生效 : {'✓ PASS' if ok_accum else '✗ FAIL'}")
    print(f"  对照证明 reducer 下 init 清不掉      : {'✓ PASS' if ok_noop else '✗ FAIL'}")
    print("    └ 结论：要累加又要能清空 → 替换语义 + 节点内手动 append（不上 reducer）")
    allok = ok_accum and ok_noop
    print(f"\nE3 {'全部通过 ✓' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
