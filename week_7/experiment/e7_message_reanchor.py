"""
E7（补充探针）— 跨子任务 messages 重锚：executor 窗口靠 SYSTEM_PROMPT 锚隔离
为什么补：messages 是全图唯一带 add_messages 的累加字段，且 init 故意不重置它
（保留 thread 历史）。v5.0 里 executor 引擎每个子任务都往 messages 追加（AI 回复 +
tool 结果），跨子任务只增不减。那"executor 只看当前子任务"（草稿 §四）靠什么实现？
答案是复用 v4.x 的窗口切片：agent 发给模型的窗口 = 最后一条 content==SYSTEM_PROMPT
的 system 消息起（见 nodes._window_start）。本探针验：每个子任务的 assemble 重新产出
SYSTEM_PROMPT 锚后，子任务 2 的 executor 窗口是否只含子任务 2 的块、不漏子任务 1。
对照：assemble 不重产 SYSTEM_PROMPT（只追加 Human）→ 窗口锚死在子任务 1，发生串台。
桩：用真实的 _window_start 逻辑 + LangChain 消息对象，不调模型。
"""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage,
)

SYSTEM_PROMPT = "SYSTEM_PROMPT::research-agent"   # 桩里的段 1 锚（真实工程是 config.SYSTEM_PROMPT）
N_SUBTASKS = 2


def _window_start(messages: list) -> int:
    """原样复刻 nodes._window_start：最后一条 content==SYSTEM_PROMPT 的下标，找不到退回最后一条 system。"""
    idx, fallback, found = 0, 0, False
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            fallback = i
            if m.content == SYSTEM_PROMPT:
                idx, found = i, True
    return idx if found else fallback


def _window(messages: list) -> list:
    return messages[_window_start(messages):]


class State(TypedDict):
    messages: Annotated[list, add_messages]   # 真实 reducer：累加、从不按子任务清
    step_index: int
    window_sizes: list        # 每子任务 agent 看到的窗口大小
    bled_prev: list           # 每子任务 agent 窗口里是否混进更早子任务的 tool 结果


def assemble(state: State, *, reanchor: bool):
    """role=executor 的桩：重产 SYSTEM_PROMPT 锚（reanchor=True）+ 本子任务 query。"""
    k = state["step_index"]
    new = []
    if reanchor:
        new.append(SystemMessage(content=SYSTEM_PROMPT))   # 段 1 锚 → 窗口重锚到这里
    new.append(HumanMessage(content=f"[subtask {k}] query"))
    return {"messages": new}


def agent(state: State):
    """executor 引擎的桩：取窗口、记录、再追加本子任务的 AI 回复 + tool 结果。"""
    k = state["step_index"]
    win = _window(state["messages"])
    # 窗口里有没有更早子任务（< k）的 tool 结果 = 串台
    bled = any(isinstance(m, ToolMessage) and f"subtask " in (m.content or "")
               and m.content != f"[subtask {k}] tool result"
               for m in win)
    sizes = state.get("window_sizes", []) + [len(win)]
    bleds = state.get("bled_prev", []) + [bled]
    return {"messages": [AIMessage(content=f"[subtask {k}] summary"),
                         ToolMessage(content=f"[subtask {k}] tool result", tool_call_id=f"t{k}")],
            "window_sizes": sizes, "bled_prev": bleds,
            "step_index": k + 1}


def route(state: State) -> str:
    return "assemble" if state["step_index"] < N_SUBTASKS else "finalize"


def finalize(state: State):
    return {}


def build(reanchor: bool):
    from functools import partial
    g = StateGraph(State)
    g.add_node("assemble", partial(assemble, reanchor=reanchor))
    g.add_node("agent", agent)
    g.add_node("finalize", finalize)
    g.add_edge(START, "assemble")
    g.add_edge("assemble", "agent")
    g.add_conditional_edges("agent", route, {"assemble": "assemble", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def _run(reanchor: bool):
    return build(reanchor).invoke({"messages": [], "step_index": 0,
                                   "window_sizes": [], "bled_prev": []})


def main():
    print("=== E7: 跨子任务 messages 重锚 ===\n")

    # 设计方案：每子任务重产 SYSTEM_PROMPT 锚
    f = _run(reanchor=True)
    total = len(f["messages"])
    # 全局 messages 累加（两子任务各 3 条：system+human+（ai+tool=2）→ 实际 4 条/子任务）
    print("[重锚：每子任务 assemble 重产 SYSTEM_PROMPT]")
    print(f"  全局 messages 累加条数 : {total}（跨子任务只增不减，init 不清）")
    print(f"  每子任务 executor 窗口大小 : {f['window_sizes']}")
    print(f"  每子任务窗口是否串台早期 tool: {f['bled_prev']}  (期望 [False, False])")
    ok_accum = total == N_SUBTASKS * 4          # 每子任务 4 条消息累加
    ok_isolated = f["bled_prev"] == [False, False]
    # 窗口大小应稳定（每子任务窗口 = 自己那段），不随累加增长
    ok_window_flat = len(set(f["window_sizes"])) == 1

    # 对照：assemble 不重锚（只追加 Human）
    leaked = _run(reanchor=False)
    print("\n[对照：assemble 不重产 SYSTEM_PROMPT（窗口锚死子任务 0）]")
    print(f"  每子任务 executor 窗口大小 : {leaked['window_sizes']}（应随累加变大）")
    print(f"  每子任务窗口是否串台早期 tool: {leaked['bled_prev']}  (串台则 [.., True])")
    ok_control_bleed = leaked["bled_prev"][-1] is True

    print("\n[通过判据 / 发现]")
    print(f"  全局 messages 跨子任务累加（init 不清）   : {'✓ PASS' if ok_accum else '✗ FAIL'}")
    print(f"  重锚后窗口隔离、不串台早期子任务          : {'✓ PASS' if ok_isolated else '✗ FAIL'}")
    print(f"  重锚后窗口大小稳定（不随 messages 增长）  : {'✓ PASS' if ok_window_flat else '✗ FAIL'}")
    print(f"  对照证明：不重锚则窗口锚死、发生串台      : {'✓ PASS' if ok_control_bleed else '✗ FAIL'}")
    print("    └ 结论：v5.0 的'executor 看局部'不是新机制——assemble(role=executor)")
    print("      每子任务重产 SYSTEM_PROMPT 锚，复用 _window_start 切片即天然隔离。")
    print("      草稿 §四 必须写明这条；漏了就会让后一子任务看到前面全部 tool 历史。")
    allok = ok_accum and ok_isolated and ok_window_flat and ok_control_bleed
    print(f"\nE7 {'全部通过 ✓' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
