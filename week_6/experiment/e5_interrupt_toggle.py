"""
E5 — human_review interrupt 开关透明性(决策 F)
验证:human_review 节点里 `if INTERRUPT_ENABLED: interrupt(...)`——
  关闭时完全透明(不暂停,一路到 finalize);
  打开时在此暂停(stream 吐 __interrupt__),再用 Command(resume=...) 接着跑完。
注:interrupt 必须配 checkpointer 才能用。
对应草稿:设计问题 ④ / 周三验证清单 item 5。
"""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# 用模块级开关模拟 config.INTERRUPT_ENABLED
FLAGS = {"INTERRUPT_ENABLED": False}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    review: str


def make_answer(state: State):
    return {"messages": [{"role": "assistant", "content": "draft answer"}]}


def human_review(state: State):
    if FLAGS["INTERRUPT_ENABLED"]:
        decision = interrupt("approve the answer?")   # 暂停,等 Command(resume=...)
        return {"review": decision}
    return {"review": "auto(disabled)"}               # 透明放行


def finalize(state: State):
    return {"messages": [{"role": "assistant", "content": "final delivered"}]}


def build():
    g = StateGraph(State)
    g.add_node("make_answer", make_answer)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "make_answer")
    g.add_edge("make_answer", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())   # interrupt 必需


def stream_events(graph, inp, cfg):
    events = []
    for ev in graph.stream(inp, cfg, stream_mode="updates"):
        events.append(ev)
    return events


def main():
    print("=== E5: interrupt 开关透明性 ===\n")
    graph = build()

    # --- A: 开关关闭 -> 应透明,无 __interrupt__,直达 finalize ---
    FLAGS["INTERRUPT_ENABLED"] = False
    cfg_a = {"configurable": {"thread_id": "off"}}
    ev_a = stream_events(graph, {"messages": [{"role": "user", "content": "Q"}]}, cfg_a)
    interrupted_a = any("__interrupt__" in e for e in ev_a)
    final_a = graph.get_state(cfg_a).values
    print("[A 开关 OFF]")
    print(f"  出现 __interrupt__ : {interrupted_a}   (期望 False)")
    print(f"  review 字段        : {final_a.get('review')!r}   (期望 'auto(disabled)')")
    print(f"  跑到 finalize      : {'final delivered' in [m.content if hasattr(m,'content') else m['content'] for m in final_a['messages']]}")

    # --- B: 开关打开 -> 应在 human_review 暂停,再 resume 跑完 ---
    FLAGS["INTERRUPT_ENABLED"] = True
    cfg_b = {"configurable": {"thread_id": "on"}}
    ev_b1 = stream_events(graph, {"messages": [{"role": "user", "content": "Q"}]}, cfg_b)
    interrupted_b = any("__interrupt__" in e for e in ev_b1)
    state_mid = graph.get_state(cfg_b)
    paused = len(state_mid.next) > 0    # 还有待执行节点 = 被暂停
    print("\n[B 开关 ON]")
    print(f"  出现 __interrupt__ : {interrupted_b}   (期望 True)")
    print(f"  暂停在节点         : {state_mid.next}   (期望非空:卡在 human_review)")

    # resume
    ev_b2 = stream_events(graph, Command(resume="approved"), cfg_b)
    final_b = graph.get_state(cfg_b).values
    resumed_done = len(graph.get_state(cfg_b).next) == 0
    print(f"  resume 后 review   : {final_b.get('review')!r}   (期望 'approved')")
    print(f"  resume 后跑完      : {resumed_done}   (期望 True)")

    ok_off = (not interrupted_a) and final_a.get("review") == "auto(disabled)"
    ok_on_pause = interrupted_b and paused
    ok_on_resume = final_b.get("review") == "approved" and resumed_done

    print("\n[通过判据]")
    print(f"  OFF 透明放行 (无 interrupt)         : {'✓ PASS' if ok_off else '✗ FAIL'}")
    print(f"  ON 在 human_review 暂停            : {'✓ PASS' if ok_on_pause else '✗ FAIL'}")
    print(f"  ON resume(Command) 后跑完          : {'✓ PASS' if ok_on_resume else '✗ FAIL'}")
    allok = ok_off and ok_on_pause and ok_on_resume
    print(f"\nE5 {'全部通过 ✓ —— 决策 F 坐实' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
