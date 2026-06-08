"""
E5 — 两档重试计数器互不串扰：empty_retries（传输层）vs retry_count（业务层）
验证：agent 节点内的空回答重试（empty_retries，不进拓扑、不耗 turn）与 critic 驱动的
      业务重试（retry_count，走 retry 边回 executor）是两个独立计数器——一次 API 抖动
      不会吃掉一次业务重试额度。这是本周最该亲眼看到的一条（职责边界 §4 末尾）。
对应草稿：设计问题 ③（§三 三级重试）/ 周三验证清单 item 5 / 决策 B。
桩节点：agent 桩在节点内 while 模拟"先空一次再成功"（empty_retries+1，不画进图）；
        critic 桩头一次发 retry、第二次 accept（retry_count 经 retry 边 +1）。
"""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

MAX_RETRY = 2


class State(TypedDict):
    turn_count: int
    empty_retries: int       # 传输层：节点内累加，不进拓扑
    retry_count: int         # 业务层：critic 驱动，走 retry 边
    critic_verdict: str      # critic 写、route 读（必须声明，否则更新被丢弃）
    business_retry_done: bool
    log: Annotated[list, add]


def agent(state: State):
    """桩：节点内 while 重试一次（模拟 API fast-fail 空回答），不耗 turn_count 之外的拓扑。"""
    retried = 0
    while True:
        empty = (retried == 0)        # 第一次"空"，重试后非空
        if empty and retried < 1:
            retried += 1
            continue
        break
    return {"turn_count": state.get("turn_count", 0) + 1,
            "empty_retries": state.get("empty_retries", 0) + retried,   # 本步手动累加
            "log": [f"agent(empty_retried={retried})"]}


def critic(state: State):
    """桩：第一次裁 retry（业务层重做该步），第二次裁 accept。"""
    if not state.get("business_retry_done", False):
        return {"critic_verdict": "retry",
                "retry_count": state.get("retry_count", 0) + 1,   # 业务层 +1
                "business_retry_done": True,
                "log": ["critic→retry"]}
    return {"critic_verdict": "accept", "log": ["critic→accept"]}


def route_after_critic(state: State) -> str:
    if state.get("critic_verdict") == "retry" and state.get("retry_count", 0) < MAX_RETRY:
        return "agent"          # 业务 retry：回 executor（这里桩用 agent 代表执行）
    return "finalize"


def finalize(state: State):
    return {}


def build():
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("critic", critic)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_edge("agent", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"agent": "agent", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def main():
    print("=== E5: 两档计数器互不串扰 ===\n")
    final = build().invoke({"turn_count": 0, "empty_retries": 0, "retry_count": 0,
                            "business_retry_done": False, "log": []})

    # agent 跑了两遍（首遍 + 业务 retry 回来一遍），每遍各自空重试一次 → empty_retries=2
    # 业务 retry 恰好发生一次 → retry_count=1
    er = final["empty_retries"]
    rc = final["retry_count"]
    print(f"  执行轨迹 log : {final['log']}")
    print(f"  empty_retries (传输层) : {er}")
    print(f"  retry_count   (业务层) : {rc}")

    ok_split = (er == 2 and rc == 1)            # 两个计数器各记各的
    # 关键反证：传输层的 2 次空重试没有挤占业务层额度（业务只发了 1 次 retry）
    ok_no_cross = (rc == 1 and rc < MAX_RETRY)

    print("\n[通过判据]")
    print(f"  两计数器各自独立计数 (empty=2, retry=1): {'✓ PASS' if ok_split else '✗ FAIL'}")
    print(f"  空回答重试未吃掉业务 retry 额度        : {'✓ PASS' if ok_no_cross else '✗ FAIL'}")
    print("    └ 结论：传输层抖动留在 agent 节点内（empty_retries），")
    print("      业务层重做走拓扑（retry_count）——别合成一个计数器。")
    allok = ok_split and ok_no_cross
    print(f"\nE5 {'全部通过 ✓' if allok else '存在失败 ✗'}")


if __name__ == "__main__":
    main()
