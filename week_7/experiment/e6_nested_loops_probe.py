"""
E6（补充探针）— 嵌套双循环：内层 turn_count 循环嵌在外层 step 循环里
为什么补：E1–E5 都是单机制隔离验。v5.0 真实形态是 executor 引擎（agent↔tools，
内层 turn_count 闸门）跑在外层 plan 循环（step_index/replan 闸门）里。组合行为没验过。
重点验三件 E1–E5 照不到的事：
  (a) step 转移真能把内层 turn_count 归零（E2 用的是不跑内层循环的平凡桩）；
  (b) 内外两道闸门嵌套时各自照常收口、不互相误伤；
  (c) recursion_limit 在嵌套下还能不能只当"兜底"——内层每轮也吃同一份预算。
桩：agent/tools 来回跑构成内层循环，每子任务跑 INNER_TURNS 轮工具后 stop。
"""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

MAX_STEPS = 4        # 外层：子任务上限（草稿建议 5–8，这里取 4 个就够暴露问题）
MAX_TURNS = 6        # 内层：单子任务 tool-use 上限（v4.2 原样）
INNER_TURNS = 2      # 桩里每子任务实际跑几轮工具（很温和的设定）
PER_SUBTASK_DEFAULTS = {"turn_count": 0, "inner_phase": "call"}


class State(TypedDict):
    step_index: int
    turn_count: int               # per-subtask：每子任务应归零
    inner_phase: str              # per-subtask 内层状态机：call / stop
    first_turn_log: Annotated[list, add]   # 每子任务首次进 agent 时的 turn_count
    node_log: Annotated[list, add]         # 每个节点跑一次记一条（数 super-step）


def init(state: State):
    return {"step_index": 0, **PER_SUBTASK_DEFAULTS}


def planner(state: State):
    return {"node_log": ["planner"]}


def step_reset(state: State):
    # 外层 step 转移：归零内层 per-subtask 状态（含 turn_count）
    return {**PER_SUBTASK_DEFAULTS, "node_log": ["step_reset"]}


def assemble(state: State):
    return {"node_log": ["assemble"]}


def agent(state: State):
    t = state.get("turn_count", 0) + 1
    upd = {"turn_count": t, "node_log": ["agent"]}
    # 每子任务首次进 agent（t==1）记下 turn_count，验 step_reset 归零生效
    if t == 1:
        upd["first_turn_log"] = [t]
    # 跑满 INNER_TURNS 轮工具就 stop，否则继续调工具
    upd["inner_phase"] = "stop" if t > INNER_TURNS else "call"
    return upd


def tools(state: State):
    return {"node_log": ["tools"]}


def critic(state: State):
    return {"node_log": ["critic"]}


def route_after_agent(state: State) -> str:
    # 内层闸门：未到 stop 且未超 turn_count → 回 tools 再来一轮；否则交给 critic
    if state.get("inner_phase") == "stop" or state.get("turn_count", 0) >= MAX_TURNS:
        return "critic"
    return "tools"


def route_after_critic(state: State) -> str:
    # 桩里 critic 恒 accept → 回 planner 推进下一步
    return "planner"


def route_after_planner(state: State) -> str:
    # 外层闸门：步数到顶 → finalize；否则下一子任务（先 step_reset）
    if state.get("step_index", 0) >= MAX_STEPS:
        return "finalize"
    return "step_reset"


def step_advance(state: State):
    # step_reset 之后推进 step_index（拆开是为了让 step_reset 只管归零）
    return {"step_index": state["step_index"] + 1, "node_log": ["advance"]}


def finalize(state: State):
    return {"node_log": ["finalize"]}


def build():
    g = StateGraph(State)
    for name, fn in [("init", init), ("planner", planner), ("step_reset", step_reset),
                     ("step_advance", step_advance), ("assemble", assemble),
                     ("agent", agent), ("tools", tools), ("critic", critic),
                     ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "init")
    g.add_edge("init", "planner")
    g.add_conditional_edges("planner", route_after_planner,
                            {"step_reset": "step_reset", "finalize": "finalize"})
    g.add_edge("step_reset", "step_advance")
    g.add_edge("step_advance", "assemble")
    g.add_edge("assemble", "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "critic": "critic"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("critic", route_after_critic, {"planner": "planner"})
    g.add_edge("finalize", END)
    return g.compile()


def main():
    print("=== E6: 嵌套双循环 + recursion_limit 预算 ===\n")
    graph = build()

    # (a)+(b)：用充裕的 recursion_limit 先跑通，看内层归零 + 总节点数
    final = graph.invoke({"step_index": 0, "turn_count": 0, "inner_phase": "call",
                          "first_turn_log": [], "node_log": []},
                         {"recursion_limit": 200})
    first_turns = final["first_turn_log"]
    total_nodes = len(final["node_log"])
    ok_reset = all(t == 1 for t in first_turns) and len(first_turns) == MAX_STEPS
    print(f"[跑通：{MAX_STEPS} 子任务 × 每步 {INNER_TURNS} 轮工具]")
    print(f"  每子任务首进 agent 的 turn_count : {first_turns}  (期望 {[1]*MAX_STEPS})")
    print(f"  总节点执行次数（≈super-step 数）  : {total_nodes}")

    # (c)：关键探针——用 LangGraph 默认 recursion_limit 跑同一个"正常"任务
    print(f"\n[关键探针：默认 recursion_limit 下跑同一正常任务]")
    hit_default = False
    try:
        graph.invoke({"step_index": 0, "turn_count": 0, "inner_phase": "call",
                      "first_turn_log": [], "node_log": []})  # 不传 = 默认
    except GraphRecursionError as e:
        hit_default = True
        print(f"  ✗ 撞限：{str(e)[:70]}...")
    else:
        print(f"  ✓ 默认 limit 下正常跑完（{total_nodes} 个 super-step 未触顶）")

    print("\n[通过判据 / 发现]")
    print(f"  (a) step 转移把内层 turn_count 归零      : {'✓ PASS' if ok_reset else '✗ FAIL'}")
    print(f"  (b) 内外两层闸门嵌套下都正常收口          : {'✓ PASS' if len(first_turns)==MAX_STEPS else '✗ FAIL'}")
    if hit_default:
        print(f"  (c) ⚠ 正常任务在默认 recursion_limit 下就撞限！")
        print(f"      → v5.0 必须显式调高 recursion_limit。")
    else:
        print(f"  (c) ✓ 默认 recursion_limit 远大于本任务的 {total_nodes} 步——")
        print(f"      实测 LangGraph 1.2.4 默认值很高（单节点自环 250 步仍不触顶），")
        print(f"      原假设'正常嵌套会撞限'被证伪：recursion_limit 确实只当兜底成立。")
        print(f"      仅当 MAX_STEPS×MAX_TURNS 很大（如 8×6）时才需复核一次。")


if __name__ == "__main__":
    main()
