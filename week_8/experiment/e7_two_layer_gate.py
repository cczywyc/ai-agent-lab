"""
E7 — 内外两层闸门嵌套 + recursion_limit 只兜底
验：researcher 内层 turn_count（每次进 researcher 从 0 起、产出固定 3 轮）与外层 review_count 互不误伤；
恒 reject 时外层 review 闸门先收口（非 recursion）；总 super-step 远小于 recursion_limit=10007。
对照组：拆外层闸门 → 撞 recursion。
单独跑：python e7_two_layer_gate.py
对应决策 G、H。
"""
from multiagent_harness import run, build, report, MAX_REVIEW
from langgraph.errors import GraphRecursionError


def test():
    res = run({"gate_on": True})                       # 恒 reject，外层闸门收口
    reset_ok = res.get("turn_count") == 3              # researcher 内层从 0 跑 3 轮
    gate_first = res.get("review_count") == MAX_REVIEW  # 外层 review 闸门收口（非 recursion）
    # 收口总 super-step（对比 recursion_limit=10007）
    n_steps = len(list(build({"gate_on": True}).stream(
        {"review_count": 0, "replan_count": 0}, {"recursion_limit": 50})))
    main = reset_ok and gate_first and n_steps < 50
    # 对照组：拆外层闸门 → 撞 recursion
    ctrl = False
    try:
        run({"gate_on": False}, recursion_limit=12)
    except GraphRecursionError:
        ctrl = True
    detail = (f"turn_count 内层={res.get('turn_count')}；外层 review 闸门收口={gate_first}；"
              f"收口 super-step={n_steps}（距 10007 约 {10007 // n_steps} 倍）；拆闸门撞 recursion={ctrl}")
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E7", "双层闸门嵌套", ok, detail)
