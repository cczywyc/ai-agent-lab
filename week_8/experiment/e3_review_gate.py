"""
E3 — review_count 打回闸门收口
验：reviewer 恒 reject 时，外层 review_count 闸门先于 recursion_limit 收口（停在 MAX_REVIEW、走 best-so-far）。
对照组：拆掉闸门只靠 recursion_limit → 抛 GraphRecursionError。
单独跑：python e3_review_gate.py
对应决策 E。
"""
from multiagent_harness import run, report, MAX_REVIEW
from langgraph.errors import GraphRecursionError


def test():
    # 主组：gate ON，恒 reject → 停在 review_count==MAX_REVIEW、走 finalize
    res = run({"gate_on": True})
    main = res.get("review_count") == MAX_REVIEW and res["worker_result"]["kind"] == "final"
    # 对照组：gate OFF，只靠 recursion_limit
    ctrl = False
    try:
        run({"gate_on": False}, recursion_limit=12)
    except GraphRecursionError:
        ctrl = True
    detail = f"闸门收口@review_count={res.get('review_count')}/{MAX_REVIEW}；拆闸门撞 recursion={ctrl}"
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E3", "review_count 闸门收口", ok, detail)
