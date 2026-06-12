"""
E4 — review_count 与 replan_count 不串扰（两个独立计数器）
验：制造"一次子任务 skip(+replan) + 恒 reject 跑满 review"，两计数器各记各的。
对照组：合并成一个计数器 → skip 先占一格，闸门提前收口（skip 吃掉一次返修额度）。
单独跑：python e4_counters.py
对应决策 E。

★本条首跑暴露的精确化：对照组若让计数器写字段与闸门读字段不一致，
不是"少跑几次 review"而是直接死锁撞 recursion——计数器写字段 ≡ 闸门读字段。
"""
from multiagent_harness import run, report


def test():
    # 主组：独立计数器，researcher skip 一次 + 恒 reject 跑满 2 次 review
    res = run({"researcher_skips_once": True})
    main = (res.get("replan_count") == 1 and res.get("review_count") == 2
            and res.get("_n_reviews") == 2)
    # 对照组：合并计数器，skip(+1) 占一格 → 1 次 review 后闸门就收口
    res2 = run({"researcher_skips_once": True, "shared_counter": True})
    ctrl = (res2.get("replan_count") == 2 and res2.get("review_count", 0) == 0
            and res2.get("_n_reviews") == 1)
    detail = (f"独立: review 跑满 {res.get('_n_reviews')} 次（replan={res.get('replan_count')}/"
              f"review={res.get('review_count')}）；合并: skip 吃额度→只 review {res2.get('_n_reviews')} 次")
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E4", "两计数器不串扰", ok, detail)
