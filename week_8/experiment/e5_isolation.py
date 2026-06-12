"""
E5 — 上下文隔离（worker 只看该看的）★A→B 唯一的真·新机制
验：researcher 只见 task_description、writer 只见 findings+review_notes、reviewer 只见 draft。
对照组：关隔离 + 恒 reject（writer 返修第二遍）→ writer 越界读到 reviewer 私有字段（draft_score/review_verdict）= 串台。
单独跑：python e5_isolation.py
对应决策 D、F。

★本条首跑暴露的精确化：串台只在"返修第二遍"才暴露（首轮 writer 在 reviewer 之前跑、
无私有字段可越界）。隔离必须在返修路径上验，不能只验 happy path。
"""
from multiagent_harness import run, report


def test():
    res = run({"isolation_on": True, "accept_at_review": 0})
    rv = set(res.get("_researcher_visible", []))
    wv = set(res.get("_writer_visible", []))
    main = rv == {"task_description"} and wv == {"findings", "review_notes"}
    # 对照组：关隔离 + 恒 reject → writer 返修第二遍时读全 state，越界看到 reviewer 私有
    leaked = set(run({"isolation_on": False}).get("_writer_visible", []))
    ctrl = "draft_score" in leaked and "review_verdict" in leaked
    detail = f"researcher 见={sorted(rv)}；writer 见={sorted(wv)}；隔离关→writer 越界读到 reviewer 私有={ctrl}"
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E5", "上下文隔离", ok, detail)
