"""
E2 — task_description 四要素完整传递（objective/output_format/tools_hint/boundary）
验：supervisor 派活的契约四要素齐全，尤其第 4 条 boundary 不被吞（v5.0 唯一系统性缺的那条）。
对照组：漏 boundary 可被检出（缺要素 worker 会漂）。
单独跑：python e2_task_description.py
对应决策 C、D。
"""
from multiagent_harness import make_supervisor, report

FOUR = {"objective", "output_format", "tools_hint", "boundary"}


def test():
    td = make_supervisor({})({})["task_description"]
    main = set(td) >= FOUR and bool(td["boundary"])
    # 对照组：drop_boundary → 缺第 4 要素
    td_bad = make_supervisor({"drop_boundary": True})({})["task_description"]
    ctrl = "boundary" not in td_bad
    detail = f"四要素齐全={main}（boundary='{td.get('boundary','')[:12]}…'）；漏 boundary 可检出={ctrl}"
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E2", "task_description 四要素传递", ok, detail)
