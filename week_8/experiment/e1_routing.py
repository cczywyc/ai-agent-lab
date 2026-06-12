"""
E1 — supervisor 路由正确性（routing accuracy）★最先跑
验：supervisor 按当前阶段派给正确 worker（无 findings→researcher；有 findings 无 draft→writer；有 draft→reviewer）。
把这 3 条沉淀成 routing_accuracy 评测种子集（第十周评测周扩到 20+ 条）。
对照组：写歪的路由（恒派 writer、无视阶段）被 routing_accuracy 评测真实检出（命中 < 3/3）。
单独跑：python e1_routing.py
对应决策 A、B。

★本条首跑暴露的精确化：草稿/实验计划原定 E1 对照组用 prebuilt create_supervisor 做"对拍基线"，
但实测 prebuilt 的路由是 LLM 工具调用决策（需绑定 model），不是零 API——与桩测"零 API"纪律冲突。
故对照组改为零 API 的"写歪路由"探针：真跑一个恒派 writer 的坏 supervisor，让 routing_accuracy 评测抓它。
"""
from multiagent_harness import make_supervisor, report

# routing_accuracy 评测种子集（state 快照 → 期望 worker）
SEED = [
    ({}, "researcher"),                             # 无 findings → researcher
    ({"findings": [1]}, "writer"),                  # 有 findings 无 draft → writer
    ({"findings": [1], "draft": "d"}, "reviewer"),  # 有 draft → reviewer
]


def routing_accuracy(cfg):
    sup = make_supervisor(cfg)
    hits = [sup(st)["active_worker"] == exp for st, exp in SEED]
    return sum(hits), len(hits)


def test():
    hit, n = routing_accuracy({})                   # 手搓路由
    main = hit == n                                 # 3/3 派对
    bad_hit, _ = routing_accuracy({"route_bug": True})  # 写歪：恒派 writer、无视阶段
    ctrl = bad_hit < n                              # 评测能抓出写歪（命中 < 3）
    detail = (f"routing_accuracy={hit}/{n} 派对；"
              f"对照(恒派writer)={bad_hit}/{n}→评测可检出写歪={ctrl}")
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E1", "supervisor 路由正确性", ok, detail)
