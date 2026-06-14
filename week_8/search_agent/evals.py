"""
evals.py — v6.0 可复用评测（E1 沉淀）。

第八周桩测 E1 把"supervisor 按阶段派对 worker"沉淀成可复用的 routing_accuracy 评测，而非一次性断言。
本文件提供：
  ROUTING_SEED        三级阶段判定种子集（无 findings→researcher / 有 findings 无 draft→writer /
                      有 draft 无 verdict→reviewer）。**第十周评测周扩到 20+ 条**。
  routing_accuracy()  跑种子集、返回命中数——既验"对的路由能跑"，也验"写歪路由被抓出"（对照组）。

设计依据：草稿 v0.2 §一 决策 A′（routing_accuracy 函数）+ 已验证清单 1。
真实 supervisor 路由是 nodes._supervisor_route（条件函数、不接 LLM——E1）。
"""

from nodes import _supervisor_route


def _seed_state(plan, step_index, findings, draft, review_verdict, replan_count=0):
    """构造一个 routing 种子 state（只放 _supervisor_route 实际会读的键）。"""
    return {
        "user_message": "调研多 Agent 并写技术综述",
        "plan": plan, "step_index": step_index, "replan_count": replan_count,
        "findings": findings, "draft": draft, "review_verdict": review_verdict,
    }


_PLAN2 = [{"id": 0, "query": "拓扑", "status": "pending"},
          {"id": 1, "query": "契约", "status": "pending"}]
_PLAN2_DONE = [{"id": 0, "query": "拓扑", "status": "done"},
               {"id": 1, "query": "契约", "status": "done"}]
_FINDING = [{"subtask": "拓扑", "point": "A/B/C", "citations": ["d#s"], "status": "ok"}]

# E1 三条种子（三级跌落）。每条 = {name, state, expected}。
ROUTING_SEED = [
    {"name": "研究未做完（step_index<len, findings 空） → researcher",
     "state": _seed_state(_PLAN2, 0, [], "", ""), "expected": "researcher"},
    {"name": "研究做完、有 findings 无 draft → writer",
     "state": _seed_state(_PLAN2_DONE, 2, _FINDING, "", ""), "expected": "writer"},
    {"name": "有 draft 无 verdict → reviewer",
     "state": _seed_state(_PLAN2_DONE, 2, _FINDING, "已有初稿 [d#s]", ""), "expected": "reviewer"},
]


def _real_route(state) -> str:
    """真实 supervisor 路由：跑 _supervisor_route、取 active_worker。"""
    return _supervisor_route(state).get("active_worker", "")


def routing_accuracy(route_fn=None, seed=ROUTING_SEED):
    """跑路由种子集，返回 (命中数, 总数, 明细列表[(name, expected, got, ok)])。
    route_fn(state)->worker 字符串；默认用真实 _supervisor_route。对照组传"写歪路由"（如恒派 writer）验评测能抓出。"""
    route_fn = route_fn or _real_route
    hits, detail = 0, []
    for case in seed:
        got = route_fn(case["state"])
        ok = (got == case["expected"])
        hits += ok
        detail.append((case["name"], case["expected"], got, ok))
    return hits, len(seed), detail


def bad_route_always_writer(state) -> str:
    """对照组：写歪的路由（恒派 writer、无视阶段）——routing_accuracy 应只命中"有 findings 无 draft"那条（1/3）。"""
    return "writer"


if __name__ == "__main__":
    hits, total, detail = routing_accuracy()
    print(f"routing_accuracy（真实路由）= {hits}/{total}")
    for name, exp, got, ok in detail:
        print(f"  {'✓' if ok else '✗'} {name}：expected={exp} got={got}")
    bhits, btotal, _ = routing_accuracy(bad_route_always_writer)
    print(f"对照（恒派 writer）= {bhits}/{btotal}（评测能抓出写歪 = {bhits < total}）")
