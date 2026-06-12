"""
E6 — best-so-far 收口（防 behavioral collapse）
验：达 MAX_REVIEW 仍未 accept 时，收口取历史评分最高的稿，而非最新稿。
评分序列 [0.8, 0.5]：预算内最新稿(0.5)比历史最好(0.8)更差。
对照组：取最新 → 交付更差的 0.5。
单独跑：python e6_best_so_far.py
对应决策 E。

★本条首跑暴露的精确化：best-so-far 仅在"MAX_REVIEW 预算内真发生 collapse"时 load-bearing；
若 writer 单调变好则 best≡latest、机制空转。保留它（极低成本兜底），但它是很少触发的安全网。
"""
from multiagent_harness import run, report


def test():
    res = run({"best_so_far": True, "scores": [0.8, 0.5]})
    chosen = res["worker_result"]["chosen"]
    main = abs(chosen["score"] - 0.8) < 1e-9
    # 对照组：取最新 → 0.5（更差）
    chosen2 = run({"best_so_far": False, "scores": [0.8, 0.5]})["worker_result"]["chosen"]
    ctrl = abs(chosen2["score"] - 0.5) < 1e-9
    detail = f"best-so-far 取={chosen['score']}（历史最好）；取最新={chosen2['score']}（更差）"
    return main and ctrl, detail


if __name__ == "__main__":
    ok, detail = test()
    report("E6", "best-so-far 收口", ok, detail)
