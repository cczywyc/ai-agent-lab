"""
verify_store_migration.py — v4.0 → v4.1 长期记忆迁移验证（计划 §四 阶段四-14）

针对现有 11 条事实设计的标注查询集，三种模式：
  --baseline  迁移前：旧 LongTermMemory.recall_facts 路径（手动 embed_query），结果存 JSON
  --after     迁移后：LangGraph Store 原生 search 路径（store 内部 embed），结果存 JSON
  --compare   对照两份 JSON，输出 hit@1 / hit@3 是否退化

跑法（cwd = week_6/search_agent）：
  ../../.venv/bin/python verify_store_migration.py --baseline
  ../../.venv/bin/python verify_store_migration.py --after
  ../../.venv/bin/python verify_store_migration.py --compare

注意：--baseline 必须在重构前跑（依赖旧 recall_facts 代码路径）；
     真实调 text-embedding-v3，需要 DASHSCOPE_API_KEY。
"""

import argparse
import json
from pathlib import Path

from config import MEMORY_DIR, MEMORY_FACTS_FILE, MEMORY_FACTS_TOP_K, MEMORY_FACTS_MIN_SCORE

BASELINE_FILE = Path(__file__).parent / "verify_ltm_baseline.json"
AFTER_FILE = Path(__file__).parent / "verify_ltm_after.json"

# ============================================================
# 标注查询集 — expected 为 memory_facts.json 中的事实下标（命中任一即算 hit）
# 事实 #6/#7 是抽取噪音（踩坑 #3），不作为任何查询的期望命中
# ============================================================
LABELED_QUERIES = [
    {"query": "连续失败计数器是怎么重置的？",                  "expected": [0, 3]},
    {"query": "工具连续失败之后什么时候触发降级？",            "expected": [1, 4]},
    {"query": "工具返回的结构化错误元数据有哪些字段？",        "expected": [2, 5]},
    {"query": "recoverable 和 suggestion 字段是干什么用的？",  "expected": [2, 5]},
    {"query": "文档 chunking 是按什么切分的？",                "expected": [8, 10]},
    {"query": "每个 chunk 需要携带哪些元数据？",               "expected": [9]},
    {"query": "为什么没有采用纯语义分块？",                    "expected": [10]},
    {"query": "fetch_webpage 连续 403 失败时 Agent 怎么办？",  "expected": [1, 4]},
    {"query": "consecutive_errors 达到阈值会发生什么？",       "expected": [0, 3, 4]},
    {"query": "Markdown 标题切分策略是怎么设计的？",           "expected": [8]},
    {"query": "成功即重置是什么策略？",                        "expected": [0, 3]},
    {"query": "降级之后 Agent 基于什么内容回答？",             "expected": [1, 4]},
]


def _load_fact_index() -> dict[str, int]:
    """fact 原文 → memory_facts.json 下标（两条路径的召回结果都映射回同一坐标系）。"""
    with open(Path(MEMORY_DIR) / MEMORY_FACTS_FILE, "r", encoding="utf-8") as f:
        facts = json.load(f)
    return {fa["fact"]: i for i, fa in enumerate(facts)}, len(facts)


def _evaluate(recall_fn) -> dict:
    """
    recall_fn(query, top_k, min_score) -> list[{fact, score}]（已按 score 降序）。
    统计 hit@1 / hit@3（min_score 过滤后），并存每条查询的完整 top-5 无过滤明细。
    """
    fact_to_idx, n_facts = _load_fact_index()
    per_query, hit1, hit3 = [], 0, 0

    for item in LABELED_QUERIES:
        q, expected = item["query"], set(item["expected"])

        # 与线上语义一致：top_k=3 + min_score=0.30
        filtered = recall_fn(q, MEMORY_FACTS_TOP_K, MEMORY_FACTS_MIN_SCORE)
        # 诊断用：top5 无阈值
        raw5 = recall_fn(q, 5, -1.0)

        idxs = [fact_to_idx.get(r["fact"], -1) for r in filtered]
        q_hit1 = bool(idxs) and idxs[0] in expected
        q_hit3 = any(i in expected for i in idxs[:3])
        hit1 += q_hit1
        hit3 += q_hit3

        per_query.append({
            "query": q,
            "expected": sorted(expected),
            "recalled_idx_top3": idxs,
            "hit@1": q_hit1,
            "hit@3": q_hit3,
            "filtered_top3": [
                {"idx": fact_to_idx.get(r["fact"], -1), "score": round(r["score"], 4)}
                for r in filtered
            ],
            "raw_top5": [
                {"idx": fact_to_idx.get(r["fact"], -1), "score": round(r["score"], 4)}
                for r in raw5
            ],
        })
        print(f"  [{'✓' if q_hit3 else '✗'}] {q}  top3={idxs} expected={sorted(expected)}")

    n = len(LABELED_QUERIES)
    return {
        "n_facts": n_facts,
        "n_queries": n,
        "top_k": MEMORY_FACTS_TOP_K,
        "min_score": MEMORY_FACTS_MIN_SCORE,
        "hit@1": f"{hit1}/{n}",
        "hit@3": f"{hit3}/{n}",
        "per_query": per_query,
    }


# ============================================================
# 迁移前路径：旧 LongTermMemory（手动 embed_query → recall_facts）
# ============================================================
def run_baseline():
    from rag.embed import embed_query
    from memory.long_term import LongTermMemory

    long = LongTermMemory()
    long.load()
    assert long.facts, "memory_facts.json 为空，没有可验证的数据"
    print(f"基线采集：{len(long.facts)} 条事实，{len(LABELED_QUERIES)} 条标注查询\n")

    _qvec_cache = {}

    def recall(q, top_k, min_score):
        if q not in _qvec_cache:
            _qvec_cache[q] = embed_query(q)
        return long.recall_facts(_qvec_cache[q], top_k=top_k, min_score=min_score)

    res = _evaluate(recall)
    res["mode"] = "baseline (旧 recall_facts)"
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nhit@1={res['hit@1']} hit@3={res['hit@3']} → 已存 {BASELINE_FILE.name}")


# ============================================================
# 迁移后路径：LangGraph Store 原生 search（store 内部 embed）
# ============================================================
def run_after():
    from memory.ltm_store import get_ltm_store, NS_FACTS

    store = get_ltm_store()

    def recall(q, top_k, min_score):
        hits = store.search(NS_FACTS, query=q, limit=top_k)
        return [
            {"fact": h.value["fact"], "score": float(h.score)}
            for h in hits
            if h.score is not None and h.score >= min_score
        ]

    res = _evaluate(recall)
    res["mode"] = "after (LangGraph Store native search)"
    with open(AFTER_FILE, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nhit@1={res['hit@1']} hit@3={res['hit@3']} → 已存 {AFTER_FILE.name}")


# ============================================================
# 对照
# ============================================================
def run_compare():
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(AFTER_FILE, "r", encoding="utf-8") as f:
        after = json.load(f)

    print(f"{'查询':<28} {'迁移前 top3':<16} {'迁移后 top3':<16} 变化")
    print("-" * 80)
    regressed = []
    for b, a in zip(base["per_query"], after["per_query"]):
        change = "一致" if b["recalled_idx_top3"] == a["recalled_idx_top3"] else "有差异"
        if b["hit@3"] and not a["hit@3"]:
            change = "❌ 退化"
            regressed.append(b["query"])
        elif not b["hit@3"] and a["hit@3"]:
            change = "✅ 改善"
        print(f"{b['query'][:26]:<28} {str(b['recalled_idx_top3']):<16} "
              f"{str(a['recalled_idx_top3']):<16} {change}")

    print("-" * 80)
    print(f"hit@1: {base['hit@1']} → {after['hit@1']}")
    print(f"hit@3: {base['hit@3']} → {after['hit@3']}")
    if regressed:
        print(f"\n❌ {len(regressed)} 条查询召回退化: {regressed}")
    else:
        print("\n✅ 召回不退化")
    return not regressed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline", action="store_true")
    g.add_argument("--after", action="store_true")
    g.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.baseline:
        run_baseline()
    elif args.after:
        run_after()
    else:
        run_compare()
