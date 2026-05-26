"""
搜索 Agent v3.0 入口

运行模式：
  python main.py --ingest          # 扫描项目内 *.md 重建本地向量库
  python main.py --ingest --dry-run  # 只切块不嵌入，看看 chunk 切得对不对
  python main.py                   # 交互模式
  python main.py --query "你的问题"   # 单次查询
  python main.py --test            # 批量测试
  python main.py --test --rag-only # 只跑 RAG 类用例
"""

import argparse
import json
import logging
import sys
from datetime import datetime

from agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 测试用例 — v3.0
# 在 v2.0 用例基础上新增 RAG 类
# ============================================================

TEST_CASES = [
    # === 联网搜索类（v2.0 用例，验证旧路径未坏） ===
    {
        "id": 1,
        "query": "2024年诺贝尔物理学奖颁给了谁？",
        "expect_search": True,
        "expect_retrieve": False,
        "category": "factual_time",
    },
    {
        "id": 2,
        "query": "写一首关于春天的五言绝句",
        "expect_search": False,
        "expect_retrieve": False,
        "category": "creative",
    },
    {
        "id": 3,
        "query": "帮我算一下 234 × 567",
        "expect_search": False,
        "expect_retrieve": False,
        "category": "math",
    },

    # === RAG 类（v3.0 新增）===
    {
        "id": 4,
        "query": "我们第三周的 Agent Loop 是怎么处理工具连续失败的？",
        "expect_search": False,
        "expect_retrieve": True,
        "category": "rag_internal",
        "note": "应触发 retrieve_documents，回答带 [doc#section] 引用",
    },
    {
        "id": 5,
        "query": "在我们的设计里，should_have_searched 用了哪些规则？",
        "expect_search": False,
        "expect_retrieve": True,
        "category": "rag_internal",
    },
    {
        "id": 6,
        "query": "本项目第四周的 RAG 是怎么决定 chunking 策略的？",
        "expect_search": False,
        "expect_retrieve": True,
        "category": "rag_internal",
    },
    {
        "id": 7,
        "query": "我之前的笔记里，URL 黑名单是怎么用的？",
        "expect_search": False,
        "expect_retrieve": True,
        "category": "rag_internal",
    },

    # === 混合（本地没覆盖应走联网） ===
    {
        "id": 8,
        "query": "MCP 协议是什么？",
        "expect_search": True,
        "expect_retrieve": False,  # 笔记里没专门写 MCP，应该走联网
        "category": "tech_concept",
        "note": "本地未覆盖，走联网兜底",
    },
]


def _rag_filter(case: dict) -> bool:
    return case.get("expect_retrieve", False)


# ============================================================
# Ingest 模式
# ============================================================

def run_ingest(dry_run: bool = False):
    """扫描项目内 markdown 建库。"""
    from rag.ingest import ingest_all
    from rag.retriever import get_retriever

    print("=" * 60)
    print(f"  v3.0 — Ingest 模式 {'(dry-run)' if dry_run else ''}")
    print("=" * 60)

    chunks = ingest_all()
    print(f"\n→ Got {len(chunks)} chunks across files.")

    if not chunks:
        print("[!] 没切出任何 chunk，请检查 DEFAULT_INGEST_DIRS")
        return

    # 预览前 3 个 chunk
    print("\n--- 抽样预览 ---")
    for c in chunks[:3]:
        preview = c["text"][:200].replace("\n", " ")
        print(
            f"  [{c['doc']} | {c['section']} | id={c['chunk_id']}]\n"
            f"  {preview}...\n"
        )

    if dry_run:
        print("\n[dry-run] 跳过 embedding 与持久化。")
        return

    # 调用 embedding 并入库
    retriever = get_retriever(namespace="docs", autoload=False)
    retriever.rebuild_from_chunks(chunks, show_progress=True)

    info = retriever.store.info()
    print(f"\n✅ Vector store built: {info}")


# ============================================================
# 交互模式
# ============================================================

def run_interactive(use_memory: bool = True):
    from memory import get_memory

    memory = get_memory() if use_memory else None

    print("=" * 60)
    print(f"  搜索 Agent v3.0 — 交互模式（{'记忆已加载' if use_memory else '无记忆'}）")
    print("  输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("  特殊命令：/memory 查看记忆 / /reset 清空记忆")
    print("=" * 60)
    if memory is not None:
        info = memory.info()
        print(
            f"  已加载: {info['short_term_turns']} 轮 | "
            f"{len(info['preferences'])} 偏好 | "
            f"{info['facts']} 条事实 | "
            f"{len(info['topics_top5'])} 个高频主题"
        )
    print()

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 调试命令
        if user_input == "/memory" and memory is not None:
            import json as _json
            print(_json.dumps(memory.info(), ensure_ascii=False, indent=2))
            continue
        if user_input == "/reset" and memory is not None:
            memory.reset()
            print("[已清空记忆]")
            continue

        print()
        answer, trace = run_agent(user_input, memory=memory)
        print(f"助手: {answer}")
        print(f"\n  📊 {trace.summary()}")
        print()


# ============================================================
# 单次查询
# ============================================================

def run_single(query: str, show_trace: bool = False, use_memory: bool = False):
    from memory import get_memory

    memory = get_memory() if use_memory else None
    print(f"问题: {query}\n")
    answer, trace = run_agent(query, memory=memory)
    print(f"回答: {answer}")
    print(f"\n📊 {trace.summary()}")
    if show_trace:
        print("\n--- 完整 Trace ---")
        print(trace.to_json())


# ============================================================
# 记忆多轮 demo
# ============================================================

MEMORY_DEMO_TURNS = [
    "我们第三周的 Agent Loop 是怎么处理工具连续失败的？",
    "请记住：以后回答涉及本地文档时，先列结论再列引用。",
    "刚才那个连续失败机制里，consecutive_errors 是怎么重置的？",
    "另外简单说一下，本项目的 RAG chunking 是按什么切的？",
]


def run_memory_demo():
    """跑一个 4 轮模拟对话，演示装配、偏好、事实召回。"""
    from memory import get_memory
    memory = get_memory()
    memory.reset()  # 从干净状态开始

    print("=" * 60)
    print("  v3.0 — 记忆系统多轮 demo（4 轮）")
    print("=" * 60)

    for i, q in enumerate(MEMORY_DEMO_TURNS, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"你: {q}")
        answer, trace = run_agent(q, memory=memory)
        print(f"助手: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print(f"  📊 {trace.summary()}")

    print("\n=" * 30)
    print("  Demo 后的记忆状态")
    print("=" * 30)
    info = memory.info()
    import json as _json
    print(_json.dumps(info, ensure_ascii=False, indent=2))


# ============================================================
# 测试模式
# ============================================================

def run_test(rag_only: bool = False):
    cases = [c for c in TEST_CASES if (not rag_only or _rag_filter(c))]

    print("=" * 60)
    print(f"  v3.0 — 测试模式 ({len(cases)} 用例{'，仅 RAG' if rag_only else ''})")
    print("=" * 60)

    report = {
        "version": "3.0",
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(cases),
        "results": [],
        "summary": {},
    }

    passed = 0
    for case in cases:
        cid = case["id"]
        query = case["query"]
        exp_search = case["expect_search"]
        exp_retrieve = case["expect_retrieve"]

        print(f"--- Case {cid}: {query}")
        print(f"    预期: search={exp_search}, retrieve={exp_retrieve}")

        answer, trace = run_agent(query)

        search_correct = (trace.searched == exp_search)
        retrieve_correct = (trace.retrieved == exp_retrieve)
        has_answer = (
            answer is not None
            and not answer.startswith("[达到最大轮次]")
            and not answer.startswith("[错误]")
        )
        case_passed = search_correct and retrieve_correct and has_answer

        if case_passed:
            passed += 1

        status = "✅ PASS" if case_passed else "❌ FAIL"
        print(f"    实际: search={trace.searched}, retrieve={trace.retrieved}")
        print(f"    {trace.summary()}")
        print(f"    回答片段: {(answer or '')[:120]}")
        print(f"    结果: {status}\n")

        report["results"].append({
            "id": cid,
            "query": query,
            "category": case["category"],
            "expect_search": exp_search,
            "expect_retrieve": exp_retrieve,
            "actual_search": trace.searched,
            "actual_retrieve": trace.retrieved,
            "search_correct": search_correct,
            "retrieve_correct": retrieve_correct,
            "has_answer": has_answer,
            "passed": case_passed,
            "correction_triggered": trace.correction_triggered,
            "retrieval_correction_triggered": trace.retrieval_correction_triggered,
            "fallback_triggered": trace.fallback_triggered,
            "total_turns": trace.total_turns,
            "duration_ms": trace.total_duration_ms,
            "answer_preview": (answer or "")[:300],
            "trace": trace.to_dict(),
        })

    total = len(cases)
    report["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)" if total else "N/A",
        "search_correction_count": sum(
            1 for r in report["results"] if r["correction_triggered"]
        ),
        "retrieval_correction_count": sum(
            1 for r in report["results"] if r["retrieval_correction_triggered"]
        ),
        "fallback_count": sum(
            1 for r in report["results"] if r["fallback_triggered"]
        ),
        "avg_turns": sum(r["total_turns"] for r in report["results"]) / total if total else 0,
        "avg_duration_ms": sum(r["duration_ms"] for r in report["results"]) / total if total else 0,
    }

    print("=" * 60)
    print("  测试总结")
    print("=" * 60)
    s = report["summary"]
    print(f"  通过率:         {s['pass_rate']}")
    print(f"  联网纠正触发:   {s['search_correction_count']} 次")
    print(f"  检索纠正触发:   {s['retrieval_correction_count']} 次")
    print(f"  降级触发:       {s['fallback_count']} 次")
    print(f"  平均轮次:       {s['avg_turns']:.1f}")
    print(f"  平均耗时:       {s['avg_duration_ms']:.0f}ms")

    failed_cases = [r for r in report["results"] if not r["passed"]]
    if failed_cases:
        print("\n  失败用例:")
        for r in failed_cases:
            reasons = []
            if not r["search_correct"]:
                reasons.append(
                    f"search 预期{r['expect_search']}/实际{r['actual_search']}"
                )
            if not r["retrieve_correct"]:
                reasons.append(
                    f"retrieve 预期{r['expect_retrieve']}/实际{r['actual_retrieve']}"
                )
            if not r["has_answer"]:
                reasons.append("无有效回答")
            print(f"    Case {r['id']}: {r['query'][:30]}... — {'; '.join(reasons)}")

    report_file = "test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {report_file}")

    return report


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="搜索 Agent v3.0")
    parser.add_argument("--query", "-q", type=str, help="单次查询")
    parser.add_argument("--test", "-t", action="store_true", help="测试模式（无记忆）")
    parser.add_argument("--rag-only", action="store_true", help="测试模式：仅 RAG 用例")
    parser.add_argument("--ingest", action="store_true", help="重建本地向量库")
    parser.add_argument("--dry-run", action="store_true", help="ingest 时只切块不嵌入")
    parser.add_argument("--trace", action="store_true", help="单次查询时显示完整 trace")
    # 记忆系统
    parser.add_argument("--memory-demo", action="store_true", help="跑 4 轮记忆 demo")
    parser.add_argument("--memory-info", action="store_true", help="打印当前记忆状态后退出")
    parser.add_argument("--reset-memory", action="store_true", help="清空记忆并退出")
    parser.add_argument("--no-memory", action="store_true", help="交互/单查时禁用记忆")
    parser.add_argument("--with-memory", action="store_true",
                        help="单次查询时启用记忆（默认禁用）")

    args = parser.parse_args()

    # --- 记忆运维命令 ---
    if args.reset_memory:
        from memory import get_memory
        get_memory().reset()
        print("[已清空记忆]")
        return
    if args.memory_info:
        from memory import get_memory
        import json as _json
        print(_json.dumps(get_memory().info(), ensure_ascii=False, indent=2))
        return

    if args.ingest:
        run_ingest(dry_run=args.dry_run)
    elif args.memory_demo:
        run_memory_demo()
    elif args.test:
        run_test(rag_only=args.rag_only)
    elif args.query:
        run_single(args.query, show_trace=args.trace, use_memory=args.with_memory)
    else:
        run_interactive(use_memory=not args.no_memory)


if __name__ == "__main__":
    main()
