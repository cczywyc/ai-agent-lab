"""
技术写作 Agent v6.0 入口 — supervisor 多 Agent（LangGraph 状态化工作流）

v6.0 第八周：在 v5.0 外循环上把角色节点升格成 supervisor 协调的独立 workers。
  输入一个技术写作主题 → supervisor 拆研究子任务 → researcher 逐个检索+压缩 findings →
  writer 把 findings 写成初稿 → reviewer 审稿出 verdict → （reject 则带 review_notes 返修，
  最多 MAX_REVIEW 次，达上限取 best-so-far）→ finalize 交付被评审的稿 → human_review → update_memory。

  诚实一笔（《什么时候该用多 Agent》反例）：researcher→writer→reviewer 是强顺序依赖流水线，
  supervisor 是"串接专家"不是并行 fan-out——本周买到的是"角色专精 + 干净的 handoff 契约"，
  不是"真并行更强"。选 supervisor（拓扑 B）是为学机制；纯交付报告其实拓扑 A（一张图）更省。

运行模式：
  python main.py --ingest             # 扫描项目内 *.md 重建本地向量库（含 week_8 docs）
  python main.py --ingest --dry-run   # 只切块不嵌入
  python main.py                      # 交互技术写作模式（记忆默认开）
  python main.py --review             # 交互 + human_review 审批（决策 F）
  python main.py --query "写作主题"     # 单次技术写作 → 交付稿
  python main.py --query "..." --state # 附带打印最终 state（findings / draft / review_* 等）
  python main.py --test               # 批量技术写作测试（无记忆、interrupt 关，可复现观测多 Agent 外循环）

可观测（多 Agent 外循环）：研究子任务数 / 各子任务 status / findings 数 / review_count（打回）/
  replan_count（研究 skip）/ review_verdict / best_score / termination_reason。

判据测试见 test_graph.py（v6.0 离线桩，M1–M9，48 项）与 evals.py（routing_accuracy 种子集，E1 沉淀）。
"""

import argparse
import json
import logging
import time
import uuid
from datetime import datetime

from langgraph.types import Command

import config
from config import (
    RECURSION_LIMIT,
    MAX_REVIEW,
    PLACEHOLDER_MAX_TURNS,
    PLACEHOLDER_ERROR,
    is_placeholder_answer,
)
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 调研测试用例（v5.0：每条是一个调研问题，验外循环端到端）
# ============================================================

WRITING_TEST_CASES = [
    {"id": 1, "category": "rag_internal",
     "query": "写一篇技术综述：我们项目从第三周到第六周，Agent Loop 的控制流是怎么演进的？"},
    {"id": 2, "category": "rag_internal",
     "query": "写一篇对比短文：我们项目里 RAG 的 chunking 策略和检索纠正（correction）机制各自怎么设计、有何取舍？"},
    {"id": 3, "category": "mixed",
     "query": "写一篇技术说明：多 Agent 的 supervisor 与 swarm 拓扑各自适合什么场景，结合我们第八周的设计草稿。"},
]


# ============================================================
# v4.2 判据纯函数（保留：test_criteria.py 单测；executor 单步路径口径不变）
# ============================================================

def judge_case(case: dict, state: dict) -> dict:
    """单步搜索/检索路径判定（v4.2 口径，纯函数；test_criteria.py 单测保留）。v6.0 整体写作判定见 judge_writing。"""
    actual_search = state.get("has_searched", False)
    actual_retrieve = state.get("has_retrieved", False)
    answer = state.get("answer", "")

    def dim_ok(expected, actual):
        return True if expected is None else expected == actual

    search_correct = dim_ok(case["expect_search"], actual_search)
    retrieve_correct = dim_ok(case["expect_retrieve"], actual_retrieve)
    has_answer = not is_placeholder_answer(answer)

    legacy_search = case.get("legacy_expect_search", case["expect_search"])
    legacy_retrieve = case.get("legacy_expect_retrieve", case["expect_retrieve"])
    has_answer_legacy = (bool(answer)
                         and not answer.startswith(PLACEHOLDER_MAX_TURNS)
                         and not answer.startswith(PLACEHOLDER_ERROR))

    return {
        "actual_search": actual_search,
        "actual_retrieve": actual_retrieve,
        "search_correct": search_correct,
        "retrieve_correct": retrieve_correct,
        "has_answer": has_answer,
        "passed": search_correct and retrieve_correct and has_answer,
        "passed_legacy": (dim_ok(legacy_search, actual_search)
                          and dim_ok(legacy_retrieve, actual_retrieve)
                          and has_answer_legacy),
    }


def build_case_record(case: dict, state: dict, duration_ms: int) -> dict:
    """v4.2 单步报告行（保留：test_criteria.py 单测）。"""
    answer = state.get("answer", "")
    return {
        "id": case["id"],
        "query": case["query"],
        "category": case["category"],
        "expect_search": case["expect_search"],
        "expect_retrieve": case["expect_retrieve"],
        **judge_case(case, state),
        "correction_triggered": state.get("correction_triggered", False),
        "retrieval_correction_injected": state.get("retrieval_correction_injected", False),
        "search_correction_injected": state.get("search_correction_injected", False),
        "fallback_triggered": state.get("fallback_triggered", False),
        "empty_retries": state.get("empty_retries", 0),
        "total_turns": state.get("turn_count", 0),
        "duration_ms": duration_ms,
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "answer_preview": answer[:300],
    }


# ============================================================
# v6.0 技术写作判定（多 Agent 外循环口径）
# ============================================================

def judge_writing(state: dict) -> dict:
    """技术写作任务整体判定：supervisor 拆了研究子任务、researcher 回了 findings、
    writer 出了 draft、reviewer 给了 verdict、且最终交付非占位。"""
    plan = state.get("plan", [])
    findings = state.get("findings", [])
    draft = state.get("draft", "")
    delivery = state.get("answer", "")
    ok_findings = sum(1 for f in findings if f.get("status") == "ok")
    has_delivery = not is_placeholder_answer(delivery)
    return {
        "n_subtasks": len(plan),
        "n_findings": len(findings),
        "ok_findings": ok_findings,
        "has_draft": bool(draft),
        "review_verdict": state.get("review_verdict", ""),
        "review_count": state.get("review_count", 0),
        "replan_count": state.get("replan_count", 0),
        "best_score": (state.get("best_draft", {}) or {}).get("score"),
        "termination_reason": state.get("termination_reason", ""),
        "has_delivery": has_delivery,
        # 通过 = 拆了研究子任务 + 至少一条有效 finding + 写出了稿 + 交付非占位
        "passed": bool(plan) and ok_findings >= 1 and bool(draft) and has_delivery,
    }


# ============================================================
# 图调用封装
# ============================================================

def invoke_graph(graph, query: str, thread_id: str, *,
                 use_memory: bool = False, on_interrupt=None) -> tuple[dict, int]:
    cfg = {
        "configurable": {"thread_id": thread_id, "use_memory": use_memory},
        "recursion_limit": RECURSION_LIMIT,
    }
    start = time.time()
    result = graph.invoke({"user_message": query}, cfg)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decision = on_interrupt(payload) if on_interrupt else "approve"
        result = graph.invoke(Command(resume=decision), cfg)
    duration_ms = int((time.time() - start) * 1000)
    return result, duration_ms


def plan_summary(state: dict) -> str:
    """研究子任务一览（supervisor 拆 + 各子任务 status）。"""
    plan = state.get("plan", [])
    if not plan:
        return "（无研究计划）"
    lines = []
    for s in plan:
        mark = {"done": "✓", "skipped": "⤬", "pending": "·"}.get(s.get("status"), "·")
        lines.append(f"  {mark} {s.get('id', 0) + 1}. {s.get('query', '')}")
    return "\n".join(lines)


def state_summary(state: dict, duration_ms: int) -> str:
    """多 Agent 外循环可观测一行。"""
    plan = state.get("plan", [])
    n_done = sum(1 for s in plan if s.get("status") == "done")
    findings = state.get("findings", [])
    best = state.get("best_draft", {}) or {}
    bits = [
        f"{len(plan)}研究子任务(done {n_done})",
        f"findings {len(findings)}",
        f"review {state.get('review_count', 0)}/{MAX_REVIEW}",
        f"replan {state.get('replan_count', 0)}",
        f"verdict={state.get('review_verdict', '') or '—'}",
        f"best={best.get('score')}",
        f"{duration_ms}ms",
    ]
    return f"[{' | '.join(bits)}] 终止={state.get('termination_reason', '') or 'done'}"


def cli_review(payload: dict) -> str:
    print("\n┌─ human_review ─ 待审批调研报告 ─────────────")
    print(f"│ {payload.get('draft_answer', '')[:600]}")
    print("└─────────────────────────────────────────")
    try:
        decision = input("回车通过，或输入改写后的报告: ").strip()
    except (EOFError, KeyboardInterrupt):
        decision = ""
    return decision or "approve"


# ============================================================
# Ingest 模式（v3.0 原样，ingest 范围已含 week_7 docs）
# ============================================================

def run_ingest(dry_run: bool = False):
    from rag.ingest import ingest_all
    from rag.retriever import get_retriever

    print("=" * 60)
    print(f"  v5.0 — Ingest 模式 {'(dry-run)' if dry_run else ''}")
    print("=" * 60)

    chunks = ingest_all()
    print(f"\n→ Got {len(chunks)} chunks across files.")
    if not chunks:
        print("[!] 没切出任何 chunk，请检查 DEFAULT_INGEST_DIRS")
        return

    print("\n--- 抽样预览 ---")
    for c in chunks[:3]:
        preview = c["text"][:200].replace("\n", " ")
        print(f"  [{c['doc']} | {c['section']} | id={c['chunk_id']}]\n  {preview}...\n")

    if dry_run:
        print("\n[dry-run] 跳过 embedding 与持久化。")
        return

    retriever = get_retriever(namespace="docs", autoload=False)
    retriever.rebuild_from_chunks(chunks, show_progress=True)
    print(f"\n✅ Vector store built: {retriever.store.info()}")


# ============================================================
# 交互调研模式
# ============================================================

def run_interactive(use_memory: bool = True, review: bool = False):
    if review:
        config.INTERRUPT_ENABLED = True

    graph = build_graph()
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print(f"  技术调研 Agent v5.0 (planner-executor-critic) — 交互模式"
          f"（{'记忆已加载' if use_memory else '无记忆'}"
          f"{'，human_review 开' if review else ''}）")
    print("  输入调研问题开始，输入 'quit' / 'exit' 退出")
    print("  特殊命令：/memory 查看记忆 / /reset 清空记忆 / /plan 看上轮研究计划 / /state 看 state 摘要")
    print("=" * 60)

    memory, store = None, None
    if use_memory:
        from memory import get_memory, get_ltm_store
        memory = get_memory()
        store = get_ltm_store()
        info = memory.info(store)
        print(f"  已加载: {info['short_term_turns']} 轮 | {len(info['preferences'])} 偏好 | "
              f"{info['facts']} 条事实 | {len(info['topics_top5'])} 个高频主题")
    print()

    last_state, last_ms = None, 0
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
        if user_input == "/memory" and memory is not None:
            print(json.dumps(memory.info(store), ensure_ascii=False, indent=2, default=str))
            continue
        if user_input == "/reset" and memory is not None:
            memory.reset(store)
            print("[已清空记忆]")
            continue
        if user_input == "/plan":
            print(plan_summary(last_state or {}))
            continue
        if user_input == "/state":
            if last_state is None:
                print("[还没有跑过问题]")
            else:
                keys = ("plan", "step_index", "replan_count", "plan_version",
                        "done", "termination_reason", "empty_retries")
                print(json.dumps({k: last_state.get(k) for k in keys},
                                 ensure_ascii=False, indent=2, default=str))
            continue

        print()
        last_state, last_ms = invoke_graph(
            graph, user_input, thread_id,
            use_memory=use_memory, on_interrupt=cli_review,
        )
        print("研究计划:")
        print(plan_summary(last_state))
        print(f"\n{last_state['answer']}")
        print(f"\n  📊 {state_summary(last_state, last_ms)}\n")


# ============================================================
# 单次调研
# ============================================================

def run_single(query: str, show_state: bool = False, use_memory: bool = False):
    graph = build_graph()
    print(f"调研问题: {query}\n")
    state, ms = invoke_graph(graph, query, f"single-{uuid.uuid4().hex[:8]}",
                             use_memory=use_memory, on_interrupt=cli_review)
    print("研究计划:")
    print(plan_summary(state))
    print(f"\n{'=' * 60}\n{state['answer']}\n{'=' * 60}")
    print(f"\n📊 {state_summary(state, ms)}")
    if show_state:
        printable = {k: v for k, v in state.items() if k != "messages"}
        print("\n--- 最终 State（messages 略） ---")
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))


# ============================================================
# 记忆多轮 demo（v4.0 用例原样，同一 thread 体现 checkpointer）
# ============================================================

MEMORY_DEMO_TURNS = [
    "调研我们第三周 Agent Loop 是怎么处理工具连续失败的",
    "请记住：以后回答涉及本地文档时，先列结论再列引用。",
    "刚才那个连续失败机制里，consecutive_errors 是怎么重置的？",
    "另外简单调研一下，本项目的 RAG chunking 是按什么切的？",
]


def run_memory_demo():
    from memory import get_memory, get_ltm_store
    memory = get_memory()
    store = get_ltm_store()
    memory.reset(store)

    graph = build_graph(store=store)
    thread_id = "memory-demo"

    print("=" * 60)
    print("  v5.0 — 记忆系统多轮 demo（4 轮，同一 thread）")
    print("=" * 60)

    for i, q in enumerate(MEMORY_DEMO_TURNS, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"你: {q}")
        state, ms = invoke_graph(graph, q, thread_id, use_memory=True)
        answer = state["answer"]
        print(f"助手: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print(f"  📊 {state_summary(state, ms)}")

    print("\n" + "=" * 30)
    print("  Demo 后的记忆状态")
    print("=" * 30)
    print(json.dumps(memory.info(store), ensure_ascii=False, indent=2, default=str))


# ============================================================
# 测试模式（v6.0：批量技术写作，验多 Agent 外循环端到端 + 可观测）
# ============================================================

def run_test():
    cases = WRITING_TEST_CASES

    print("=" * 60)
    print(f"  v6.0 — 技术写作测试模式 ({len(cases)} 用例；无记忆、interrupt 关)")
    print("=" * 60)

    graph = build_graph()
    report = {
        "version": "6.0",
        "engine": "langgraph-supervisor-multiagent",
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(cases),
        "results": [],
        "summary": {},
    }

    passed = 0
    for case in cases:
        cid, query = case["id"], case["query"]
        print(f"--- Case {cid}: {query}")

        state, ms = invoke_graph(graph, query, f"test-{cid}", use_memory=False)
        j = judge_writing(state)
        passed += j["passed"]

        print("    研究计划（supervisor 拆）:")
        print("    " + plan_summary(state).replace("\n", "\n    "))
        print(f"    {state_summary(state, ms)}")
        print(f"    交付片段: {state.get('answer', '')[:140].strip()}…")
        print(f"    结果: {'✅ PASS' if j['passed'] else '❌ FAIL'}"
              f"（研究子任务 {j['n_subtasks']} / findings {j['n_findings']}(ok {j['ok_findings']}) / "
              f"verdict {j['review_verdict'] or '—'} / review {j['review_count']} / replan {j['replan_count']}）\n")

        report["results"].append({
            "id": cid, "query": query, "category": case["category"],
            "duration_ms": ms, **j, "delivery_preview": state.get("answer", "")[:400],
        })

    total = len(cases)
    report["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)" if total else "N/A",
        "total_subtasks": sum(r["n_subtasks"] for r in report["results"]),
        "total_findings": sum(r["n_findings"] for r in report["results"]),
        "total_review_rounds": sum(r["review_count"] for r in report["results"]),
        "total_replan": sum(r["replan_count"] for r in report["results"]),
        "accepted_deliveries": sum(1 for r in report["results"] if r["review_verdict"] == "accept"),
        "avg_duration_ms": sum(r["duration_ms"] for r in report["results"]) / total if total else 0,
    }

    print("=" * 60)
    print("  测试总结")
    print("=" * 60)
    s = report["summary"]
    print(f"  通过率:         {s['pass_rate']}")
    print(f"  研究子任务总数: {s['total_subtasks']}（findings {s['total_findings']}）")
    print(f"  打回轮次总数:   {s['total_review_rounds']}（reviewer accept 交付 {s['accepted_deliveries']}）")
    print(f"  研究 skip 总数: {s['total_replan']}")
    print(f"  平均耗时:       {s['avg_duration_ms']:.0f}ms")

    report_file = "test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {report_file}")
    return report


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="技术调研 Agent v5.0 (planner-executor-critic)")
    parser.add_argument("--query", "-q", type=str, help="单次调研问题")
    parser.add_argument("--test", "-t", action="store_true", help="批量调研测试（无记忆）")
    parser.add_argument("--ingest", action="store_true", help="重建本地向量库（含 week_7 docs）")
    parser.add_argument("--dry-run", action="store_true", help="ingest 时只切块不嵌入")
    parser.add_argument("--state", action="store_true", help="单次调研时显示最终 state")
    parser.add_argument("--review", action="store_true",
                        help="交互模式开启 human_review 审批（决策 F）")
    parser.add_argument("--memory-demo", action="store_true", help="跑 4 轮记忆 demo")
    parser.add_argument("--memory-info", action="store_true", help="打印当前记忆状态后退出")
    parser.add_argument("--reset-memory", action="store_true", help="清空记忆并退出")
    parser.add_argument("--no-memory", action="store_true", help="交互/单查时禁用记忆")
    parser.add_argument("--with-memory", action="store_true",
                        help="单次调研时启用记忆（默认禁用）")

    args = parser.parse_args()

    if args.reset_memory:
        from memory import get_memory, get_ltm_store
        get_memory().reset(get_ltm_store())
        print("[已清空记忆]")
        return
    if args.memory_info:
        from memory import get_memory, get_ltm_store
        print(json.dumps(get_memory().info(get_ltm_store()),
                         ensure_ascii=False, indent=2, default=str))
        return

    if args.ingest:
        run_ingest(dry_run=args.dry_run)
    elif args.memory_demo:
        run_memory_demo()
    elif args.test:
        run_test()
    elif args.query:
        run_single(args.query, show_state=args.state, use_memory=args.with_memory)
    else:
        run_interactive(use_memory=not args.no_memory, review=args.review)


if __name__ == "__main__":
    main()
