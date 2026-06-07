"""
搜索 Agent v4.1 入口 — LangGraph 状态化工作流（长期记忆走 Store）

运行模式（与 v3.0 对齐）：
  python main.py --ingest            # 扫描项目内 *.md 重建本地向量库
  python main.py --ingest --dry-run  # 只切块不嵌入
  python main.py                     # 交互模式（记忆默认开）
  python main.py --review            # 交互模式 + human_review 审批（决策 F 开关）
  python main.py --query "你的问题"    # 单次查询
  python main.py --test              # 批量测试（无记忆、interrupt 关，保证可复现）
  python main.py --test --rag-only   # 只跑 RAG 类用例

v3.0 → v4.0 变化：
  - run_agent 的 317 行 for 循环 → graph.invoke（控制流进图）
  - AgentTrace → 吸收进 state（--test 报告从最终 state 读取）
  - 多轮对话靠 checkpointer 的 thread_id 续上（InMemorySaver，进程内）
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
# 测试用例 — 沿用 v3.0 全部 8 例（对照可比）
# ============================================================

TEST_CASES = [
    # === 联网搜索类（验证旧路径未坏） ===
    {"id": 1, "query": "2024年诺贝尔物理学奖颁给了谁？",
     "expect_search": True, "expect_retrieve": False, "category": "factual_time"},
    {"id": 2, "query": "写一首关于春天的五言绝句",
     "expect_search": False, "expect_retrieve": False, "category": "creative"},
    {"id": 3, "query": "帮我算一下 234 × 567",
     "expect_search": False, "expect_retrieve": False, "category": "math"},

    # === RAG 类 ===
    {"id": 4, "query": "我们第三周的 Agent Loop 是怎么处理工具连续失败的？",
     "expect_search": False, "expect_retrieve": True, "category": "rag_internal",
     "note": "应触发 retrieve_documents，回答带 [doc#section] 引用"},
    {"id": 5, "query": "在我们的设计里，should_have_searched 用了哪些规则？",
     "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},
    {"id": 6, "query": "本项目第四周的 RAG 是怎么决定 chunking 策略的？",
     "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},
    {"id": 7, "query": "我之前的笔记里，URL 黑名单是怎么用的？",
     "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},

    # === 混合（本地没覆盖应走联网） ===
    # expect_retrieve=None：该维度不判（v4.2 判据重审——06-05 复跑两次失败方向相反：
    # 一次只走本地、一次先本地再联网。边界用例"两种走法都算对"，必判的是联网兜底发生）。
    # legacy_expect_retrieve 保留旧口径供 passed_legacy 对照。
    {"id": 8, "query": "MCP 协议是什么？",
     "expect_search": True, "expect_retrieve": None, "legacy_expect_retrieve": False,
     "category": "tech_concept",
     "note": "本地未覆盖，必须联网兜底；先试本地检索不做要求"},
]


# ============================================================
# 用例判定（v4.2 从 run_test 抽出为纯函数，test_criteria.py 单测）
# ============================================================

def judge_case(case: dict, state: dict) -> dict:
    """
    单用例判定。v4.2 判据重审（06-05 复跑两项发现的落地）：
      - has_answer 用 config.is_placeholder_answer——占位符名单收口一处，
        [模型返回空回答] 不再算有效回答（Case 5 假阳性修复）
      - expect_search / expect_retrieve 允许 None = 该维度不判
        （Case 8 边界用例"两种走法都算对"的编码）
    可比性：同时产出 passed_legacy = 06-04 报告口径（占位符只排除
    [达到最大轮次]/[错误]，维度全严判；None 维度回退 legacy_expect_*）。
    """
    actual_search = state.get("has_searched", False)
    actual_retrieve = state.get("has_retrieved", False)
    answer = state.get("answer", "")

    def dim_ok(expected, actual):
        return True if expected is None else expected == actual

    search_correct = dim_ok(case["expect_search"], actual_search)
    retrieve_correct = dim_ok(case["expect_retrieve"], actual_retrieve)
    has_answer = not is_placeholder_answer(answer)

    # —— 旧口径（与 v3.0 / 06-04 报告对照用，不参与 PASS 判定） ——
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
    """
    单用例的完整报告行 = judge_case 判定 + state 可观测字段。
    抽出为纯函数与 judge_case 同款可单测（test_criteria.py C5）。
    empty_retries：agent 节点空回答重试计数——量化"重试救回率"用。
    """
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
# 图调用封装
# ============================================================

def invoke_graph(graph, query: str, thread_id: str, *,
                 use_memory: bool = False, on_interrupt=None) -> tuple[dict, int]:
    """
    跑一个问题到结束（含 interrupt 续跑），返回 (最终 state, 耗时 ms)。
    on_interrupt(payload) → resume 决定；为 None 时自动 approve（容错）。
    """
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


def state_summary(state: dict, duration_ms: int) -> str:
    """v3.0 trace.summary() 的 state 版。"""
    from langchain_core.messages import AIMessage, ToolMessage
    from nodes import _window_messages

    id_to_name, tools_used = {}, []
    for m in _window_messages(state.get("messages", [])):
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                id_to_name[tc["id"]] = tc["name"]
        elif isinstance(m, ToolMessage):
            try:
                ok = not (json.loads(m.content) or {}).get("error", False)
            except (json.JSONDecodeError, TypeError):
                ok = True
            name = id_to_name.get(m.tool_call_id, "unknown")
            tools_used.append(f"{name}({'✓' if ok else '✗'})")
    tools_str = " → ".join(tools_used) if tools_used else "无工具调用"

    flags = []
    if state.get("search_correction_injected"):
        flags.append("搜索纠正")
    if state.get("retrieval_correction_injected"):
        flags.append("检索纠正")
    if state.get("fallback_triggered"):
        flags.append("降级")
    flags_str = f" [{','.join(flags)}]" if flags else ""
    return f"[{state.get('turn_count', 0)}轮 | {duration_ms}ms] {tools_str}{flags_str}"


def cli_review(payload: dict) -> str:
    """交互模式的 interrupt 处理：回车通过，输入文本改写。"""
    print("\n┌─ human_review ─ 待审批草稿 ─────────────")
    print(f"│ {payload.get('draft_answer', '')[:500]}")
    print("└─────────────────────────────────────────")
    try:
        decision = input("回车通过，或输入改写后的答案: ").strip()
    except (EOFError, KeyboardInterrupt):
        decision = ""
    return decision or "approve"


# ============================================================
# Ingest 模式（v3.0 原样）
# ============================================================

def run_ingest(dry_run: bool = False):
    from rag.ingest import ingest_all
    from rag.retriever import get_retriever

    print("=" * 60)
    print(f"  v4.0 — Ingest 模式 {'(dry-run)' if dry_run else ''}")
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
# 交互模式
# ============================================================

def run_interactive(use_memory: bool = True, review: bool = False):
    if review:
        config.INTERRUPT_ENABLED = True

    graph = build_graph()
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print(f"  搜索 Agent v4.0 (LangGraph) — 交互模式"
          f"（{'记忆已加载' if use_memory else '无记忆'}"
          f"{'，human_review 开' if review else ''}）")
    print("  输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("  特殊命令：/memory 查看记忆 / /reset 清空记忆 / /state 看本轮 state 摘要")
    print("=" * 60)

    memory, store = None, None
    if use_memory:
        from memory import get_memory, get_ltm_store
        memory = get_memory()
        store = get_ltm_store()  # 与 build_graph 共用同一单例
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
        if user_input == "/state":
            if last_state is None:
                print("[还没有跑过问题]")
            else:
                keys = ("turn_count", "has_searched", "has_retrieved",
                        "correction_triggered", "fallback_triggered", "assembly_report")
                print(json.dumps({k: last_state.get(k) for k in keys},
                                 ensure_ascii=False, indent=2, default=str))
            continue

        print()
        last_state, last_ms = invoke_graph(
            graph, user_input, thread_id,
            use_memory=use_memory, on_interrupt=cli_review,
        )
        print(f"助手: {last_state['answer']}")
        print(f"\n  📊 {state_summary(last_state, last_ms)}\n")


# ============================================================
# 单次查询
# ============================================================

def run_single(query: str, show_state: bool = False, use_memory: bool = False):
    graph = build_graph()
    print(f"问题: {query}\n")
    state, ms = invoke_graph(graph, query, f"single-{uuid.uuid4().hex[:8]}",
                             use_memory=use_memory, on_interrupt=cli_review)
    print(f"回答: {state['answer']}")
    print(f"\n📊 {state_summary(state, ms)}")
    if show_state:
        printable = {k: v for k, v in state.items() if k != "messages"}
        print("\n--- 最终 State（messages 略） ---")
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))


# ============================================================
# 记忆多轮 demo（v3.0 用例原样，同一 thread 体现 checkpointer）
# ============================================================

MEMORY_DEMO_TURNS = [
    "我们第三周的 Agent Loop 是怎么处理工具连续失败的？",
    "请记住：以后回答涉及本地文档时，先列结论再列引用。",
    "刚才那个连续失败机制里，consecutive_errors 是怎么重置的？",
    "另外简单说一下，本项目的 RAG chunking 是按什么切的？",
]


def run_memory_demo():
    from memory import get_memory, get_ltm_store
    memory = get_memory()
    store = get_ltm_store()
    memory.reset(store)

    graph = build_graph(store=store)
    thread_id = "memory-demo"

    print("=" * 60)
    print("  v4.0 — 记忆系统多轮 demo（4 轮，同一 thread）")
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
# 测试模式（用例与判据沿 v3.0，数据源从 trace 改为最终 state）
# ============================================================

def run_test(rag_only: bool = False):
    cases = [c for c in TEST_CASES if (not rag_only or c.get("expect_retrieve"))]

    print("=" * 60)
    print(f"  v4.0 — 测试模式 ({len(cases)} 用例{'，仅 RAG' if rag_only else ''}"
          f"；无记忆、interrupt 关)")
    print("=" * 60)

    graph = build_graph()
    report = {
        "version": "4.2",  # v4.2：判据重审（占位符名单收口 + None=不判维度），见 judge_case
        "engine": "langgraph",
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(cases),
        "results": [],
        "summary": {},
    }

    def fmt_expect(v):
        return "any" if v is None else v

    passed = passed_legacy = 0
    for case in cases:
        cid, query = case["id"], case["query"]

        print(f"--- Case {cid}: {query}")
        print(f"    预期: search={fmt_expect(case['expect_search'])}, "
              f"retrieve={fmt_expect(case['expect_retrieve'])}")

        # 每个用例独立 thread：测试隔离（per-case 状态互不串扰）
        state, ms = invoke_graph(graph, query, f"test-{cid}", use_memory=False)

        row = build_case_record(case, state, ms)
        passed += row["passed"]
        passed_legacy += row["passed_legacy"]

        print(f"    实际: search={row['actual_search']}, "
              f"retrieve={row['actual_retrieve']}")
        print(f"    {state_summary(state, ms)}")
        print(f"    回答片段: {row['answer_preview'][:120]}")
        print(f"    结果: {'✅ PASS' if row['passed'] else '❌ FAIL'}\n")

        report["results"].append(row)

    total = len(cases)
    report["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)" if total else "N/A",
        # 旧口径通过数（与 v3.0 / 06-04 报告可比；不参与本版 PASS 判定）
        "passed_legacy": passed_legacy,
        "pass_rate_legacy": (f"{passed_legacy}/{total} ({100 * passed_legacy / total:.0f}%)"
                             if total else "N/A"),
        "search_correction_count": sum(
            1 for r in report["results"] if r["search_correction_injected"]),
        "retrieval_correction_count": sum(
            1 for r in report["results"] if r["retrieval_correction_injected"]),
        "fallback_count": sum(1 for r in report["results"] if r["fallback_triggered"]),
        # 空回答重试总数（agent 节点内，T9）——重试救回率 = 有重试但 has_answer 的占比
        "empty_retry_count": sum(r["empty_retries"] for r in report["results"]),
        "avg_turns": sum(r["total_turns"] for r in report["results"]) / total if total else 0,
        "avg_duration_ms": sum(r["duration_ms"] for r in report["results"]) / total if total else 0,
    }

    print("=" * 60)
    print("  测试总结")
    print("=" * 60)
    s = report["summary"]
    print(f"  通过率:         {s['pass_rate']}（旧口径对照: {s['pass_rate_legacy']}）")
    print(f"  联网纠正触发:   {s['search_correction_count']} 次")
    print(f"  检索纠正触发:   {s['retrieval_correction_count']} 次")
    print(f"  降级触发:       {s['fallback_count']} 次")
    print(f"  空回答重试:     {s['empty_retry_count']} 次")
    print(f"  平均轮次:       {s['avg_turns']:.1f}")
    print(f"  平均耗时:       {s['avg_duration_ms']:.0f}ms")

    failed_cases = [r for r in report["results"] if not r["passed"]]
    if failed_cases:
        print("\n  失败用例:")
        for r in failed_cases:
            reasons = []
            if not r["search_correct"]:
                reasons.append(f"search 预期{r['expect_search']}/实际{r['actual_search']}")
            if not r["retrieve_correct"]:
                reasons.append(f"retrieve 预期{r['expect_retrieve']}/实际{r['actual_retrieve']}")
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
    parser = argparse.ArgumentParser(description="搜索 Agent v4.0 (LangGraph)")
    parser.add_argument("--query", "-q", type=str, help="单次查询")
    parser.add_argument("--test", "-t", action="store_true", help="测试模式（无记忆）")
    parser.add_argument("--rag-only", action="store_true", help="测试模式：仅 RAG 用例")
    parser.add_argument("--ingest", action="store_true", help="重建本地向量库")
    parser.add_argument("--dry-run", action="store_true", help="ingest 时只切块不嵌入")
    parser.add_argument("--state", action="store_true", help="单次查询时显示最终 state")
    parser.add_argument("--review", action="store_true",
                        help="交互模式开启 human_review 审批（决策 F）")
    # 记忆系统
    parser.add_argument("--memory-demo", action="store_true", help="跑 4 轮记忆 demo")
    parser.add_argument("--memory-info", action="store_true", help="打印当前记忆状态后退出")
    parser.add_argument("--reset-memory", action="store_true", help="清空记忆并退出")
    parser.add_argument("--no-memory", action="store_true", help="交互/单查时禁用记忆")
    parser.add_argument("--with-memory", action="store_true",
                        help="单次查询时启用记忆（默认禁用）")

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
        run_test(rag_only=args.rag_only)
    elif args.query:
        run_single(args.query, show_state=args.state, use_memory=args.with_memory)
    else:
        run_interactive(use_memory=not args.no_memory, review=args.review)


if __name__ == "__main__":
    main()
