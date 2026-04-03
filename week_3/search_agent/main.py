"""
搜索 Agent 入口

三种运行模式：
  1. 交互模式：python main.py
  2. 单次查询：python main.py --query "你的问题"
  3. 测试模式：python main.py --test

v2.0 变更：
  - 所有模式都输出 trace 摘要
  - 测试模式生成带完整 trace 的 JSON 报告
  - 新增 --format 选项，测试模式下用 json_schema 格式化回答
"""

import argparse
import json
import logging
from datetime import datetime

from agent import run_agent, format_answer_as_json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 测试用例
# ============================================================

TEST_CASES = [
    # === 第二周原始 7 个 case ===
    {
        "id": 1,
        "query": "2024年诺贝尔物理学奖颁给了谁？",
        "expect_search": True,
        "category": "factual_time",
        "note": "v1 失败：模型跳过搜索直接回答",
    },
    {
        "id": 2,
        "query": "MCP 协议是什么？",
        "expect_search": True,
        "category": "tech_concept",
        "note": "v1 通过",
    },
    {
        "id": 3,
        "query": "对比 LangGraph 和 CrewAI 的优缺点",
        "expect_search": True,
        "category": "comparison",
        "note": "v1 失败：连续403无输出",
    },
    {
        "id": 4,
        "query": "What are the new features in Qwen3?",
        "expect_search": True,
        "category": "tech_product",
        "note": "v1 通过",
    },
    {
        "id": 5,
        "query": "用中文解释什么是 RAG",
        "expect_search": True,
        "category": "tech_concept",
        "note": "v1 失败：模型跳过搜索直接回答",
    },
    {
        "id": 6,
        "query": "写一首关于春天的五言绝句",
        "expect_search": False,
        "category": "creative",
        "note": "v1 通过",
    },
    {
        "id": 7,
        "query": "xyzabc123 是什么？",
        "expect_search": True,
        "category": "unknown_term",
        "note": "v1 失败：模型直接说不知道",
    },
    # === 第三周新增边界 case ===
    {
        "id": 8,
        "query": "帮我算一下 234 × 567",
        "expect_search": False,
        "category": "math",
        "note": "v2 新增：数学应直接回答",
    },
    {
        "id": 9,
        "query": "你好，你能做什么？",
        "expect_search": False,
        "category": "chat",
        "note": "v2 新增：闲聊应直接回答",
    },
    {
        "id": 10,
        "query": "Anthropic 最近发布了什么新产品？",
        "expect_search": True,
        "category": "factual_time",
        "note": "v2 新增：时间敏感的公司动态",
    },
]


# ============================================================
# 模式 1：交互模式
# ============================================================

def run_interactive():
    """交互式对话循环"""
    print("=" * 60)
    print("  搜索 Agent v2.0 — 交互模式")
    print("  输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("=" * 60)
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

        print()
        answer, trace = run_agent(user_input)
        print(f"助手: {answer}")
        print(f"\n  📊 {trace.summary()}")
        print()


# ============================================================
# 模式 2：单次查询
# ============================================================

def run_single(query: str, show_trace: bool = False):
    """单次查询"""
    print(f"问题: {query}\n")

    answer, trace = run_agent(query)

    print(f"回答: {answer}")
    print(f"\n📊 {trace.summary()}")

    if show_trace:
        print("\n--- 完整 Trace ---")
        print(trace.to_json())


# ============================================================
# 模式 3：测试模式
# ============================================================

def run_test(use_format: bool = False):
    """批量测试所有用例，生成报告"""
    print("=" * 60)
    print("  搜索 Agent v2.0 — 测试模式")
    print(f"  测试用例: {len(TEST_CASES)} 个")
    print(f"  格式化输出: {'是' if use_format else '否'}")
    print("=" * 60)
    print()

    report = {
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(TEST_CASES),
        "results": [],
        "summary": {},
    }

    passed = 0
    total = len(TEST_CASES)

    for case in TEST_CASES:
        case_id = case["id"]
        query = case["query"]
        expect_search = case["expect_search"]

        print(f"--- Case {case_id}: {query}")
        print(f"    预期搜索: {'是' if expect_search else '否'}")

        answer, trace = run_agent(query)

        # 判断是否通过
        search_correct = (trace.searched == expect_search)
        has_answer = (
                answer is not None
                and not answer.startswith("[达到最大轮次]")
                and not answer.startswith("[错误]")
        )
        case_passed = search_correct and has_answer

        if case_passed:
            passed += 1

        status = "✅ PASS" if case_passed else "❌ FAIL"
        print(f"    实际搜索: {'是' if trace.searched else '否'}")
        print(f"    有回答: {'是' if has_answer else '否'}")
        print(f"    纠正触发: {'是' if trace.correction_triggered else '否'}")
        print(f"    降级触发: {'是' if trace.fallback_triggered else '否'}")
        print(f"    {trace.summary()}")
        print(f"    结果: {status}")

        # 可选：格式化回答
        formatted = None
        if use_format and has_answer:
            try:
                formatted = format_answer_as_json(answer, query, trace)
                print(f"    格式化置信度: {formatted.get('confidence', 'N/A')}")
            except Exception as e:
                print(f"    格式化失败: {e}")

        # 记录到报告
        case_result = {
            "id": case_id,
            "query": query,
            "category": case["category"],
            "expect_search": expect_search,
            "actual_search": trace.searched,
            "search_correct": search_correct,
            "has_answer": has_answer,
            "passed": case_passed,
            "correction_triggered": trace.correction_triggered,
            "fallback_triggered": trace.fallback_triggered,
            "total_turns": trace.total_turns,
            "duration_ms": trace.total_duration_ms,
            "answer_preview": (answer or "")[:200],
            "trace": trace.to_dict(),
            "note": case.get("note", ""),
        }
        if formatted:
            case_result["formatted_answer"] = formatted

        report["results"].append(case_result)
        print()

    # 生成汇总
    report["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)",
        "correction_count": sum(
            1 for r in report["results"] if r["correction_triggered"]
        ),
        "fallback_count": sum(
            1 for r in report["results"] if r["fallback_triggered"]
        ),
        "avg_turns": sum(r["total_turns"] for r in report["results"]) / total,
        "avg_duration_ms": sum(
            r["duration_ms"] for r in report["results"]
        ) / total,
    }

    # 打印总结
    print("=" * 60)
    print("  测试总结")
    print("=" * 60)
    s = report["summary"]
    print(f"  通过率:     {s['pass_rate']}")
    print(f"  纠正触发:   {s['correction_count']} 次")
    print(f"  降级触发:   {s['fallback_count']} 次")
    print(f"  平均轮次:   {s['avg_turns']:.1f}")
    print(f"  平均耗时:   {s['avg_duration_ms']:.0f}ms")
    print()

    # 失败用例清单
    failed_cases = [r for r in report["results"] if not r["passed"]]
    if failed_cases:
        print("  失败用例:")
        for r in failed_cases:
            reason = []
            if not r["search_correct"]:
                reason.append(
                    f"搜索{'未触发' if r['expect_search'] else '误触发'}"
                )
            if not r["has_answer"]:
                reason.append("无有效回答")
            print(f"    Case {r['id']}: {r['query'][:30]}... — {', '.join(reason)}")
        print()

    # 保存报告
    report_file = "test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_file}")

    return report


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="搜索 Agent v2.0")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="单次查询模式：直接提问",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="测试模式：运行所有测试用例",
    )
    parser.add_argument(
        "--format", "-f",
        action="store_true",
        help="测试模式下启用 json_schema 格式化输出",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="单次查询模式下显示完整 trace",
    )

    args = parser.parse_args()

    if args.test:
        run_test(use_format=args.format)
    elif args.query:
        run_single(args.query, show_trace=args.trace)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
