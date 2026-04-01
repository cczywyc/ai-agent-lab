"""
main.py — 入口文件
==================
三种运行模式：
  python main.py                    → 交互式对话
  python main.py -q "你的问题"      → 单次查询
  python main.py --test             → 运行测试套件
"""
import sys
import json

from agent import run_agent, run_agent_with_trace


# ===================================================================
# 模式 1：交互式对话
# ===================================================================

def interactive_mode():
    """交互式对话模式 — 像聊天一样和 Agent 交互"""
    print("=" * 60)
    print("🤖 搜索助手已启动")
    print("=" * 60)
    print("命令:")
    print("  输入问题  → 搜索并回答")
    print("  trace     → 查看上一次的完整 trace")
    print("  quit      → 退出")
    print("=" * 60)

    last_trace = None

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "trace":
            if last_trace:
                # 打印精简版 trace（不含超长的 result_preview）
                summary = {
                    "timestamp": last_trace["timestamp"],
                    "user_message": last_trace["user_message"],
                    "total_turns": len(last_trace["turns"]),
                    "total_tool_calls": last_trace["total_tool_calls"],
                    "errors": last_trace["errors"],
                    "finished": last_trace["finished"],
                    "turns": [
                        {
                            "turn": t["turn"],
                            "finish_reason": t.get("finish_reason"),
                            "tool_calls": [
                                {"tool": tc["tool"], "args": tc["args"], "has_error": tc["has_error"]}
                                for tc in t["tool_calls"]
                            ],
                            "usage": t.get("usage")
                        }
                        for t in last_trace["turns"]
                    ]
                }
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                print("还没有 trace 记录，先提一个问题吧。")
            continue

        last_trace = run_agent_with_trace(user_input)


# ===================================================================
# 模式 2：单次查询
# ===================================================================

def single_query_mode(query: str):
    """单次查询模式 — 适合脚本调用"""
    trace = run_agent_with_trace(query)
    return trace


# ===================================================================
# 模式 3：测试套件
# ===================================================================

TEST_CASES = [
    # --- 正常场景：搜索即可回答 ---
    {
        "query": "2024年诺贝尔物理学奖颁给了谁？",
        "expected_behavior": "搜索 → 直接从 snippet 总结",
        "needs_tools": True
    },
    # --- 正常场景：可能需要抓取网页 ---
    {
        "query": "MCP 协议是什么？它在 AI Agent 中有什么作用？",
        "expected_behavior": "搜索 → 可能需要 fetch_webpage 获取详细内容",
        "needs_tools": True
    },
    # --- 对比类问题：可能多次搜索 ---
    {
        "query": "对比一下 LangGraph 和 CrewAI 的主要区别",
        "expected_behavior": "搜索 → 可能多次搜索或抓取不同页面",
        "needs_tools": True
    },
    # --- 英文问题 ---
    {
        "query": "What are the new features in Qwen3?",
        "expected_behavior": "英文搜索 → 英文回答",
        "needs_tools": True
    },
    # --- 中文问题要求中文回答 ---
    {
        "query": "用中文解释一下什么是 RAG（检索增强生成）",
        "expected_behavior": "可能英文搜索 → 中文回答",
        "needs_tools": True
    },
    # --- 不需要工具的场景 ---
    {
        "query": "帮我写一首关于春天的五言绝句",
        "expected_behavior": "不调用工具 → 直接回答",
        "needs_tools": False
    },
    # --- 搜索不到的场景 ---
    {
        "query": "xyzabc123 这个完全不存在的东西是什么？",
        "expected_behavior": "搜索无结果或无意义结果 → 诚实告知",
        "needs_tools": True
    },
]


def run_test_suite():
    """运行测试套件并输出汇总报告"""
    print("🧪 开始运行测试套件")
    print(f"   共 {len(TEST_CASES)} 个测试用例\n")

    results = []

    for i, case in enumerate(TEST_CASES):
        print(f"\n{'#'*60}")
        print(f"# 测试 {i+1}/{len(TEST_CASES)}")
        print(f"# 问题: {case['query']}")
        print(f"# 预期: {case['expected_behavior']}")
        print(f"{'#'*60}")

        trace = run_agent_with_trace(case["query"])

        # 分析结果
        actually_used_tools = trace["total_tool_calls"] > 0
        tool_match = actually_used_tools == case["needs_tools"]

        results.append({
            "index": i + 1,
            "query": case["query"],
            "expected": case["expected_behavior"],
            "needs_tools": case["needs_tools"],
            "actually_used_tools": actually_used_tools,
            "tool_match": tool_match,
            "total_turns": len(trace["turns"]),
            "total_tool_calls": trace["total_tool_calls"],
            "errors": len(trace["errors"]),
            "has_answer": trace["final_answer"] is not None,
            "answer_length": len(trace["final_answer"]) if trace["final_answer"] else 0
        })

    # 打印汇总报告
    print(f"\n\n{'='*70}")
    print("📊 测试汇总报告")
    print(f"{'='*70}")
    print(f"{'#':>3} {'状态':>4} {'工具匹配':>8} {'轮次':>4} {'调用':>4} {'错误':>4} {'回答长度':>8}  问题")
    print(f"{'-'*70}")

    passed = 0
    for r in results:
        status = "✅" if r["has_answer"] and r["tool_match"] else "⚠️"
        if r["has_answer"] and r["tool_match"]:
            passed += 1
        match = "✓" if r["tool_match"] else "✗"
        print(
            f"{r['index']:>3} {status:>4} {match:>8} "
            f"{r['total_turns']:>4} {r['total_tool_calls']:>4} {r['errors']:>4} "
            f"{r['answer_length']:>8}  {r['query'][:35]}"
        )

    print(f"{'-'*70}")
    print(f"通过: {passed}/{len(results)}")

    # 保存详细结果到文件
    report_path = "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存到 {report_path}")

    return results


# ===================================================================
# 入口
# ===================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_test_suite()
        elif sys.argv[1] == "-q" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            single_query_mode(query)
        else:
            print("用法:")
            print("  python main.py              交互式对话")
            print("  python main.py -q '问题'    单次查询")
            print("  python main.py --test       运行测试套件")
    else:
        interactive_mode()