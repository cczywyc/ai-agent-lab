"""
v4.0 图结构测试 — 桩模型 + 桩工具，不调真实 API，离线可复现。

测试的是"图的接线"而不是模型质量：
  T1 直答路径（无纠正、无工具）
  T2 检索纠正 cycle + retrieved_chunks 累加（决策 B/D）
  T3 联网纠正 cycle（决策 D）
  T4 纠正只注入一次（防护限制复刻）
  T5 连续失败 ≥2 触发降级，且只降级一次
  T6 turn_count 闸门收口（E4 语义进真实图）
  T7 interrupt 开关：关→透明；开→暂停/通过/改写（决策 F）
  T8 同 thread 多问题：messages 持久化 + per-query 重置 + 装配窗口切片（决策 C / E2 语义）

跑法：../../.venv/bin/python test_graph.py
"""

import json
import sys
from types import SimpleNamespace

from langgraph.types import Command

import config
import nodes
from config import (
    MAX_TURNS,
    RECURSION_LIMIT,
    RETRIEVAL_CORRECTION_MESSAGE,
    CORRECTION_MESSAGE,
    FALLBACK_MESSAGE,
)
from graph import build_graph


# ============================================================
# 桩：OpenAI 响应 + 脚本化模型 + 工具
# ============================================================

def resp_stop(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop",
    )])


def resp_tools(*calls):
    """calls: (tool_name, args_dict) 列表 → 一条 tool_calls 响应。"""
    tcs = [
        SimpleNamespace(
            id=f"call_{i}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
        )
        for i, (name, args) in enumerate(calls)
    ]
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=tcs),
        finish_reason="tool_calls",
    )])


class ScriptedModel:
    """按脚本依次吐响应；记录每次收到的消息窗口（用于断言切片）。"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []  # 每次 call_model 收到的 openai 格式消息列表

    def __call__(self, oai_messages):
        self.calls.append(list(oai_messages))
        if not self.steps:
            raise AssertionError("StubModel: 脚本已耗尽，图多调了一次模型")
        return self.steps.pop(0)


def make_execute_tool(results_by_name):
    """桩 execute_tool：按工具名返回固定结果；记录调用。"""
    log = []

    def _exec(tool_name, tool_args):
        log.append((tool_name, tool_args))
        result = results_by_name.get(tool_name, {"ok": True})
        return result(log) if callable(result) else result

    _exec.log = log
    return _exec


def run(graph, query, thread_id, use_memory=False):
    cfg = {
        "configurable": {"thread_id": thread_id, "use_memory": use_memory},
        "recursion_limit": RECURSION_LIMIT,
    }
    result = graph.invoke({"user_message": query}, cfg)
    return result, cfg


CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"  {'✓ PASS' if cond else '✗ FAIL'}  {name}")
    return bool(cond)


# ============================================================
# T1 直答路径
# ============================================================

def t1():
    print("\n[T1] 直答路径（创意问题，无纠正、无工具）")
    model = ScriptedModel([resp_stop("春眠不觉晓")])
    nodes.call_model = model
    g = build_graph()
    final, _ = run(g, "写一首关于春天的五言绝句", "t1")

    check("answer = 模型 stop 内容", final["answer"] == "春眠不觉晓")
    check("turn_count = 1", final["turn_count"] == 1)
    check("无纠正/降级", not final["correction_triggered"] and not final["fallback_triggered"])
    check("messages = [system, user, assistant]",
          [m.type for m in final["messages"]] == ["system", "human", "ai"])
    check("发给模型的窗口以 system 开头", model.calls[0][0]["role"] == "system")


# ============================================================
# T2 检索纠正 cycle + chunks 累加
# ============================================================

def t2():
    print("\n[T2] 检索纠正 cycle + retrieved_chunks 累加")
    q = "我们第三周的 Agent Loop 是怎么处理工具连续失败的？"
    model = ScriptedModel([
        resp_stop("凭记忆直答（应被纠正）"),
        resp_tools(("retrieve_documents", {"query": q})),
        resp_stop("基于 [Agent_Loop_设计笔记#连续失败] 的回答"),
    ])
    nodes.call_model = model
    nodes.execute_tool = make_execute_tool({
        "retrieve_documents": {"results": [
            {"doc": "Agent_Loop_设计笔记", "section": "连续失败", "chunk_id": 7, "score": 0.88},
        ]},
    })
    g = build_graph()
    final, _ = run(g, q, "t2")

    check("检索纠正已注入", final["retrieval_correction_injected"])
    check("correction_triggered 标注", final["correction_triggered"])
    check("纠正消息原文进入第 2 次模型窗口",
          model.calls[1][-1] == {"role": "user", "content": RETRIEVAL_CORRECTION_MESSAGE})
    check("has_retrieved = True", final["has_retrieved"])
    check("retrieved_chunks 累加 1 条且含元信息",
          final["retrieved_chunks"] == [
              {"doc": "Agent_Loop_设计笔记", "section": "连续失败", "chunk_id": 7, "score": 0.88}])
    check("最终 answer 来自第 3 次 stop", final["answer"].startswith("基于 [Agent_Loop_设计笔记"))
    check("共 3 个 turn", final["turn_count"] == 3)


# ============================================================
# T3 联网纠正 cycle
# ============================================================

def t3():
    print("\n[T3] 联网纠正 cycle")
    q = "2024年诺贝尔物理学奖颁给了谁？"
    model = ScriptedModel([
        resp_stop("凭记忆直答（应被纠正）"),
        resp_tools(("web_search", {"query": "2024 Nobel Prize Physics"})),
        resp_stop("Hopfield 和 Hinton（来源：nobelprize.org）"),
    ])
    nodes.call_model = model
    nodes.execute_tool = make_execute_tool({
        "web_search": {"results": [{"title": "Nobel 2024", "url": "https://nobelprize.org"}]},
    })
    g = build_graph()
    final, _ = run(g, q, "t3")

    check("联网纠正已注入", final["search_correction_injected"])
    check("纠正消息原文进入第 2 次模型窗口",
          model.calls[1][-1] == {"role": "user", "content": CORRECTION_MESSAGE})
    check("has_searched = True", final["has_searched"])
    check("检索纠正未触发（优先级互斥）", not final["retrieval_correction_injected"])
    check("最终 answer 来自第 3 次 stop", final["answer"].startswith("Hopfield"))


# ============================================================
# T4 纠正只注入一次
# ============================================================

def t4():
    print("\n[T4] 纠正只注入一次（模型坚持直答时尊重第二次 stop）")
    q = "我们第三周的 Agent Loop 是怎么处理工具连续失败的？"
    model = ScriptedModel([
        resp_stop("第一次直答"),
        resp_stop("第二次仍直答（应被接受）"),
    ])
    nodes.call_model = model
    g = build_graph()
    final, _ = run(g, q, "t4")

    check("检索纠正恰好注入一次", final["retrieval_correction_injected"])
    check("第二次 stop 被接受为答案", final["answer"] == "第二次仍直答（应被接受）")
    check("联网纠正未连带触发", not final["search_correction_injected"])
    check("共 2 个 turn", final["turn_count"] == 2)


# ============================================================
# T5 连续失败降级
# ============================================================

def t5():
    print("\n[T5] fetch_webpage 连续失败 ≥2 → 降级注入一次")
    q = "请帮我读取 example.com 网页的内容"
    model = ScriptedModel([
        resp_tools(("fetch_webpage", {"url": "https://example.com/a"})),
        resp_tools(("fetch_webpage", {"url": "https://example.com/b"})),
        resp_stop("以下回答基于搜索摘要，未能获取完整文章内容。……"),
    ])
    nodes.call_model = model
    nodes.execute_tool = make_execute_tool({
        "fetch_webpage": {"error": True, "error_type": "HTTPError", "message": "403"},
    })
    g = build_graph()
    final, _ = run(g, q, "t5")

    check("降级已注入", final["fallback_injected"] and final["fallback_triggered"])
    check("降级消息原文进入第 3 次模型窗口",
          model.calls[2][-1] == {"role": "user", "content": FALLBACK_MESSAGE})
    check("consecutive_failures 计到 2", final["consecutive_failures"] == 2)
    check("最终给出降级回答", final["answer"].startswith("以下回答基于搜索摘要"))


# ============================================================
# T6 turn_count 闸门
# ============================================================

def t6():
    print(f"\n[T6] turn_count 闸门（模型永远要工具，应停在 {MAX_TURNS} 轮）")
    q = "一个让模型停不下来的问题"
    model = ScriptedModel([
        resp_tools(("web_search", {"query": f"step {i}"})) for i in range(MAX_TURNS)
    ])
    nodes.call_model = model
    nodes.execute_tool = make_execute_tool({"web_search": {"results": []}})
    g = build_graph()
    final, _ = run(g, q, "t6")  # 不该抛 GraphRecursionError

    check(f"恰好调用模型 {MAX_TURNS} 次后收口", final["turn_count"] == MAX_TURNS and not model.steps)
    check("answer 为达到最大轮次提示", final["answer"].startswith("[达到最大轮次]"))


# ============================================================
# T7 interrupt 开关
# ============================================================

def t7():
    print("\n[T7] human_review interrupt：关→透明；开→暂停/通过/改写")
    q = "写一首关于秋天的诗"

    # --- 关：完全透明 ---
    config.INTERRUPT_ENABLED = False
    nodes.call_model = ScriptedModel([resp_stop("秋风起兮白云飞")])
    g = build_graph()
    final, _ = run(g, q, "t7-off")
    check("OFF：无 __interrupt__", "__interrupt__" not in final)
    check("OFF：直达 answer", final["answer"] == "秋风起兮白云飞")

    # --- 开：暂停，approve 放行 ---
    config.INTERRUPT_ENABLED = True
    nodes.call_model = ScriptedModel([resp_stop("秋风起兮白云飞")])
    g = build_graph()
    mid, cfg = run(g, q, "t7-on")
    paused_at = g.get_state(cfg).next
    check("ON：出现 __interrupt__", "__interrupt__" in mid)
    check("ON：暂停在 human_review", paused_at == ("human_review",))
    final = g.invoke(Command(resume="approve"), cfg)
    check("ON：approve 后跑完且答案保留", final["answer"] == "秋风起兮白云飞")

    # --- 开：resume 改写答案 ---
    nodes.call_model = ScriptedModel([resp_stop("草稿答案")])
    g2 = build_graph()
    mid, cfg2 = run(g2, q, "t7-rewrite")
    final = g2.invoke(Command(resume="人工改写后的答案"), cfg2)
    check("ON：resume 文本改写最终答案", final["answer"] == "人工改写后的答案")

    config.INTERRUPT_ENABLED = False  # 恢复默认


# ============================================================
# T8 同 thread 多问题：持久化 + 重置 + 窗口切片
# ============================================================

def t8():
    print("\n[T8] 同 thread 两问题：messages 持久化、per-query 重置、窗口切片")
    model = ScriptedModel([resp_stop("答案一"), resp_stop("答案二")])
    nodes.call_model = model
    g = build_graph()
    run(g, "写一首关于冬天的诗", "t8")
    final, cfg = run(g, "再写一首关于夏天的诗", "t8")

    check("第二问 answer 正确", final["answer"] == "答案二")
    check("第二问 turn_count 被 init 重置后 = 1", final["turn_count"] == 1)
    check("messages 跨问题持久化（2×[sys,user,ai] = 6 条）", len(final["messages"]) == 6)
    check("第二问模型窗口只含本问装配（切片自最后一条 system）",
          len(model.calls[1]) == 2 and model.calls[1][0]["role"] == "system"
          and model.calls[1][1]["content"] == "再写一首关于夏天的诗")


# ============================================================
# 入口
# ============================================================

def main():
    print("=== v4.0 图结构测试（桩模型，离线） ===")
    for t in (t1, t2, t3, t4, t5, t6, t7, t8):
        t()

    passed = sum(1 for _, ok in CHECKS if ok)
    total = len(CHECKS)
    print(f"\n{'=' * 40}\n结果: {passed}/{total} 项判据通过", end="")
    if passed == total:
        print(" —— 全部通过 ✓")
        sys.exit(0)
    print("\n失败项:")
    for name, ok in CHECKS:
        if not ok:
            print(f"  ✗ {name}")
    sys.exit(1)


if __name__ == "__main__":
    main()
