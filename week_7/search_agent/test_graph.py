"""
v5.0 图结构测试 — 桩模型 + 桩工具，不调真实 API，离线可复现。

测的是"外循环的接线"而不是模型质量。新增的外循环机制各自有桩探针 E1–E7 单独验过
（week_7/experiment/），这里把同样的机制放进**真实 v5.0 图**里端到端复跑一遍：

  W1  happy path：planner 拆解 → 两子任务执行 → critic accept → 推进 → 结构化报告
  W2  route_after_critic 三路（E1 进真实图）：accept→planner / retry<上限→重做该步 / escalate→planner
  W3  两层重置并存（E2/E6 进真实图）：per-subtask 标志（has_retrieved/turn_count/retry_count）
      每子任务清零；per-task step_results 跨子任务累加；replan_count 不随 step 清零
  W4  step_results 节点内手动累加 + init 跨问题清空（E3 进真实图）
  W5  外循环双闸门（E4 进真实图）：critic 恒 escalate → 停在 replan_count，不撞 recursion_limit
  W6  两档计数器互不串扰（E5 进真实图）：empty_retries（传输层）vs retry_count（业务层）
  W7  跨子任务 messages 重锚（E7 进真实图）：executor 窗口逐子任务隔离、不串台前序 tool 历史
  W8  executor 引擎原样复用：子任务内 tool 调用 / 检索纠正 / 降级（v4.2 内层循环保真）
  W9  内层 turn_count 闸门收口到 critic（不再是 finalize）
  W10 收尾链尾保真：结构化报告 → human_review → update_memory（审批框永不空）

跑法：../../.venv/bin/python test_graph.py
"""

import json
import sys
from types import SimpleNamespace

from langgraph.types import Command

import config
import nodes
from config import MAX_TURNS, MAX_STEPS, MAX_REPLAN, MAX_RETRY, RECURSION_LIMIT
from graph import build_graph
from memory.ltm_store import make_inmemory_store


def build_test_graph():
    """离线图：注入 stub embed 的 InMemoryStore（不碰真实 ltm.db / sqlite 扩展）。"""
    return build_graph(store=make_inmemory_store(embed_fn=lambda ts: [[0.0] * 3 for _ in ts], dims=3))


# ============================================================
# 桩：executor 模型（call_model）+ 工具
# ============================================================

def resp_stop(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop",
    )])


def resp_tools(*calls):
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
    """executor 模型桩：按脚本依次吐响应；记录每次收到的窗口（断言切片/重锚）。"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, oai_messages):
        self.calls.append(list(oai_messages))
        if not self.steps:
            raise AssertionError("ScriptedModel: 脚本已耗尽，executor 多调了一次模型")
        return self.steps.pop(0)


class InfiniteToolModel:
    """永远要工具的 executor 桩（用于 turn_count 闸门压力测试，不会被脚本耗尽）。"""

    def __init__(self, tool="web_search"):
        self.tool = tool
        self.calls = []

    def __call__(self, oai_messages):
        self.calls.append(list(oai_messages))
        return resp_tools((self.tool, {"query": "loop"}))


class ScriptedPlanner:
    """planner 模型桩：每次 decompose/re-plan 返回一段计划文本（JSON 子任务数组）。
    脚本耗尽时重复最后一条（便于 re-plan 压力测试无限取用）。"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = 0
        self._last = steps[-1] if steps else "[]"

    def __call__(self, messages):
        self.calls += 1
        if len(self.steps) > 1:
            self._last = self.steps.pop(0)
        elif self.steps:
            self._last = self.steps[0]
        return self._last


class ScriptedCritic:
    """critic 模型桩：按脚本依次裁决（accept/retry/escalate 文本）；耗尽重复最后一条。"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = 0
        self._last = steps[-1] if steps else "accept"

    def __call__(self, messages):
        self.calls += 1
        if len(self.steps) > 1:
            self._last = self.steps.pop(0)
        elif self.steps:
            self._last = self.steps[0]
        return self._last


def plan_json(*queries):
    """planner 桩返回的计划文本：子任务 query 的 JSON 数组。"""
    return json.dumps(list(queries), ensure_ascii=False)


def make_execute_tool(results_by_name):
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


def set_stubs(*, executor=None, planner=None, critic=None, tools=None):
    """统一注入桩——executor/planner/critic 三档模型 + 工具。"""
    if executor is not None:
        nodes.call_model = executor
    if planner is not None:
        nodes.call_planner_model = planner
    if critic is not None:
        nodes.call_critic_model = critic
    if tools is not None:
        nodes.execute_tool = tools


# ============================================================
# W1 happy path：拆解 → 两子任务 → accept → 报告
# ============================================================

def w1():
    print("\n[W1] happy path：planner 拆解两子任务 → 各 accept → 结构化报告")
    q = "调研一下我们项目的 Agent Loop 与 RAG 设计"
    set_stubs(
        planner=ScriptedPlanner([plan_json("Agent Loop 怎么设计的", "RAG chunking 策略")]),
        executor=ScriptedModel([
            resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("子任务1结论"),
            resp_tools(("retrieve_documents", {"query": "y"})), resp_stop("子任务2结论"),
        ]),
        critic=ScriptedCritic(["accept"]),
        tools=make_execute_tool({"retrieve_documents": {"results": [
            {"doc": "d", "section": "s", "chunk_id": 1, "score": 0.9}]}}),
    )
    final, _ = run(build_test_graph(), q, "w1")

    check("plan 拆出 2 个子任务", len(final["plan"]) == 2)
    check("两子任务都标 done", all(s.get("status") == "done" for s in final["plan"]))
    check("step_results 累加 2 条", len(final["step_results"]) == 2)
    check("step_results 各步结论正确",
          [r["text"] for r in sorted(final["step_results"], key=lambda r: r["step_id"])]
          == ["子任务1结论", "子任务2结论"])
    check("done=True 且终止原因为 all_steps_done",
          final["done"] and final["termination_reason"] == "all_steps_done")
    check("最终报告非占位符且含两步结论",
          "子任务1结论" in final["answer"] and "子任务2结论" in final["answer"])
    check("plan_version=1（无 re-plan）", final["plan_version"] == 1)


# ============================================================
# W2 route_after_critic 三路（E1 进真实图）
# ============================================================

def w2():
    print("\n[W2] route_after_critic 三路：accept→推进 / retry→重做该步 / escalate→planner re-plan")

    # --- accept：单子任务一次过 ---
    set_stubs(
        planner=ScriptedPlanner([plan_json("唯一子任务")]),
        executor=ScriptedModel([resp_stop("一次过的结论")]),
        critic=ScriptedCritic(["accept"]),
    )
    final, _ = run(build_test_graph(), "q-accept", "w2-accept")
    check("accept：直接推进收口，答案含结论", "一次过的结论" in final["answer"])

    # --- retry：critic 先 retry 再 accept，executor 重做该步一次 ---
    set_stubs(
        planner=ScriptedPlanner([plan_json("要重做的子任务")]),
        executor=ScriptedModel([resp_stop("初稿(被打回)"), resp_stop("重做后的结论")]),
        critic=ScriptedCritic(["retry", "accept"]),
    )
    final, _ = run(build_test_graph(), "q-retry", "w2-retry")
    check("retry：executor 重做该步后产出新结论", "重做后的结论" in final["answer"])
    check("retry：retry_count 记到 1（业务层）", final["retry_count"] == 1)

    # --- escalate：critic escalate s0 → skip-and-advance → s1 照常 accept ---
    set_stubs(
        planner=ScriptedPlanner([plan_json("会被跳过的子任务", "正常子任务")]),
        executor=ScriptedModel([resp_stop("方向不对的结论"), resp_stop("正常子任务结论")]),
        critic=ScriptedCritic(["escalate", "accept"]),
    )
    final, _ = run(build_test_graph(), "q-escalate", "w2-escalate")
    check("escalate：当前步标记 skipped", final["plan"][0]["status"] == "skipped")
    check("escalate：skip-and-advance，replan_count=1", final["replan_count"] == 1)
    check("escalate：后续子任务照常执行并 accept", final["plan"][1]["status"] == "done")
    check("escalate：报告含后续子任务结论（不卡死在失败步）", "正常子任务结论" in final["answer"])


# ============================================================
# W3 两层重置并存（E2/E6 进真实图）
# ============================================================

def w3():
    print("\n[W3] 两层重置：per-subtask 标志每子任务清零；per-task 跨子任务累加")
    set_stubs(
        planner=ScriptedPlanner([plan_json("子任务A", "子任务B")]),
        executor=ScriptedModel([
            resp_tools(("retrieve_documents", {"query": "A"})), resp_stop("A 结论"),
            resp_tools(("retrieve_documents", {"query": "B"})), resp_stop("B 结论"),
        ]),
        critic=ScriptedCritic(["accept"]),
        tools=make_execute_tool({"retrieve_documents": {"results": [
            {"doc": "d", "section": "s", "chunk_id": 1, "score": 0.9}]}}),
    )
    final, _ = run(build_test_graph(), "q-reset", "w3")

    check("step_results 跨两子任务累加 2 条", len(final["step_results"]) == 2)
    check("turn_count 是单子任务内计数（≤ MAX_TURNS）", final["turn_count"] <= MAX_TURNS)
    check("retry_count 每子任务独立、happy path 为 0", final["retry_count"] == 0)
    check("has_retrieved 末子任务为 True（自己检索过）", final["has_retrieved"])
    check("retrieved_chunks per-subtask（末子任务仅 1 条、不累加前序）",
          len(final["retrieved_chunks"]) == 1)


# ============================================================
# W4 step_results 手动累加 + init 跨问题清空（E3 进真实图）
# ============================================================

def w4():
    print("\n[W4] step_results 跨问题清空：同一 thread 第二个问题不带上一问题的结论")
    g = build_test_graph()
    set_stubs(
        planner=ScriptedPlanner([plan_json("问题1子任务")]),
        executor=ScriptedModel([resp_stop("问题1结论")]),
        critic=ScriptedCritic(["accept"]),
    )
    run(g, "问题一", "w4")
    set_stubs(
        planner=ScriptedPlanner([plan_json("问题2子任务")]),
        executor=ScriptedModel([resp_stop("问题2结论")]),
        critic=ScriptedCritic(["accept"]),
    )
    final, _ = run(g, "问题二", "w4")
    check("第二问 step_results 只含本问题（1 条）", len(final["step_results"]) == 1)
    check("第二问不残留问题1的结论",
          all("问题1" not in r["text"] for r in final["step_results"]))
    check("第二问 plan_version 被 init 重置后重新计起", final["plan_version"] == 1)


# ============================================================
# W5 外循环双闸门（E4 进真实图）
# ============================================================

def w5():
    print(f"\n[W5] 双闸门：critic 恒 escalate → skip-and-advance 烧到 replan_count={MAX_REPLAN} 提前收口，不撞 recursion_limit")
    # 需 ≥ MAX_REPLAN+1 个子任务：每步 escalate→skip(replan+1)，达 MAX_REPLAN 即放弃剩余
    set_stubs(
        planner=ScriptedPlanner([plan_json("过不了的子任务1", "过不了的子任务2", "过不了的子任务3")]),
        executor=ScriptedModel([resp_stop("总是被否的结论")] * 50),
        critic=ScriptedCritic(["escalate"]),
    )
    final, _ = run(build_test_graph(), "q-loop", "w5")  # 不该抛 GraphRecursionError
    check(f"replan_count 闸门收口（达 {MAX_REPLAN}）", final["replan_count"] >= MAX_REPLAN)
    check("终止原因为 max_replan（太多步失败 → 放弃剩余）", final["termination_reason"] == "max_replan")
    check("剩余步未跑（第 3 步仍 pending）", final["plan"][2]["status"] == "pending")
    check("仍产出报告（优雅收口，非抛异常）", isinstance(final.get("answer"), str) and bool(final["answer"]))


# ============================================================
# W6 两档计数器互不串扰（E5 进真实图）
# ============================================================

def w6():
    print("\n[W6] 两档计数器：empty_retries（传输层）vs retry_count（业务层）互不吃额度")
    set_stubs(
        planner=ScriptedPlanner([plan_json("子任务")]),
        executor=ScriptedModel([
            resp_stop(""), resp_stop("初稿"),       # 第 1 次执行：空→重试→正文
            resp_stop(""), resp_stop("重做稿"),     # 业务 retry 后再执行：空→重试→正文
        ]),
        critic=ScriptedCritic(["retry", "accept"]),
    )
    final, _ = run(build_test_graph(), "q-counters", "w6")
    check("empty_retries（传输层）= 2", final["empty_retries"] == 2)
    check("retry_count（业务层）= 1", final["retry_count"] == 1)
    check("空回答重试未吃业务额度（retry_count ≤ MAX_RETRY）", final["retry_count"] <= MAX_RETRY)
    check("最终拿到重做稿", "重做稿" in final["answer"])


# ============================================================
# W7 跨子任务 messages 重锚（E7 进真实图）
# ============================================================

def w7():
    print("\n[W7] 跨子任务重锚：executor 窗口逐子任务隔离、不串台前序 tool 历史")
    set_stubs(
        planner=ScriptedPlanner([plan_json("子任务A", "子任务B")]),
        executor=ScriptedModel([
            resp_tools(("web_search", {"query": "A"})), resp_stop("A 结论"),
            resp_tools(("web_search", {"query": "B"})), resp_stop("B 结论"),
        ]),
        critic=ScriptedCritic(["accept"]),
        tools=make_execute_tool({"web_search": {"results": [{"title": "t", "url": "u"}]}}),
    )
    final, _ = run(build_test_graph(), "q-reanchor", "w7")
    calls = nodes.call_model.calls
    second_subtask_first_window = calls[2]
    check("第二子任务窗口以 system 锚开头", second_subtask_first_window[0]["role"] == "system")
    bled = any(m.get("role") == "tool" for m in second_subtask_first_window)
    check("第二子任务首窗口不串台子任务A 的 tool 结果", not bled)
    check("messages 跨子任务只增不减（远多于单窗口）",
          len(final["messages"]) > len(second_subtask_first_window))


# ============================================================
# W8 executor 引擎原样复用（v4.2 内层循环保真）
# ============================================================

def w8():
    print("\n[W8] executor 引擎：子任务内 tool 调用 + 检索纠正 + 降级（v4.2 内层保真）")

    # --- 子任务内：先凭记忆直答被检索纠正，再检索后产出 ---
    set_stubs(
        planner=ScriptedPlanner([plan_json("我们第三周的 Agent Loop 是怎么处理工具连续失败的")]),
        executor=ScriptedModel([
            resp_stop("凭记忆直答(应被纠正)"),
            resp_tools(("retrieve_documents", {"query": "连续失败"})),
            resp_stop("基于 [doc#连续失败] 的结论"),
        ]),
        critic=ScriptedCritic(["accept"]),
        tools=make_execute_tool({"retrieve_documents": {"results": [
            {"doc": "doc", "section": "连续失败", "chunk_id": 7, "score": 0.88}]}}),
    )
    final, _ = run(build_test_graph(), "q-correct", "w8-correct")
    check("子任务内检索纠正已注入", final["retrieval_correction_injected"])
    check("子任务内 has_retrieved=True", final["has_retrieved"])
    check("最终结论来自检索后产出", "基于 [doc#连续失败]" in final["answer"])

    # --- 子任务内：fetch 连续失败 ≥2 → 降级注入一次 ---
    set_stubs(
        planner=ScriptedPlanner([plan_json("读取网页内容")]),
        executor=ScriptedModel([
            resp_tools(("fetch_webpage", {"url": "https://x/a"})),
            resp_tools(("fetch_webpage", {"url": "https://x/b"})),
            resp_stop("以下回答基于搜索摘要…"),
        ]),
        critic=ScriptedCritic(["accept"]),
        tools=make_execute_tool({"fetch_webpage": {"error": True, "error_type": "HTTPError", "message": "403"}}),
    )
    final, _ = run(build_test_graph(), "q-fallback", "w8-fallback")
    check("子任务内降级已注入", final["fallback_injected"] and final["fallback_triggered"])
    check("子任务内 consecutive_failures 计到 2", final["consecutive_failures"] == 2)


# ============================================================
# W9 内层 turn_count 闸门收口到 critic
# ============================================================

def w9():
    print(f"\n[W9] 内层 turn_count 闸门：子任务内模型永远要工具 → 停在 {MAX_TURNS} 轮后交 critic")
    set_stubs(
        planner=ScriptedPlanner([plan_json("停不下来的子任务")]),
        executor=InfiniteToolModel("web_search"),   # 永远要工具，靠 turn_count 闸门收口
        critic=ScriptedCritic(["accept"]),          # 跑满 turn 仍无答案 → critic 硬闸门转 escalate
        tools=make_execute_tool({"web_search": {"results": []}}),
    )
    final, _ = run(build_test_graph(), "q-turngate", "w9")  # 不该抛 GraphRecursionError
    check(f"每次执行恰好 {MAX_TURNS} 轮后收口（内层闸门）", final["turn_count"] == MAX_TURNS)
    check("收口后交 critic 审（step_results 留有该步裁决）", len(final["step_results"]) >= 1)
    check("turn 跑满无答案 → critic escalate → skip-and-advance 优雅收口（单步走完）",
          final["termination_reason"] == "all_steps_done")
    check("仍产出报告，未撞 recursion_limit", isinstance(final.get("answer"), str) and bool(final["answer"]))


# ============================================================
# W10 收尾链尾保真：报告 → human_review → update_memory
# ============================================================

def w10():
    print("\n[W10] 收尾链尾：结构化报告 → human_review（审批框永不空）→ update_memory")
    config.INTERRUPT_ENABLED = True
    try:
        set_stubs(
            planner=ScriptedPlanner([plan_json("子任务")]),
            executor=ScriptedModel([resp_stop("可审批的结论")]),
            critic=ScriptedCritic(["accept"]),
        )
        g = build_test_graph()
        mid, cfg = run(g, "q-review", "w10")
        check("出现 __interrupt__ 暂停在 human_review", "__interrupt__" in mid)
        check("审批框收到的是组装好的报告（非空）",
              bool(mid["__interrupt__"][0].value.get("draft_answer")))
        check("审批框报告含子任务结论",
              "可审批的结论" in mid["__interrupt__"][0].value.get("draft_answer", ""))
        final = g.invoke(Command(resume="人工最终版报告"), cfg)
        check("resume 文本改写最终报告", final["answer"] == "人工最终版报告")
    finally:
        config.INTERRUPT_ENABLED = False


# ============================================================
# W11 业务 retry 轻量重置（retry_reset：每次 retry 是带满额 turn 预算的全新尝试）
# ============================================================

def w11():
    print("\n[W11] 业务 retry 轻量重置：retry 全新 turn 预算 + 重置工具标志；保留 retry_count/chunks")
    set_stubs(
        planner=ScriptedPlanner([plan_json("子任务X")]),
        executor=ScriptedModel([
            resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("初稿(被打回)"),  # 首次 2 turn + 检索
            resp_stop("重做稿"),                                                            # retry：全新预算
        ]),
        critic=ScriptedCritic(["retry", "accept"]),
        tools=make_execute_tool({"retrieve_documents": {"results": [
            {"doc": "d", "section": "s", "chunk_id": 1, "score": 0.9}]}}),
    )
    final, _ = run(build_test_graph(), "q-retry-reset", "w11")
    check("retry 重置 turn_count（全新满额预算 =1，而非累加到 3）", final["turn_count"] == 1)
    check("retry 重置 has_retrieved（纠正闸恢复；首次置 True 后被打回）", final["has_retrieved"] is False)
    check("retry 保留 retrieved_chunks（换措辞重做仍可引用首次召回）", len(final["retrieved_chunks"]) == 1)
    check("retry 保留 retry_count（业务重试额度累计）", final["retry_count"] == 1)
    check("最终拿到重做稿", "重做稿" in final["answer"])


# ============================================================
# W12 引用闸门：容忍 section 缩写，仍拒非检索来源（真实跑暴露的误杀修复）
# ============================================================

def w12():
    print("\n[W12] 引用闸门：容忍 section 缩写 + 接地比例软闸门（修真实跑暴露的误杀级联）")
    from nodes import _citations_legal, _citation_grounding
    from config import CITATION_MIN_GROUNDING
    # 本地 chunk 的 section 是很长的层级路径（真实库就是这样）
    full = "第二周学习复盘：工具调用与单 Agent 雏形 > 二、关键认知变化 > 认知 2：模型不执行工具，只决定调用什么"
    chunks = [{"doc": "第二周学习复盘", "section": full, "chunk_id": 1},
              {"doc": "Agent_Loop_设计笔记", "section": "Agent_Loop_设计笔记 > 三、检查机制 > 降级阈值为什么是 2", "chunk_id": 2}]

    # —— 单条匹配（容忍缩写，仍拒编造） ——
    check("缩写到叶子的引用合法（修前被误杀 → retry 级联）",
          _citations_legal(["第二周学习复盘#认知 2：模型不执行工具，只决定调用什么"], chunks))
    check("完整 section 合法（回归）", _citations_legal([f"第二周学习复盘#{full}"], chunks))
    check("叶子前缀缩写也合法（认知 2）", _citations_legal(["第二周学习复盘#认知 2"], chunks))
    check("编造 doc 被拒（doc 未检索）", not _citations_legal(["幻觉文档#不存在的节"], chunks))
    check("真 doc + 编造 section 被拒", not _citations_legal(["第二周学习复盘#完全不存在的小节"], chunks))

    # —— 接地比例软闸门（critic 实际用的）——
    rich = [f"第二周学习复盘#认知 2", "Agent_Loop_设计笔记#降级阈值为什么是 2",
            "第二周学习复盘#认知 2：模型不执行工具，只决定调用什么", "幻觉文档#编造节"]  # 3 真 1 假 = 0.75
    check("富报告多数引用接地（3 真 1 假=0.75）→ 放行",
          _citation_grounding(rich, chunks) >= CITATION_MIN_GROUNDING)
    check("多数编造（1 真 2 假≈0.33）→ 不放行",
          _citation_grounding(["第二周学习复盘#认知 2", "幻觉A#x", "幻觉B#y"], chunks) < CITATION_MIN_GROUNDING)
    check("零检索满篇引用 → 0.0 不放行", _citation_grounding(["第二周学习复盘#认知 2"], []) == 0.0)
    check("无引用不在此闸门管（记 1.0）", _citation_grounding([], chunks) == 1.0)


# ============================================================
# 入口
# ============================================================

def main():
    print("=== v5.0 图结构测试（桩 planner/executor/critic，离线） ===")
    for t in (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12):
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
