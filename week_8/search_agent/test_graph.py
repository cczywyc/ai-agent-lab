"""
v6.0 图结构测试 — 桩 supervisor/researcher/writer/reviewer，不调真实 API，离线可复现。

测的是"supervisor 多 Agent 外循环的接线"而不是模型质量。第八周 E1–E7 桩探针
（week_8/experiment/）在最小图上单独验过框架机制，这里把同样的机制放进**真实 v6.0 图**
里端到端复跑一遍（对应设计草稿 v0.2 §七 状态图 + 已验证清单 7 项）：

  M1  happy path：supervisor 拆研究子任务 → researcher 逐个检索+压缩 findings →
      writer 写初稿 → reviewer accept → 结构化交付（researcher→writer→reviewer 串接）
  M2  supervisor 三级路由（E1 进真实图）：无 findings→researcher / 有 findings 无 draft→writer
      / 有 draft 无 verdict→reviewer；routing_accuracy=3/3，写歪路由被评测抓出
  M3  task_description 四要素（E2 进真实图）：objective/output_format/tools_hint/boundary
      齐全到达 worker；boundary（第 4 条）不丢
  M4  writer↔reviewer 打回循环（E3 进真实图）：恒 reject → review_count 闸门收口、走 best-so-far，
      不撞 recursion_limit
  M5  两计数器不串扰（E4 进真实图）：review_count（打回）vs replan_count（researcher skip）
      各记各的；写字段≡读字段
  M6  上下文隔离（E5 进真实图）：三 worker 喂 LLM 的是各自投影、不是全 state；
      关隔离 → writer 越界读 reviewer 私有字段
  M7  best-so-far 收口（E6 进真实图）：达上限取历史最好稿（0.8）而非最新稿（0.5）
  M8  内外双闸门嵌套（E7 进真实图）：researcher 内层 turn_count 与外层 review_count 互不误伤；
      恒 reject 外层先收口；turn_count 每次进 researcher 归零（本周接回 plan 内循环后复证）
  M9  researcher 引擎原样复用（v5.0 内层保真）：子任务内 tool 调用 + 检索纠正 + 降级

跑法：../../.venv/bin/python test_graph.py
"""

import json
import sys
from types import SimpleNamespace

from langgraph.types import Command

import config
import nodes
from config import MAX_TURNS, MAX_REVIEW, RECURSION_LIMIT
from graph import build_graph
from memory.ltm_store import make_inmemory_store


def build_test_graph():
    """离线图：注入 stub embed 的 InMemoryStore（不碰真实 ltm.db / sqlite 扩展）。"""
    return build_graph(store=make_inmemory_store(embed_fn=lambda ts: [[0.0] * 3 for _ in ts], dims=3))


# ============================================================
# 桩：researcher executor 模型（call_model）+ 工具
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
    """researcher executor 模型桩：按脚本依次吐响应；记录每次收到的窗口（断言切片/重锚）。"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, oai_messages):
        self.calls.append(list(oai_messages))
        if not self.steps:
            raise AssertionError("ScriptedModel: 脚本已耗尽，researcher 多调了一次模型")
        return self.steps.pop(0)


class InfiniteToolModel:
    """永远要工具的 researcher executor 桩（turn_count 闸门压力测试，不会被脚本耗尽）。"""

    def __init__(self, tool="web_search"):
        self.tool = tool
        self.calls = []

    def __call__(self, oai_messages):
        self.calls.append(list(oai_messages))
        return resp_tools((self.tool, {"query": "loop"}))


class ScriptedSupervisor:
    """supervisor 拆解模型桩（= v5.0 planner 桩）：返回研究子任务计划文本（JSON 数组）。
    耗尽重复最后一条（re-decompose 压力测试）。注意：supervisor **路由**是条件函数、不接 LLM（E1），
    这个桩只喂"拆解"这一次 LLM 调用。"""

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
    """researcher 内层 critic 桩：按脚本裁决（accept/retry/escalate）；耗尽重复最后一条。"""

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


class ScriptedWriter:
    """writer 模型桩：把 findings 写成初稿。记录每次收到的输入（断言投影 = writer_view）。
    返修轮用收到的输入条数区分稿次。"""

    def __init__(self, template="初稿v{v}：{body} [d#s]"):
        self.template = template
        self.inputs = []

    def __call__(self, messages):
        self.inputs.append(list(messages))
        return self.template.format(v=len(self.inputs) - 1, body="基于findings的草稿")


class ScriptedReviewer:
    """reviewer 模型桩：按脚本裁决 + 打分。记录每次收到的输入（断言投影 = reviewer_view）。
    输出格式与真实 reviewer 一致：VERDICT/SCORE/NOTES 三行。"""

    def __init__(self, verdicts, scores):
        self.verdicts = list(verdicts)
        self.scores = list(scores)
        self.inputs = []

    def __call__(self, messages):
        self.inputs.append(list(messages))
        i = len(self.inputs) - 1
        verdict = self.verdicts[min(i, len(self.verdicts) - 1)]
        score = self.scores[min(i, len(self.scores) - 1)]
        return f"VERDICT: {verdict}\nSCORE: {score}\nNOTES: 第{i}轮意见：引用是否齐全=否"


def plan_json(*queries):
    return json.dumps(list(queries), ensure_ascii=False)


def make_execute_tool(results_by_name):
    log = []

    def _exec(tool_name, tool_args):
        log.append((tool_name, tool_args))
        result = results_by_name.get(tool_name, {"ok": True})
        return result(log) if callable(result) else result

    _exec.log = log
    return _exec


def run(graph, topic, thread_id, use_memory=False):
    cfg = {
        "configurable": {"thread_id": thread_id, "use_memory": use_memory},
        "recursion_limit": RECURSION_LIMIT,
    }
    result = graph.invoke({"user_message": topic}, cfg)
    return result, cfg


CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"  {'✓ PASS' if cond else '✗ FAIL'}  {name}")
    return bool(cond)


def set_stubs(*, supervisor=None, researcher=None, critic=None,
              writer=None, reviewer=None, tools=None):
    """统一注入桩——supervisor 拆解 / researcher executor / 内层 critic / writer / reviewer / 工具。"""
    if supervisor is not None:
        nodes.call_supervisor_model = supervisor
    if researcher is not None:
        nodes.call_model = researcher
    if critic is not None:
        nodes.call_critic_model = critic
    if writer is not None:
        nodes.call_writer_model = writer
    if reviewer is not None:
        nodes.call_reviewer_model = reviewer
    if tools is not None:
        nodes.execute_tool = tools


RETRIEVE_OK = {"retrieve_documents": {"results": [
    {"doc": "d", "section": "s", "chunk_id": 1, "score": 0.9}]}}


# ============================================================
# M1 happy path：supervisor 拆 2 子任务 → researcher×2 → writer → reviewer accept → 交付
# ============================================================

def m1():
    print("\n[M1] happy path：supervisor 拆研究子任务 → researcher 检索 → writer 初稿 → reviewer accept → 交付")
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("多 Agent 拓扑有哪些", "委派契约四要素是什么")]),
        researcher=ScriptedModel([
            resp_tools(("retrieve_documents", {"query": "拓扑"})), resp_stop("拓扑 A/B/C [d#s]"),
            resp_tools(("retrieve_documents", {"query": "契约"})), resp_stop("契约四要素 [d#s]"),
        ]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(),
        reviewer=ScriptedReviewer(["accept"], [0.9]),
        tools=make_execute_tool(RETRIEVE_OK),
    )
    final, _ = run(build_test_graph(), "调研多 Agent 系统并写一篇技术综述", "m1")

    check("supervisor 拆出 2 个研究子任务", len(final.get("plan", [])) == 2)
    check("两研究子任务都标 done", all(s.get("status") == "done" for s in final.get("plan", [])))
    check("findings 累加 2 条（researcher 压缩回传）", len(final.get("findings", [])) == 2)
    check("findings 带引用", all(f.get("citations") for f in final.get("findings", [])))
    check("writer 产出了初稿（draft 非空）", bool(final.get("draft")))
    check("reviewer accept", final.get("review_verdict") == "accept")
    check("最终交付含 writer 初稿", final.get("draft", "") in final.get("answer", ""))
    check("早退：reviewer 只审一次（accept 立即出环）",
          len(nodes.call_reviewer_model.inputs) == 1)


# ============================================================
# M2 supervisor 三级路由 + routing_accuracy（E1 进真实图）
# ============================================================

def m2():
    print("\n[M2] supervisor 三级路由：无 findings→researcher / 有 findings 无 draft→writer / 有 draft→reviewer；routing_accuracy")
    from evals import routing_accuracy, bad_route_always_writer
    hits, total, detail = routing_accuracy()
    check(f"真实路由 routing_accuracy={hits}/{total}（三级阶段判定全对）", hits == total == 3)
    bhits, btotal, _ = routing_accuracy(bad_route_always_writer)
    check("对照：写歪路由（恒派 writer）被评测抓出（命中 1/3）", bhits == 1)
    check("评测能区分对/歪路由（hits > bad_hits）", hits > bhits)


# ============================================================
# M3 task_description 四要素（E2 进真实图）：boundary 不丢、到达 researcher
# ============================================================

def m3():
    print("\n[M3] task_description 四要素：objective/output_format/tools_hint/boundary 齐全、boundary 到达 researcher")
    from nodes import _researcher_td, _writer_td, _reviewer_td
    seed = {"user_message": "主题", "plan": [{"id": 0, "query": "拓扑有哪些", "status": "pending"}],
            "step_index": 0}
    four = {"objective", "output_format", "tools_hint", "boundary"}
    for name, td in (("researcher", _researcher_td(seed)), ("writer", _writer_td(seed)),
                     ("reviewer", _reviewer_td(seed))):
        check(f"{name} task_description 四要素齐全", four <= set(td))
        check(f"{name} 第 4 条 boundary 非空（v5.0 唯一系统性缺的那条）", bool(td.get("boundary")))
    check("researcher objective = 当前研究子任务 query", _researcher_td(seed)["objective"] == "拓扑有哪些")

    # boundary 真到达 researcher：跑一遍，看 researcher 第一窗口 user 内容里带了 boundary
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("拓扑有哪些")]),
        researcher=ScriptedModel([resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("结论 [d#s]")]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(), reviewer=ScriptedReviewer(["accept"], [0.9]),
        tools=make_execute_tool(RETRIEVE_OK),
    )
    run(build_test_graph(), "调研多 Agent", "m3")
    first_window = nodes.call_model.calls[0]
    user_blob = " ".join(m.get("content", "") for m in first_window if m.get("role") == "user")
    check("boundary 到达 researcher（窗口里出现『研究边界』段）", "研究边界" in user_blob)
    # 对照：drop boundary 可检出（td 缺第 4 键即被 set 判出）
    td_drop = {k: v for k, v in _researcher_td(seed).items() if k != "boundary"}
    check("对照：漏 boundary 可检出（四要素不再 ⊆ td）", not (four <= set(td_drop)))


# ============================================================
# M4 writer↔reviewer 打回循环（E3 进真实图）：恒 reject → review_count 闸门收口走 best-so-far
# ============================================================

def m4():
    print(f"\n[M4] 打回循环：reviewer 恒 reject → review_count 闸门停在 {MAX_REVIEW}、走 best-so-far、不撞 recursion")
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("唯一研究子任务")]),
        researcher=ScriptedModel([resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("结论 [d#s]")]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(),
        reviewer=ScriptedReviewer(["reject"], [0.6, 0.4]),     # 恒 reject
        tools=make_execute_tool(RETRIEVE_OK),
    )
    # recursion_limit=60：足够本有界打回循环（研究+2 轮评审约 16 个 super-step）跑完，但闸门若失效（死循环）会撞它。
    # 闸门正常 → 在 review_count==MAX_REVIEW 优雅收口、不抛 GraphRecursionError（"拆闸门撞 recursion"的负对照见 E3 桩）。
    final = build_test_graph().invoke({"user_message": "调研多 Agent 写综述"},
                                      {"configurable": {"thread_id": "m4", "use_memory": False},
                                       "recursion_limit": 60})
    check(f"恒 reject 停在 review_count=={MAX_REVIEW}（闸门收口）", final.get("review_count") == MAX_REVIEW)
    check("reviewer 恰被调 MAX_REVIEW 次", len(nodes.call_reviewer_model.inputs) == MAX_REVIEW)
    check("走 best-so-far 收口出口（termination=best_so_far）",
          "best_so_far" in (final.get("termination_reason", "")))
    check("仍产出交付（优雅收口、非抛 GraphRecursionError）",
          isinstance(final.get("answer"), str) and bool(final.get("answer")))


# ============================================================
# M5 两计数器不串扰（E4 进真实图）：review_count（打回）vs replan_count（researcher skip）各记各的
# ============================================================

def m5():
    print("\n[M5] 两计数器不串扰：研究子任务 skip(replan_count) 与 writer↔reviewer 打回(review_count) 各记各的")
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("会被跳过的子任务", "正常子任务")]),
        researcher=ScriptedModel([
            resp_tools(("retrieve_documents", {"query": "a"})), resp_stop("方向不对的结论 [d#s]"),  # 子任务0 → critic escalate → skip
            resp_tools(("retrieve_documents", {"query": "b"})), resp_stop("正常结论 [d#s]"),          # 子任务1 → accept
        ]),
        critic=ScriptedCritic(["escalate", "accept"]),
        writer=ScriptedWriter(),
        reviewer=ScriptedReviewer(["reject"], [0.5]),          # 恒 reject → review_count 跑满
        tools=make_execute_tool(RETRIEVE_OK),
    )
    final, _ = run(build_test_graph(), "调研多 Agent 写综述", "m5")
    check("replan_count=1（一个研究子任务被 skip）", final.get("replan_count") == 1)
    check(f"review_count={MAX_REVIEW}（打回跑满，未被 skip 吃额度）", final.get("review_count") == MAX_REVIEW)
    check("两计数器独立：replan=1 且 review=2", final.get("replan_count") == 1 and final.get("review_count") == 2)
    check("findings 含被 skip 的子任务标记（status=skipped）",
          any(f.get("status") == "skipped" for f in final.get("findings", [])))


# ============================================================
# M6 上下文隔离（E5 进真实图）：worker 喂 LLM 的是投影、不是全 state；关隔离 → writer 越界读 reviewer 私有
# ============================================================

def m6():
    print("\n[M6] 上下文隔离：三 worker 可见集=各自投影；关隔离 → writer 越界读 reviewer 私有（串台）")
    # --- 隔离开（默认）：三 worker 可见集 = 各自设计视图 ---
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("拓扑有哪些")]),
        researcher=ScriptedModel([resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("结论 [d#s]")]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(), reviewer=ScriptedReviewer(["accept"], [0.9]),
        tools=make_execute_tool(RETRIEVE_OK),
    )
    final, _ = run(build_test_graph(), "调研多 Agent", "m6-on")
    check("researcher 可见集 = {boundary, subtask_query}",
          set(final.get("_researcher_visible", [])) == {"boundary", "subtask_query"})
    check("writer 可见集 = {findings, outline, review_notes}",
          set(final.get("_writer_visible", [])) == {"findings", "outline", "review_notes"})
    check("reviewer 可见集 = {draft, rubric}",
          set(final.get("_reviewer_visible", [])) == {"draft", "rubric"})
    check("隔离开：writer 看不到 reviewer 私有 review_verdict/best_draft",
          "review_verdict" not in final.get("_writer_visible", [])
          and "best_draft" not in final.get("_writer_visible", []))

    # --- 关隔离（对照组）：writer 返修第二遍越界读到 reviewer 私有 = 串台 ---
    config.ISOLATION_ENABLED = False
    try:
        set_stubs(
            supervisor=ScriptedSupervisor([plan_json("拓扑有哪些")]),
            researcher=ScriptedModel([resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("结论 [d#s]")]),
            critic=ScriptedCritic(["accept"]),
            writer=ScriptedWriter(),
            reviewer=ScriptedReviewer(["reject", "accept"], [0.5, 0.9]),   # 先 reject（writer 返修）再 accept
            tools=make_execute_tool(RETRIEVE_OK),
        )
        final2, _ = run(build_test_graph(), "调研多 Agent", "m6-off")
        wv = final2.get("_writer_visible", [])
        check("关隔离：writer 越界读到 reviewer 私有 review_verdict（串台）", "review_verdict" in wv)
        check("关隔离：writer 越界读到 reviewer 私有 best_draft（串台）", "best_draft" in wv)
    finally:
        config.ISOLATION_ENABLED = True


# ============================================================
# M7 best-so-far 收口（E6 进真实图）：达上限取历史最好稿（0.8）而非最新稿（0.5）
# ============================================================

def m7():
    print("\n[M7] best-so-far：达上限取历史最好稿（0.8）而非最新更差稿（0.5，behavioral collapse）")
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("唯一研究子任务")]),
        researcher=ScriptedModel([resp_tools(("retrieve_documents", {"query": "x"})), resp_stop("结论 [d#s]")]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(),
        reviewer=ScriptedReviewer(["reject"], [0.8, 0.5]),     # 恒 reject，后稿更差（0.5<0.8）
        tools=make_execute_tool(RETRIEVE_OK),
    )
    final, _ = run(build_test_graph(), "调研多 Agent 写综述", "m7")
    check("best_draft 取历史最好分 0.8（非最新 0.5）", final.get("best_draft", {}).get("score") == 0.8)
    check("交付正文是初稿v0（best=0.8 那稿）", "初稿v0" in final.get("answer", ""))
    check("交付正文不含更差的初稿v1（0.5 那稿没被交付）", "初稿v1" not in final.get("answer", ""))
    check("worker_result.chosen 取 best（score=0.8）",
          final.get("worker_result", {}).get("chosen", {}).get("score") == 0.8)


# ============================================================
# M8 内外双闸门嵌套（E7 进真实图）：researcher 内层 turn_count 与外层 review_count 互不误伤 + 每次进归零
# ============================================================

def m8():
    print(f"\n[M8] 双闸门嵌套：researcher 内层 turn_count（每子任务归零）与外层 review_count={MAX_REVIEW} 互不误伤")
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("研究子任务A", "研究子任务B")]),
        researcher=ScriptedModel([
            resp_tools(("retrieve_documents", {"query": "a"})), resp_stop("A 结论 [d#s]"),   # 子任务A：2 turn
            resp_tools(("retrieve_documents", {"query": "b"})), resp_stop("B 结论 [d#s]"),   # 子任务B：又 2 turn（不累加成 4）
        ]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(),
        reviewer=ScriptedReviewer(["reject"], [0.5]),          # 恒 reject → 外层 review 闸门收口
        tools=make_execute_tool(RETRIEVE_OK),
    )
    final, _ = run(build_test_graph(), "调研多 Agent 写综述", "m8")
    check("turn_count 每次进 researcher 归零（终态=2、非跨子任务累加成 4）", final.get("turn_count") == 2)
    check(f"外层 review_count 独立收口到 {MAX_REVIEW}", final.get("review_count") == MAX_REVIEW)
    check("内外闸门互不误伤：turn_count 不被外层 review 循环累加", final.get("turn_count") < MAX_TURNS)
    check("replan_count=0（两子任务都 accept，研究 skip 闸门未触发）", final.get("replan_count") == 0)
    check("仍产出交付、未撞 recursion_limit", isinstance(final.get("answer"), str) and bool(final.get("answer")))


# ============================================================
# M9 researcher 引擎原样复用（v5.0 内层保真）：子任务内 tool 调用 + 检索纠正 + 降级
# ============================================================

def m9():
    print("\n[M9] researcher 引擎保真：子任务内检索纠正 + fetch 连续失败降级（v5.0 内层原样复用）")

    # --- 子任务内：凭记忆直答 → 检索纠正注入 → 检索后产出 ---
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("我们第三周的 Agent Loop 是怎么处理工具连续失败的")]),
        researcher=ScriptedModel([
            resp_stop("凭记忆直答(应被纠正)"),
            resp_tools(("retrieve_documents", {"query": "连续失败"})),
            resp_stop("基于 [doc#连续失败] 的结论"),
        ]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(), reviewer=ScriptedReviewer(["accept"], [0.9]),
        tools=make_execute_tool({"retrieve_documents": {"results": [
            {"doc": "doc", "section": "连续失败", "chunk_id": 7, "score": 0.88}]}}),
    )
    final, _ = run(build_test_graph(), "调研第三周 Agent Loop", "m9-correct")
    check("researcher 内检索纠正已注入（v5.0 内层保真）", final.get("retrieval_correction_injected"))
    check("researcher 内 has_retrieved=True", final.get("has_retrieved"))
    check("findings 来自检索后产出（带引用）",
          any("doc#连续失败" in (c for c in f.get("citations", [])) or f.get("citations")
              for f in final.get("findings", [])))

    # --- 子任务内：fetch 连续失败 ≥2 → 降级注入一次 ---
    set_stubs(
        supervisor=ScriptedSupervisor([plan_json("读取某网页正文")]),
        researcher=ScriptedModel([
            resp_tools(("fetch_webpage", {"url": "https://x/a"})),
            resp_tools(("fetch_webpage", {"url": "https://x/b"})),
            resp_stop("以下回答基于搜索摘要，未能获取完整正文。"),   # 无 [doc#section]——fetch 失败没召回，引 doc 会被 critic 判 grounding=0
        ]),
        critic=ScriptedCritic(["accept"]),
        writer=ScriptedWriter(), reviewer=ScriptedReviewer(["accept"], [0.9]),
        tools=make_execute_tool({"fetch_webpage": {"error": True, "error_type": "HTTPError", "message": "403"}}),
    )
    final, _ = run(build_test_graph(), "读网页", "m9-fallback")
    check("researcher 内降级已注入（fetch 连续失败≥2）",
          final.get("fallback_injected") and final.get("fallback_triggered"))
    check("researcher 内 consecutive_failures 计到 2", final.get("consecutive_failures") == 2)


# ============================================================
# 入口
# ============================================================

def main():
    print("=== v6.0 图结构测试（桩 supervisor/researcher/writer/reviewer，离线） ===")
    for t in (m1, m2, m3, m4, m5, m6, m7, m8, m9):
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
