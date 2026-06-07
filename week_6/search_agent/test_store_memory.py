"""
v4.1 长期记忆 × LangGraph Store 离线测试 — InMemoryStore + stub embed，不调真实 API。

对应《Store集成与重构计划》阶段四-13：
  S1 put→search 召回顺序对
  S2 min_score 过滤对
  S3 namespace 隔离对
  S4 prefs 的 get/put 对（含覆盖、遗忘）
  S5 topics 的 get/put 对（计数累加、top_topics 排序）
  S6 语义索引只给 facts：prefs/topics 读写不调 embed
  S7 assembler 消费 store：段 2/段 4 进装配，段顺序/预算逻辑不变
  S8 manager 写路径走 store：update_from_turn 三类全写入；info/reset 对
  S9 图端到端（桩模型）：store 经 compile(store=...) 注入节点，跨问题召回生效
  S10 装配窗口修复：记忆开启时发给模型的窗口包含段 1 SYSTEM_PROMPT（v4.0 bug 回归测试）

跑法：../../.venv/bin/python test_store_memory.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# ============================================================
# stub embed — 确定性向量（单位长度，余弦可手算）
# ============================================================

FACT_A = "猫是哺乳动物，体温恒定。"
FACT_B = "狗通过吠叫交流。"
FACT_C = "天空因瑞利散射呈蓝色。"

Q_CAT_DOG = "猫和狗哪个更亲人？"     # cos: A=0.8, B=0.6, C=0
Q_DOG_ONLY = "狗的交流方式"          # cos: A=0.28(<0.30), B=0.96, C=0

STUB_VECTORS = {
    FACT_A: [1.0, 0.0, 0.0],
    FACT_B: [0.0, 1.0, 0.0],
    FACT_C: [0.0, 0.0, 1.0],
    Q_CAT_DOG: [0.8, 0.6, 0.0],
    Q_DOG_ONLY: [0.28, 0.96, 0.0],
}
DEFAULT_VEC = [0.577, 0.577, 0.577]  # 未知文本 → 同一向量（互相余弦=1）

embed_calls: list[list[str]] = []


def stub_embed(texts):
    embed_calls.append(list(texts))
    return [STUB_VECTORS.get(t, DEFAULT_VEC) for t in texts]


def fresh_store():
    from memory.ltm_store import make_inmemory_store
    return make_inmemory_store(embed_fn=stub_embed, dims=3)


PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    status = "✅" if cond else "❌"
    print(f"  {status} {name}" + (f"  ({detail})" if detail and not cond else ""))
    if cond:
        PASS += 1
    else:
        raise AssertionError(f"{name}: {detail}")


# ============================================================
# S1 — put→search 召回顺序
# ============================================================
def test_s1_recall_order():
    print("\nS1 put→search 召回顺序")
    from memory import long_term as lt

    store = fresh_store()
    lt.add_facts(store, [
        lt.Fact(fact=FACT_A, source="doc#a", turn=1),
        lt.Fact(fact=FACT_B, source="doc#b", turn=1),
        lt.Fact(fact=FACT_C, source="doc#c", turn=2),
    ])
    hits = lt.recall_facts(store, Q_CAT_DOG, top_k=3, min_score=0.30)
    check("召回 2 条（C 被 0 分挡掉）", len(hits) == 2, f"got {len(hits)}")
    check("顺序 A > B（0.8 > 0.6）",
          [h["fact"] for h in hits] == [FACT_A, FACT_B],
          str([h["fact"][:6] for h in hits]))
    check("score 字段保留且近似正确",
          abs(hits[0]["score"] - 0.8) < 0.01 and abs(hits[1]["score"] - 0.6) < 0.01,
          str([h["score"] for h in hits]))
    check("元数据 source/turn 原样带回",
          hits[0]["source"] == "doc#a" and hits[0]["turn"] == 1)


# ============================================================
# S2 — min_score 过滤
# ============================================================
def test_s2_min_score():
    print("\nS2 min_score 过滤")
    from memory import long_term as lt

    store = fresh_store()
    lt.add_facts(store, [
        lt.Fact(fact=FACT_A, source="doc#a", turn=1),
        lt.Fact(fact=FACT_B, source="doc#b", turn=1),
    ])
    hits = lt.recall_facts(store, Q_DOG_ONLY, top_k=3, min_score=0.30)
    check("0.28 < 0.30 的 A 被过滤，只回 B",
          len(hits) == 1 and hits[0]["fact"] == FACT_B,
          str([(h["fact"][:6], h["score"]) for h in hits]))
    raw = lt.recall_facts(store, Q_DOG_ONLY, top_k=3, min_score=-1.0)
    check("min_score=-1 时 A/B 都在", len(raw) == 2)


# ============================================================
# S3 — namespace 隔离
# ============================================================
def test_s3_namespace_isolation():
    print("\nS3 namespace 隔离")
    from memory import long_term as lt
    from memory.ltm_store import NS_FACTS, NS_PREFS, NS_TOPICS

    store = fresh_store()
    lt.add_facts(store, [lt.Fact(fact=FACT_A, source="doc#a", turn=1)])
    lt.set_preference(store, "style", "先结论后引用")
    lt.bump_topics(store, ["rag"])

    check("三个 namespace 各自只有 1 条",
          len(store.search(NS_FACTS, limit=50)) == 1
          and len(store.search(NS_PREFS, limit=50)) == 1
          and len(store.search(NS_TOPICS, limit=50)) == 1)

    fact_hits = lt.recall_facts(store, Q_CAT_DOG, top_k=10, min_score=-1.0)
    check("facts 语义召回不串入 prefs/topics",
          all(h["fact"] == FACT_A for h in fact_hits) and len(fact_hits) == 1)

    prefs = lt.list_preferences(store)
    check("prefs 列表不串入 facts/topics", prefs == {"style": "先结论后引用"})


# ============================================================
# S4 — 偏好 get/put
# ============================================================
def test_s4_preferences():
    print("\nS4 偏好 get/put")
    from memory import long_term as lt

    store = fresh_store()
    lt.set_preference(store, "style", "先结论后引用")
    lt.set_preference(store, "lang", "中文回答")
    check("写入 2 条全量可读", lt.list_preferences(store) ==
          {"style": "先结论后引用", "lang": "中文回答"})

    lt.set_preference(store, "style", "改成表格输出")
    check("同 key 覆盖", lt.list_preferences(store)["style"] == "改成表格输出")

    check("forget 已存在的 key 返回 True", lt.forget_preference(store, "lang") is True)
    check("forget 后只剩 1 条", lt.list_preferences(store) == {"style": "改成表格输出"})
    check("forget 不存在的 key 返回 False", lt.forget_preference(store, "nope") is False)


# ============================================================
# S5 — 主题计数 get/put
# ============================================================
def test_s5_topics():
    print("\nS5 主题计数")
    from memory import long_term as lt

    store = fresh_store()
    lt.bump_topics(store, ["rag", "memory"])
    lt.bump_topics(store, ["rag"])
    lt.bump_topics(store, ["rag", "fallback"])
    tops = lt.top_topics(store, n=5)
    check("计数累加：rag=3 居首", tops[0] == ("rag", 3), str(tops))
    check("全部主题在列", dict(tops) == {"rag": 3, "memory": 1, "fallback": 1})
    check("top_topics(n=1) 截断", lt.top_topics(store, n=1) == [("rag", 3)])


# ============================================================
# S6 — 语义索引只给 facts（prefs/topics 不调 embed）
# ============================================================
def test_s6_embed_economy():
    print("\nS6 prefs/topics 读写不调 embed")
    from memory import long_term as lt

    store = fresh_store()
    embed_calls.clear()
    lt.set_preference(store, "style", "先结论后引用")
    lt.bump_topics(store, ["rag", "memory"])
    lt.bump_topics(store, ["rag"])
    lt.list_preferences(store)
    lt.top_topics(store)
    check("以上全部操作 0 次 embed", len(embed_calls) == 0, f"{len(embed_calls)} calls")

    lt.add_facts(store, [lt.Fact(fact=FACT_A, source="doc#a", turn=1)])
    check("写 1 批 facts 恰好 1 次 embed", len(embed_calls) == 1)
    lt.recall_facts(store, Q_CAT_DOG)
    check("一次语义召回恰好 1 次 embed（query）", len(embed_calls) == 2)


# ============================================================
# S7 — assembler 消费 store
# ============================================================
def test_s7_assembler():
    print("\nS7 assembler 消费 store")
    from config import SYSTEM_PROMPT
    from memory import long_term as lt
    from memory.assembler import ContextAssembler
    from memory.short_term import ShortTermMemory

    store = fresh_store()
    lt.set_preference(store, "style", "先结论后引用")
    lt.add_facts(store, [
        lt.Fact(fact=FACT_A, source="doc#a", turn=1),
        lt.Fact(fact=FACT_B, source="doc#b", turn=1),
        lt.Fact(fact=FACT_C, source="doc#c", turn=2),
    ])

    asm = ContextAssembler(
        short_term=ShortTermMemory(k=3, char_budget=4000),
        summary_text_getter=lambda: "",
    )
    msgs, report = asm.assemble(Q_CAT_DOG, SYSTEM_PROMPT, store)

    check("段 1 = SYSTEM_PROMPT 在首位",
          msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT)
    check("段 2 偏好进装配",
          msgs[1]["role"] == "system" and "[用户偏好]" in msgs[1]["content"]
          and "先结论后引用" in msgs[1]["content"])
    check("段 4 事实进装配（A、B 进，C 被 min_score 挡）",
          "[相关长期事实" in msgs[2]["content"]
          and FACT_A in msgs[2]["content"] and FACT_B in msgs[2]["content"]
          and FACT_C not in msgs[2]["content"])
    check("段 6 当前问题收尾",
          msgs[-1] == {"role": "user", "content": Q_CAT_DOG})
    check("report 计数对",
          report.facts_recalled == 2
          and report.segments_present == ["system", "preferences", "facts", "current"],
          str(report))

    # store 为空时优雅降级
    empty = fresh_store()
    msgs2, report2 = asm.assemble("随便问点什么", SYSTEM_PROMPT, empty)
    check("空 store：只有段 1 + 段 6",
          len(msgs2) == 2 and report2.segments_present == ["system", "current"])


# ============================================================
# S8 — manager 写路径走 store
# ============================================================
def test_s8_manager():
    print("\nS8 manager 写路径")
    from memory.manager import MemoryManager
    from memory import long_term as lt

    store = fresh_store()
    tmp = Path(tempfile.mkdtemp())
    mgr = MemoryManager(persist_dir=tmp, autoload=False)
    trace = SimpleNamespace(turns=[], searched=False, retrieved=True)

    mgr.update_from_turn(
        "请记住：以后回答涉及本地文档时，先列结论再列引用。",
        "好的，我会先列结论再列引用。",
        trace, store,
    )
    prefs = lt.list_preferences(store)
    check("偏好经 store 写入", len(prefs) == 1 and "先列结论再列引用" in next(iter(prefs.values())))

    mgr.update_from_turn(
        "我们的降级机制是怎么设计的？",
        "- 连续失败达到阈值后触发降级，转用已有摘要回答 [设计笔记#降级机制]。",
        trace, store,
    )
    facts = lt.recall_facts(store, "降级机制", top_k=5, min_score=-1.0)
    check("带引用回答的事实经 store 写入（不再手动 embed）",
          len(facts) == 1 and facts[0]["source"] == "设计笔记#降级机制")

    tops = dict(lt.top_topics(store, n=10))
    check("主题计数经 store 累加", tops.get("fallback", 0) >= 1, str(tops))

    info = mgr.info(store)
    check("info() 从 store 读数",
          info["facts"] == 1 and len(info["preferences"]) == 1
          and info["short_term_turns"] == 2, str(info))

    mgr.reset(store)
    check("reset() 清空 store 三个 namespace + 短期",
          lt.counts(store) == {"preferences": 0, "facts": 0, "topics": 0}
          and mgr.info(store)["short_term_turns"] == 0)


# ============================================================
# S9 — 图端到端：store 经 compile(store=...) 注入节点
# ============================================================
def test_s9_graph_e2e():
    print("\nS9 图端到端（桩模型 + InMemoryStore）")
    import nodes
    import memory as memory_pkg
    from memory.manager import MemoryManager
    from memory import long_term as lt
    from graph import build_graph
    from config import RECURSION_LIMIT

    store = fresh_store()
    tmp = Path(tempfile.mkdtemp())
    mgr = MemoryManager(persist_dir=tmp, autoload=False)

    real_call, real_get, real_exec = nodes.call_model, memory_pkg.get_memory, nodes.execute_tool
    captured_windows: list[list[dict]] = []

    def scripted_model(steps):
        """str → stop 响应；非 str → 原样返回（可混入 tool_calls 响应）。"""
        it = iter(steps)
        def call(oai_messages):
            captured_windows.append(oai_messages)
            step = next(it)
            if not isinstance(step, str):
                return step
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=step, tool_calls=None),
                finish_reason="stop",
            )])
        return call

    # 第 2 轮先检索再引用（v4.2 契约：引用须对上本轮检索来源，否则不入 store——
    # 旧 fixture "零检索带引用直答"正是该被拒收的噪音场景，见 S13/S14）
    retrieve_resp = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[SimpleNamespace(
            id="call_0",
            function=SimpleNamespace(name="retrieve_documents",
                                     arguments='{"query": "降级机制"}'),
        )]),
        finish_reason="tool_calls",
    )])

    try:
        memory_pkg.get_memory = lambda: mgr
        nodes.call_model = scripted_model([
            "好的，已记住：先列结论再列引用。",
            retrieve_resp,
            "- 连续失败达到阈值后触发降级 [设计笔记#降级机制]。",
            "结论：阈值是 2。",
        ])
        nodes.execute_tool = lambda name, args: {"results": [
            {"doc": "设计笔记", "section": "降级机制", "chunk_id": 3,
             "text": "连续失败达到阈值触发降级", "score": 0.9},
        ]}
        graph = build_graph(store=store)
        cfg = {"configurable": {"thread_id": "s9", "use_memory": True},
               "recursion_limit": RECURSION_LIMIT}

        graph.invoke({"user_message": "请记住：以后回答先列结论再列引用。"}, cfg)
        check("第 1 轮后偏好已进 store（节点注入的 store 生效）",
              len(lt.list_preferences(store)) == 1)

        # 提问刻意避开 should_have_retrieved / should_have_searched 纠正规则
        # （S9 测的是 store 接线，不是纠正路由——那是 test_graph.py T2/T3 的事）
        graph.invoke({"user_message": "帮我把降级机制总结成一句话。"}, cfg)
        check("第 2 轮后事实已进 store", lt.counts(store)["facts"] == 1)

        result = graph.invoke({"user_message": "谢谢，帮我总结一下降级机制。"}, cfg)
        win3 = captured_windows[-1]
        joined = "\n".join(m.get("content") or "" for m in win3)
        check("第 3 轮窗口含段 2 偏好（跨问题记忆生效）", "[用户偏好]" in joined)
        check("第 3 轮窗口含段 4 召回事实", "[相关长期事实" in joined and "触发降级" in joined)
        check("最终回答正常", result["answer"] == "结论：阈值是 2。")
    finally:
        nodes.call_model = real_call
        memory_pkg.get_memory = real_get
        nodes.execute_tool = real_exec


# ============================================================
# S11 — 空回答不进记忆（防一次模型抖动级联污染段 5）
# ============================================================
def test_s11_empty_answer_not_recorded():
    print("\nS11 空回答不进记忆")
    import nodes
    import memory as memory_pkg
    from memory.manager import MemoryManager
    from memory import long_term as lt
    from graph import build_graph
    from config import RECURSION_LIMIT

    store = fresh_store()
    tmp = Path(tempfile.mkdtemp())
    mgr = MemoryManager(persist_dir=tmp, autoload=False)

    real_call, real_get = nodes.call_model, memory_pkg.get_memory
    try:
        memory_pkg.get_memory = lambda: mgr
        # 模型返回空 content 的 stop（qwen 偶发抖动的复刻）
        nodes.call_model = lambda msgs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=None),
            finish_reason="stop",
        )])
        graph = build_graph(store=store)
        cfg = {"configurable": {"thread_id": "s11", "use_memory": True},
               "recursion_limit": RECURSION_LIMIT}
        result = graph.invoke({"user_message": "帮我总结一下记忆系统。"}, cfg)

        check("空回答有占位符", result["answer"] == "[模型返回空回答]")
        check("空回答轮次不进短期记忆", len(mgr.short.turns) == 0,
              f"{len(mgr.short.turns)} turns recorded")
        check("空回答不产生主题计数/事实",
              lt.counts(store) == {"preferences": 0, "facts": 0, "topics": 0},
              str(lt.counts(store)))
    finally:
        nodes.call_model = real_call
        memory_pkg.get_memory = real_get


# ============================================================
# S12 — 时序反转：记忆记录审批后的回答（update_memory 在 human_review 之后）
# ============================================================
def test_s12_memory_records_post_review_answer():
    print("\nS12 时序反转：记忆记录审批后的回答")
    import config
    import nodes
    import memory as memory_pkg
    from langgraph.types import Command
    from memory.manager import MemoryManager
    from graph import build_graph
    from config import RECURSION_LIMIT

    store = fresh_store()
    tmp = Path(tempfile.mkdtemp())
    mgr = MemoryManager(persist_dir=tmp, autoload=False)

    real_call, real_get = nodes.call_model, memory_pkg.get_memory
    try:
        memory_pkg.get_memory = lambda: mgr
        nodes.call_model = lambda msgs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="草稿回答", tool_calls=None),
            finish_reason="stop",
        )])
        config.INTERRUPT_ENABLED = True
        graph = build_graph(store=store)
        cfg = {"configurable": {"thread_id": "s12", "use_memory": True},
               "recursion_limit": RECURSION_LIMIT}

        mid = graph.invoke({"user_message": "帮我把降级机制总结成一句话。"}, cfg)
        check("暂停时草稿可见", "__interrupt__" in mid
              and mid["__interrupt__"][0].value.get("draft_answer") == "草稿回答")
        check("暂停时记忆尚未写入（update_memory 在审批之后）",
              len(mgr.short.turns) == 0, f"{len(mgr.short.turns)} turns at pause")

        final = graph.invoke(Command(resume="人工改写后的回答"), cfg)
        check("最终回答为改写文本", final["answer"] == "人工改写后的回答")
        check("短期记忆存的是用户实际看到的版本",
              len(mgr.short.turns) == 1
              and mgr.short.turns[0].assistant_message == "人工改写后的回答",
              mgr.short.turns[0].assistant_message if mgr.short.turns else "no turns")
    finally:
        config.INTERRUPT_ENABLED = False
        nodes.call_model = real_call
        memory_pkg.get_memory = real_get


# ============================================================
# S13 — 事实抽取白名单（踩坑 #3 收紧：引用须对上 allowed_sources）
# ============================================================
def test_s13_extractor_source_whitelist():
    print("\nS13 事实抽取白名单：引用须在 allowed_sources 内")
    from memory.extractor import extract_fact_candidates

    answer = ("- 降级阈值是连续失败 2 次 [设计笔记#降级机制]。\n"
              "- 这条是编造引用的句子 [幻觉文档#不存在的节]。\n"
              "- 这条引用格式带空格抖动 [设计笔记 # 降级机制]。")
    allowed = {("设计笔记", "降级机制")}

    got = extract_fact_candidates(answer, allowed_sources=allowed)
    texts = [t for t, _ in got]
    check("白名单内引用被抽取", any("降级阈值" in t for t in texts), str(texts))
    check("编造引用被拒收", not any("编造" in t for t in texts), str(texts))
    check("空格格式抖动被归一化吸收", any("空格抖动" in t for t in texts), str(texts))

    # 混合引用：保留事实，但来源只记真实的那部分
    mixed = "- 混合引用的句子结论 [设计笔记#降级机制][幻觉文档#不存在的节]。"
    got = extract_fact_candidates(mixed, allowed_sources=allowed)
    check("混合引用保留但只记真实来源",
          len(got) == 1 and got[0][1] == "设计笔记#降级机制", str(got))

    # chunk_id 三段式引用照样校验（取前两段对白名单）
    with_chunk = "- 带 chunk_id 的三段式引用 [设计笔记#降级机制#7]。"
    got = extract_fact_candidates(with_chunk, allowed_sources=allowed)
    check("三段式引用按 (doc, section) 校验通过", len(got) == 1, str(got))

    # 白名单为空集 = 本轮没检索 → 任何引用都不算确认
    got = extract_fact_candidates(answer, allowed_sources=set())
    check("本轮零检索 → 全部拒收", got == [], str(got))

    # 不传白名单 = 旧行为（向后兼容，三条都抽）
    got_legacy = extract_fact_candidates(answer)
    check("不传白名单 = 现行为（向后兼容）", len(got_legacy) == 3, str(got_legacy))


# ============================================================
# S14 — 契约贯通：图 e2e 里编造引用进不了 store
# ============================================================
def test_s14_fact_contract_e2e():
    print("\nS14 契约贯通：本轮真实检索来源才进 store")
    import nodes
    import memory as memory_pkg
    from memory.manager import MemoryManager
    from memory import long_term as lt
    from memory.ltm_store import NS_FACTS
    from graph import build_graph
    from config import RECURSION_LIMIT

    store = fresh_store()
    tmp = Path(tempfile.mkdtemp())
    mgr = MemoryManager(persist_dir=tmp, autoload=False)

    real_call, real_get, real_exec = nodes.call_model, memory_pkg.get_memory, nodes.execute_tool
    try:
        memory_pkg.get_memory = lambda: mgr

        tool_resp = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[SimpleNamespace(
                id="call_0",
                function=SimpleNamespace(name="retrieve_documents",
                                         arguments='{"query": "降级机制"}'),
            )]),
            finish_reason="tool_calls",
        )])
        answer = ("- 降级在连续失败 2 次后触发 [设计笔记#降级机制]。\n"
                  "- 这条来源是模型编造的 [幻觉文档#不存在的节]。")
        stop_resp = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer, tool_calls=None),
            finish_reason="stop",
        )])
        responses = iter([tool_resp, stop_resp])
        nodes.call_model = lambda msgs: next(responses)
        nodes.execute_tool = lambda name, args: {"results": [
            {"doc": "设计笔记", "section": "降级机制", "chunk_id": 3,
             "text": "连续失败 2 次触发降级", "score": 0.9},
        ]}

        graph = build_graph(store=store)
        cfg = {"configurable": {"thread_id": "s14", "use_memory": True},
               "recursion_limit": RECURSION_LIMIT}
        graph.invoke({"user_message": "帮我查一下降级机制的触发条件。"}, cfg)

        items = lt._list_all(store, NS_FACTS)
        facts = [it.value["fact"] for it in items]
        check("真实来源的事实进了 store",
              any("降级在连续失败" in f for f in facts), str(facts))
        check("编造来源的句子被拒收（踩坑 #3 收口）",
              not any("编造" in f for f in facts), str(facts))
        check("恰好 1 条事实", lt.counts(store)["facts"] == 1,
              str(lt.counts(store)))
    finally:
        nodes.call_model = real_call
        memory_pkg.get_memory = real_get
        nodes.execute_tool = real_exec


# ============================================================
# S10 — 装配窗口含 SYSTEM_PROMPT（v4.0 既有 bug 的回归测试）
# ============================================================
def test_s10_window_includes_system_prompt():
    print("\nS10 记忆开启时窗口仍含段 1 SYSTEM_PROMPT")
    import nodes
    from config import SYSTEM_PROMPT
    from langchain_core.messages import SystemMessage, HumanMessage

    # 模拟记忆开启时第二个问题的 messages 历史：
    # [问题1: sys, u1, a1] + [问题2: sys, prefs(sys), facts(sys), u2]
    from langchain_core.messages import AIMessage
    history = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="第一问"),
        AIMessage(content="第一答"),
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content="[用户偏好]\n- style: 先结论后引用"),
        SystemMessage(content="[相关长期事实（来自历史对话）]\n- 某事实"),
        HumanMessage(content="第二问"),
    ]
    window = nodes._window_messages(history)
    contents = [getattr(m, "content", "") for m in window]
    check("窗口起点 = 本问题块的段 1（SYSTEM_PROMPT）",
          contents[0] == SYSTEM_PROMPT and len(window) == 4,
          f"window len={len(window)}, first={contents[0][:30]!r}")
    check("窗口含偏好与事实段",
          any("[用户偏好]" in c for c in contents)
          and any("[相关长期事实" in c for c in contents))
    check("窗口不含上一问题的消息", all("第一" not in c for c in contents))


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    tests = [
        test_s1_recall_order,
        test_s2_min_score,
        test_s3_namespace_isolation,
        test_s4_preferences,
        test_s5_topics,
        test_s6_embed_economy,
        test_s7_assembler,
        test_s8_manager,
        test_s9_graph_e2e,
        test_s10_window_includes_system_prompt,
        test_s11_empty_answer_not_recorded,
        test_s12_memory_records_post_review_answer,
        test_s13_extractor_source_whitelist,
        test_s14_fact_contract_e2e,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  💥 {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  💥 {type(e).__name__}: {e}")

    print("\n" + "=" * 50)
    print(f"  {PASS} checks passed, {failed} test(s) failed")
    print("=" * 50)
    sys.exit(1 if failed else 0)
