"""
第五周复盘 — 综合实验测试套件

目的：用真实千问 API（text-embedding-v3 + qwen-plus）跑出可复现的指标，
为《第五周学习复盘》的所有【待填】提供真实数据。

实验清单（每个独立、可单独失败、增量落盘）：
  E1  embedding 链路       —— 维度 / 归一化 / 批量分批 / 单批延迟
  E2a 检索准确率评测       —— 15 条标注查询的 hit@1/3/5 + top-score
  E2b chunking A/B         —— 按标题切（现有库）vs 按字数切（新建）召回对比
  E2c namespace 隔离       —— docs 与 memory_facts 两个向量库互不污染
  E3  上下文优先级实验     —— 旧摘要 vs 本轮新检索冲突，qwen 偏向哪个
  E4  记忆单元 + 摘要触发   —— 短期双闸门裁剪 / evict / 摘要 / 事实语义召回
  E5  多轮记忆 demo        —— 4 轮真实对话，捕获每轮六段装配指标
  E6  路由 + 引用准确率     —— 8 用例路由正确率 + RAG 用例引用正确率
  E7  代码规模             —— v2.0(week_3) vs v3.0 文件/行数/常量（离线）

用法：
  python experiments.py                 # 跑全部
  python experiments.py --stages E1,E2a # 只跑指定实验
  输出：experiment_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np

# 项目内扁平 import（与 main.py 同 cwd）
from config import (
    EMBEDDING_DIM, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS,
    RETRIEVE_MIN_SCORE, SYSTEM_PROMPT, MODEL, client,
    SHORT_TERM_K, SHORT_TERM_CHAR_BUDGET,
)

REPORT_PATH = Path("experiment_report.json")
REPORT: dict = {
    "title": "第五周复盘综合实验报告",
    "model": MODEL,
    "embedding_model": "text-embedding-v3",
    "results": {},
}

# 跨进程增量：载入已有报告，新跑的 stage 只覆盖自己那段
if REPORT_PATH.exists():
    try:
        _prev = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        REPORT["results"].update(_prev.get("results", {}))
    except Exception:
        pass


def _save():
    REPORT["finished_at"] = datetime.now().isoformat()
    REPORT_PATH.write_text(
        json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _stage(name: str, fn):
    print(f"\n{'='*64}\n  {name}\n{'='*64}")
    t0 = time.time()
    try:
        out = fn()
        out["_ok"] = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        out = {"_ok": False, "error": repr(e)}
    out["_seconds"] = round(time.time() - t0, 1)
    REPORT["results"][name] = out
    _save()
    print(f"  → {name} 完成 ({out['_seconds']}s) ok={out['_ok']}")
    return out


# ============================================================
# 标注检索评测集（答案确实存在于现有 6 个文档中）
# ============================================================
EVAL_QUERIES = [
    ("Agent Loop 连续失败计数是怎么重置的？", "Agent_Loop_设计笔记"),
    ("should_have_searched 规则引擎的上限在哪里？", "Agent_Loop_设计笔记"),
    ("决策审查 检查机制 假阳性问题怎么修", "Agent_Loop_设计笔记"),
    ("Agent Loop 三个控制点的设计取舍", "Agent_Loop_设计笔记"),
    ("json_schema strict 模式实验结论", "实验结论"),
    ("json_object 模式实验观察到什么", "实验结论"),
    ("第二周 工具调用 单 Agent 雏形", "第二周学习复盘"),
    ("fetch_webpage 反爬 403 Medium 失败", "第二周学习复盘"),
    ("第三周 结构化输出 提示词工程 复盘", "第三周学习复盘"),
    ("RAG chunking 按标题切分策略怎么定的", "第四五周整合设计_RAG与记忆系统"),
    ("embedding 与向量存储 设计选型", "第四五周整合设计_RAG与记忆系统"),
    ("记忆系统 短期 长期 分层设计", "第四五周整合设计_RAG与记忆系统"),
    ("上下文装配 六段顺序 预算", "第四五周整合设计_RAG与记忆系统"),
    ("为什么 RAG 先做 记忆后做", "第四五周整合设计_RAG与记忆系统"),
    ("Week 2 Demo 功能 工具设计 已知问题", "README"),
]


# ============================================================
# E1 — embedding 链路
# ============================================================
def exp_E1():
    from rag.embed import embed_texts, embed_query

    sample = ["第一条文本", "second english text", "向量检索 RAG"]
    t0 = time.time()
    vecs = embed_texts(sample)
    single_batch_ms = int((time.time() - t0) * 1000)

    norms = np.linalg.norm(vecs, axis=1)
    # 批量分批：23 条 > 单批上限 10，应自动切 3 批
    big = [f"批量分批测试文本 number {i}" for i in range(23)]
    big_vecs = embed_texts(big)

    q = embed_query("查询向量维度测试")

    res = {
        "返回维度": int(vecs.shape[1]),
        "维度符合EMBEDDING_DIM": int(vecs.shape[1]) == EMBEDDING_DIM,
        "是否L2归一化": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "归一化范数样本": [round(float(x), 5) for x in norms],
        "单批3条延迟ms": single_batch_ms,
        "批量23条返回数": int(big_vecs.shape[0]),
        "批量分批正确": int(big_vecs.shape[0]) == 23,
        "embed_query形状": list(q.shape),
    }
    for k, v in res.items():
        print(f"    {k}: {v}")
    return res


# ============================================================
# 公共：用现有 docs 向量库做检索评测
# ============================================================
def _eval_on_store(store, query_vecs, queries, top_k=5):
    """对一组 (query, expected_doc) 跑 hit@k。query_vecs 与 queries 对齐。"""
    hit1 = hit3 = hit5 = 0
    top_scores = []
    per = []
    for (q, exp_doc), qv in zip(queries, query_vecs):
        hits = store.search(qv, top_k=top_k)  # [(meta, score)]
        docs_ranked = [m.get("doc") for m, _ in hits]
        scores = [s for _, s in hits]
        top_scores.append(scores[0] if scores else 0.0)
        r1 = exp_doc in docs_ranked[:1]
        r3 = exp_doc in docs_ranked[:3]
        r5 = exp_doc in docs_ranked[:5]
        hit1 += r1; hit3 += r3; hit5 += r5
        per.append({
            "query": q, "expected": exp_doc,
            "top1_doc": docs_ranked[0] if docs_ranked else None,
            "top1_score": round(float(scores[0]), 4) if scores else None,
            "hit@1": r1, "hit@3": r3, "hit@5": r5,
        })
    n = len(queries)
    return {
        "n": n,
        "hit@1": f"{hit1}/{n} ({100*hit1/n:.0f}%)",
        "hit@3": f"{hit3}/{n} ({100*hit3/n:.0f}%)",
        "hit@5": f"{hit5}/{n} ({100*hit5/n:.0f}%)",
        "hit@1_raw": hit1, "hit@3_raw": hit3, "hit@5_raw": hit5,
        "mean_top1_score": round(float(np.mean(top_scores)), 4),
        "per_query": per,
    }


# 进程内缓存 query 向量，避免 E2a/E2b 重复 embed
_QVEC_CACHE: dict[str, np.ndarray] = {}


def _query_vecs():
    from rag.embed import embed_query
    out = []
    for q, _ in EVAL_QUERIES:
        if q not in _QVEC_CACHE:
            _QVEC_CACHE[q] = embed_query(q)
        out.append(_QVEC_CACHE[q])
    return out


# ============================================================
# E2a — 检索准确率（现有按标题切的库）
# ============================================================
def exp_E2a():
    from rag.store import VectorStore
    store = VectorStore(namespace="docs")
    assert store.load(), "docs 向量库不存在，先 python main.py --ingest"
    print(f"    库规模: {len(store)} chunks")
    qv = _query_vecs()
    res = _eval_on_store(store, qv, EVAL_QUERIES, top_k=5)
    res["store_size"] = len(store)
    res["min_score_threshold"] = RETRIEVE_MIN_SCORE
    print(f"    hit@1={res['hit@1']} hit@3={res['hit@3']} hit@5={res['hit@5']} "
          f"mean_top1={res['mean_top1_score']}")
    miss = [p for p in res["per_query"] if not p["hit@1"]]
    if miss:
        print("    hit@1 未命中:")
        for p in miss:
            print(f"      Q={p['query'][:26]} exp={p['expected']} got={p['top1_doc']}({p['top1_score']})")
    return res


# ============================================================
# E2b — chunking A/B：按标题切 vs 按字数切
# ============================================================
def _chunk_by_chars(path: Path, doc_name=None, size=400, overlap=80):
    """朴素 RAG 风格：忽略结构，按固定字数滑窗切。"""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", raw).strip()
    doc = doc_name or path.stem
    chunks = []
    i = cid = 0
    step = max(1, size - overlap)
    while i < len(text):
        sub = text[i:i + size]
        if len(sub) < 30:
            break
        chunks.append({
            "doc": doc, "section": "(char-window)",
            "chunk_id": cid, "text": sub, "path": str(path),
        })
        cid += 1
        i += step
    return chunks


def exp_E2b():
    from rag.embed import embed_texts
    from rag.store import VectorStore
    from rag.ingest import chunk_markdown_file

    # 复用现有 docs 库作为「按标题切」组（已嵌入，省钱）
    heading_store = VectorStore(namespace="docs")
    assert heading_store.load()

    # 源文件 = 现有库的 path 字段去重（保证 A/B 同源）
    from config import PROJECT_ROOT
    paths = sorted({m["path"] for m in heading_store.metadata})
    src_files = []
    for rel in paths:
        p = (PROJECT_ROOT / rel)
        if p.exists():
            src_files.append(p)
    print(f"    A/B 同源文件 {len(src_files)} 个")

    # 重新统计「按标题切」chunk 尺寸（用现有库 metadata）
    head_sizes = [len(m["text"]) for m in heading_store.metadata]

    # 「按字数切」组：新建临时库
    char_chunks = []
    for p in src_files:
        char_chunks.extend(_chunk_by_chars(p, size=400, overlap=80))
    char_sizes = [len(c["text"]) for c in char_chunks]
    print(f"    按标题切 {len(head_sizes)} chunks (avg {np.mean(head_sizes):.0f}字) | "
          f"按字数切 {len(char_chunks)} chunks (avg {np.mean(char_sizes):.0f}字)")

    tmp = Path(tempfile.mkdtemp())
    char_store = VectorStore(namespace="docs_charAB", persist_dir=tmp)
    print(f"    嵌入按字数切的 {len(char_chunks)} chunks ...")
    cvecs = embed_texts([c["text"] for c in char_chunks], show_progress=True)
    char_store.add(cvecs, char_chunks)

    qv = _query_vecs()
    head_eval = _eval_on_store(heading_store, qv, EVAL_QUERIES, top_k=5)
    char_eval = _eval_on_store(char_store, qv, EVAL_QUERIES, top_k=5)

    res = {
        "by_heading": {
            "chunks": len(head_sizes),
            "avg_chunk_chars": round(float(np.mean(head_sizes)), 0),
            "hit@1": head_eval["hit@1"], "hit@3": head_eval["hit@3"],
            "hit@5": head_eval["hit@5"],
            "mean_top1_score": head_eval["mean_top1_score"],
        },
        "by_chars": {
            "chunks": len(char_chunks),
            "avg_chunk_chars": round(float(np.mean(char_sizes)), 0),
            "window": "size=400 overlap=80",
            "hit@1": char_eval["hit@1"], "hit@3": char_eval["hit@3"],
            "hit@5": char_eval["hit@5"],
            "mean_top1_score": char_eval["mean_top1_score"],
        },
    }
    print(f"    [按标题] hit@1={res['by_heading']['hit@1']} "
          f"hit@3={res['by_heading']['hit@3']} top1={res['by_heading']['mean_top1_score']}")
    print(f"    [按字数] hit@1={res['by_chars']['hit@1']} "
          f"hit@3={res['by_chars']['hit@3']} top1={res['by_chars']['mean_top1_score']}")
    return res


# ============================================================
# E2c — namespace 隔离
# ============================================================
def exp_E2c():
    from rag.embed import embed_texts, embed_query
    from rag.store import VectorStore

    docs = VectorStore(namespace="docs"); docs.load()
    mem = VectorStore(namespace="memory_facts"); mem.load()

    # 文件层面隔离：两个 namespace 的持久化文件不同
    file_isolated = docs._vec_path != mem._vec_path and docs._meta_path != mem._meta_path

    # 内容层面隔离：往一个临时 docs 库塞入一条"毒"事实，
    # 证明它不会出现在 memory 库的检索里（反之亦然）。
    tmp = Path(tempfile.mkdtemp())
    poison_text = "ZZZ独有标记：松鼠在月球上用算盘烤披萨。"
    pv = embed_texts([poison_text])
    docs_t = VectorStore(namespace="docs", persist_dir=tmp)
    docs_t.add(pv, [{"doc": "POISON_DOC", "section": "x", "chunk_id": 0, "text": poison_text}])
    mem_t = VectorStore(namespace="memory_facts", persist_dir=tmp)
    fact_text = "YYY独有标记：用户偏好先结论后引用。"
    fv = embed_texts([fact_text])
    mem_t.add(fv, [{"fact": fact_text, "source": "turn:1", "turn": 1}])

    qv = embed_query(poison_text)
    docs_hit = docs_t.search(qv, top_k=1)
    mem_hit = mem_t.search(qv, top_k=1)
    # 用 poison 查询：docs_t 命中 poison；mem_t 只可能命中 fact（不同内容）
    docs_returns_poison = docs_hit and docs_hit[0][0].get("doc") == "POISON_DOC"
    mem_has_no_poison = all(m.get("text", m.get("fact", "")) != poison_text for m, _ in mem_hit)

    res = {
        "文件层隔离": bool(file_isolated),
        "docs持久化文件": Path(docs._vec_path).name,
        "memory持久化文件": Path(mem._vec_path).name,
        "现有docs库chunks": len(docs),
        "现有memory_facts条数": len(mem),
        "毒事实只进docs不进memory": bool(docs_returns_poison and mem_has_no_poison),
        "说明": "两个 namespace = 两套独立 .npy/.json 文件，按前缀分离，检索互不可见",
    }
    for k, v in res.items():
        print(f"    {k}: {v}")
    return res


# ============================================================
# E3 — 上下文优先级：旧摘要 vs 本轮新检索冲突
# ============================================================
def exp_E3():
    # 受控探针：手工拼六段式 messages，把"旧摘要"和"本轮新检索"放进同一上下文，
    # 二者给出冲突的数值，看 qwen 跟随哪个。
    # 两种条件：A 带显式"以最新为准"指令；B 中性指令（隔离指令 vs 自然倾向）。
    old_summary = "[历史对话摘要]\n之前确认过：本项目单个 chunk 的硬上限是 500 字符。"
    new_chunk = (
        "[本轮检索结果]\n"
        "[doc=config#section=RAG常量] 最新代码 config.py 明确：CHUNK_MAX_CHARS = 1500，"
        "即单个 chunk 的硬上限是 1500 字符（CHUNK_TARGET_CHARS=800 为目标值）。"
    )
    question = "本项目单个 chunk 的硬上限是多少字符？请只给数字并标注来源。"

    CONDS = {
        "A_显式以最新为准": "你必须严格依据提供的上下文回答，冲突时以本轮检索（最新代码）为准。每条事实标注来源。",
        "B_中性指令": "你是一个依据上下文回答的助手。每条事实标注来源。",
    }

    def run_cond(sys_prompt, n=3):
        trials = []; follow_new = 0
        for i in range(n):
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "system", "content": old_summary},
                {"role": "user", "content": f"{new_chunk}\n\n问题：{question}"},
            ]
            resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.3)
            ans = (resp.choices[0].message.content or "").strip()
            chose_new = ("1500" in ans) and ("500" not in ans.replace("1500", ""))
            chose_old = ("500" in ans.replace("1500", "")) and "1500" not in ans
            verdict = "新检索(1500)" if chose_new else ("旧摘要(500)" if chose_old else "含糊/两者")
            follow_new += chose_new
            trials.append({"trial": i + 1, "verdict": verdict, "answer": ans[:140]})
        return follow_new, n, trials

    res = {
        "场景": "段3旧摘要说500 vs 段6本轮检索说1500，看模型跟随哪个",
        "conditions": {},
    }
    for cond, sp in CONDS.items():
        fn, n, trials = run_cond(sp)
        res["conditions"][cond] = {
            "跟随新检索": f"{fn}/{n}", "trials": trials,
        }
        print(f"    [{cond}] 跟随新检索 {fn}/{n}: {[t['verdict'] for t in trials]}")
    return res


# ============================================================
# E4 — 记忆单元 + 摘要触发 + 事实语义召回
# ============================================================
class _FakeToolCall:
    def __init__(self, name, ok=True):
        self.tool_name = name; self.result_success = ok


class _FakeTurnTrace:
    def __init__(self, tcs): self.tool_calls = tcs


class _FakeTrace:
    def __init__(self, retrieved=False, searched=False, tool=None):
        self.retrieved = retrieved; self.searched = searched
        self.turns = [_FakeTurnTrace([_FakeToolCall(tool)] if tool else [])]


def exp_E4():
    from memory.manager import MemoryManager
    tmp = Path(tempfile.mkdtemp())
    mm = MemoryManager(persist_dir=tmp, autoload=False)

    # 注入若干轮，其中带 [doc#section] 引用的会被晋升为长期事实
    turns = [
        ("第三周连续失败怎么处理？",
         "连续失败计数器成功即重置 [Agent_Loop_设计笔记#控制点2]。",
         _FakeTrace(retrieved=True, tool="retrieve_documents")),
        ("请记住：以后回答先列结论再列引用。",
         "好的，已记住该偏好。", _FakeTrace()),
        ("chunking 按什么切？",
         "按 Markdown 标题切，超长再滑窗 [第四五周整合设计_RAG与记忆系统#RAG-1]。",
         _FakeTrace(retrieved=True, tool="retrieve_documents")),
        ("embedding 用哪个模型？",
         "用 text-embedding-v3，1024 维 [第四五周整合设计_RAG与记忆系统#RAG-2]。",
         _FakeTrace(retrieved=True, tool="retrieve_documents")),
        ("向量库用什么实现？",
         "用 numpy 余弦相似度，零依赖 [第四五周整合设计_RAG与记忆系统#RAG-2]。",
         _FakeTrace(retrieved=True, tool="retrieve_documents")),
    ]
    for u, a, tr in turns:
        mm.update_from_turn(u, a, tr)

    info = mm.info()
    # 短期是否被裁到 K 轮
    short_after = len(mm.short.turns)
    # 摘要是否产生
    summary = mm.summarizer.summary_text

    # 事实语义召回：用一个与"连续失败"相关的 query
    from rag.embed import embed_query
    qv = embed_query("Agent Loop 工具连续失败 重置")
    recalled = mm.long.recall_facts(qv, top_k=3)

    res = {
        "短期双闸门": {
            "K": SHORT_TERM_K, "字符预算": SHORT_TERM_CHAR_BUDGET,
            "注入轮数": len(turns), "evict后保留轮数": short_after,
            "裁剪到K正确": short_after == SHORT_TERM_K,
        },
        "偏好捕获": dict(mm.long.preferences),
        "晋升长期事实数": len(mm.long.facts),
        "主题Top5": mm.long.top_topics(5),
        "摘要已产生": bool(summary),
        "摘要内容": summary[:200],
        "事实语义召回": [
            {"fact": r["fact"][:50], "score": r.get("score")} for r in recalled
        ],
        "召回命中相关事实": any("重置" in r["fact"] or "连续失败" in r["fact"] for r in recalled),
    }
    print(f"    短期保留 {short_after} 轮 (K={SHORT_TERM_K}) | 事实 {len(mm.long.facts)} 条 | "
          f"偏好 {len(mm.long.preferences)} 条 | 摘要={'有' if summary else '无'}")
    print(f"    摘要: {summary[:120]}")
    print(f"    召回: {[r['fact'][:30] for r in recalled]}")
    return res


# ============================================================
# E5 — 多轮记忆 demo（真实对话，捕获每轮装配指标）
# ============================================================
DEMO_TURNS = [
    "我们第三周的 Agent Loop 是怎么处理工具连续失败的？",
    "请记住：以后回答涉及本地文档时，先列结论再列引用。",
    "刚才那个连续失败机制里，consecutive_errors 是怎么重置的？",
    "另外简单说一下，本项目的 RAG chunking 是按什么切的？",
]


def exp_E5():
    from agent import run_agent
    from memory.manager import MemoryManager
    tmp = Path(tempfile.mkdtemp())
    mm = MemoryManager(persist_dir=tmp, autoload=False)

    # 包裹 assemble_context 以捕获每轮装配报告
    reports = []
    orig = mm.assemble_context

    def wrapped(um, sp):
        msgs, rep = orig(um, sp)
        reports.append(rep)
        return msgs, rep
    mm.assemble_context = wrapped

    per_turn = []
    for i, q in enumerate(DEMO_TURNS, 1):
        print(f"    --- 第{i}轮: {q[:30]}")
        ans, tr = run_agent(q, memory=mm)
        rep = reports[-1] if reports else None
        per_turn.append({
            "turn": i, "query": q,
            "segments_present": rep.segments_present if rep else [],
            "segments_trimmed": rep.segments_trimmed if rep else [],
            "context_chars": rep.total_chars if rep else None,
            "facts_recalled": rep.facts_recalled if rep else 0,
            "trace": tr.summary(),
            "retrieved": tr.retrieved, "searched": tr.searched,
            "answer_preview": (ans or "")[:120],
        })
        print(f"        装配段={rep.segments_present if rep else None} "
              f"字符={rep.total_chars if rep else None} 事实召回={rep.facts_recalled if rep else 0}")
        print(f"        {tr.summary()}")

    info = mm.info()
    ctx_chars = [t["context_chars"] for t in per_turn if t["context_chars"]]
    res = {
        "轮数": len(DEMO_TURNS),
        "per_turn": per_turn,
        "平均装配上下文字符": round(float(np.mean(ctx_chars)), 0) if ctx_chars else None,
        "平均装配上下文token估算": round(float(np.mean(ctx_chars)) / 1.6, 0) if ctx_chars else None,
        "demo后记忆状态": info,
        "偏好是否在第2轮后生效": "回答风格" in str(info["preferences"]) or len(info["preferences"]) > 0,
    }
    print(f"    平均装配上下文 {res['平均装配上下文字符']} 字符 "
          f"(~{res['平均装配上下文token估算']} token) | demo后: {info}")
    return res


# ============================================================
# E6 — 路由 + 引用准确率（8 用例，无记忆）
# ============================================================
TEST_CASES = [
    {"id": 1, "query": "2024年诺贝尔物理学奖颁给了谁？", "expect_search": True, "expect_retrieve": False, "category": "factual_time"},
    {"id": 2, "query": "写一首关于春天的五言绝句", "expect_search": False, "expect_retrieve": False, "category": "creative"},
    {"id": 3, "query": "帮我算一下 234 × 567", "expect_search": False, "expect_retrieve": False, "category": "math"},
    {"id": 4, "query": "我们第三周的 Agent Loop 是怎么处理工具连续失败的？", "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},
    {"id": 5, "query": "在我们的设计里，should_have_searched 用了哪些规则？", "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},
    {"id": 6, "query": "本项目第四周的 RAG 是怎么决定 chunking 策略的？", "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},
    {"id": 7, "query": "我之前的笔记里，URL 黑名单是怎么用的？", "expect_search": False, "expect_retrieve": True, "category": "rag_internal"},
    {"id": 8, "query": "MCP 协议是什么？", "expect_search": True, "expect_retrieve": False, "category": "tech_concept"},
]

CITATION_RE = re.compile(r"\[([^\[\]\n]+?#[^\[\]\n]+?)\]")


def exp_E6():
    from agent import run_agent
    results = []
    passed = 0
    cite_total = cite_ok = 0
    for case in TEST_CASES:
        q = case["query"]
        print(f"    Case {case['id']}: {q[:30]}")
        ans, tr = run_agent(q)
        s_ok = tr.searched == case["expect_search"]
        r_ok = tr.retrieved == case["expect_retrieve"]
        has_ans = bool(ans) and not ans.startswith("[达到最大轮次]") and not ans.startswith("[错误]")
        case_pass = s_ok and r_ok and has_ans
        passed += case_pass

        # 引用正确率：仅对 RAG 用例。命中 = 回答含 [doc#...] 且 doc ∈ 召回 doc
        cite_info = None
        if case["expect_retrieve"]:
            cite_total += 1
            retrieved_docs = set()
            for t in tr.turns:
                for tc in t.tool_calls:
                    for ch in getattr(tc, "retrieved_chunks", []):
                        if ch.get("doc"):
                            retrieved_docs.add(ch["doc"])
            cited = CITATION_RE.findall(ans or "")
            cited_docs = {c.split("#")[0].strip() for c in cited}
            matched = cited_docs & retrieved_docs
            has_valid_cite = len(matched) > 0
            if has_valid_cite:
                cite_ok += 1
            cite_info = {
                "citations_found": len(cited),
                "cited_docs": sorted(cited_docs),
                "retrieved_docs": sorted(retrieved_docs),
                "valid_citation": has_valid_cite,
            }

        results.append({
            "id": case["id"], "query": q, "category": case["category"],
            "expect_search": case["expect_search"], "expect_retrieve": case["expect_retrieve"],
            "actual_search": tr.searched, "actual_retrieve": tr.retrieved,
            "passed": case_pass, "turns": tr.total_turns,
            "duration_ms": tr.total_duration_ms,
            "retrieval_correction": tr.retrieval_correction_triggered,
            "search_correction": tr.correction_triggered,
            "fallback": tr.fallback_triggered,
            "citation": cite_info,
            "answer_preview": (ans or "")[:160],
        })
        print(f"        路由 {'✓' if case_pass else '✗'} search={tr.searched} retrieve={tr.retrieved} "
              f"{tr.total_turns}轮 {tr.total_duration_ms}ms"
              + (f" 引用{'✓' if cite_info and cite_info['valid_citation'] else '✗'}" if cite_info else ""))

    n = len(TEST_CASES)
    res = {
        "total_cases": n,
        "passed": passed,
        "pass_rate": f"{passed}/{n} ({100*passed/n:.0f}%)",
        "avg_turns": round(sum(r["turns"] for r in results) / n, 2),
        "avg_duration_ms": round(sum(r["duration_ms"] for r in results) / n, 0),
        "citation_accuracy": f"{cite_ok}/{cite_total} ({100*cite_ok/cite_total:.0f}%)" if cite_total else "N/A",
        "search_correction_count": sum(r["search_correction"] for r in results),
        "retrieval_correction_count": sum(r["retrieval_correction"] for r in results),
        "fallback_count": sum(r["fallback"] for r in results),
        "results": results,
    }
    print(f"    通过率={res['pass_rate']} 引用正确率={res['citation_accuracy']} "
          f"平均{res['avg_turns']}轮 {res['avg_duration_ms']}ms")
    return res


# ============================================================
# E7 — 代码规模（离线）
# ============================================================
def exp_E7():
    from config import PROJECT_ROOT

    def scan(base: Path):
        files = [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
        lines = 0
        for p in files:
            lines += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
        return files, lines

    def count_consts(cfg: Path):
        if not cfg.exists():
            return 0
        txt = cfg.read_text(encoding="utf-8", errors="ignore")
        # 顶层 UPPER_CASE = ... 赋值
        return len(re.findall(r"(?m)^[A-Z][A-Z0-9_]+\s*=", txt))

    v2_base = PROJECT_ROOT / "week_3" / "search_agent"
    v3_base = PROJECT_ROOT / "week_4&5" / "search_agent"
    v2_files, v2_lines = scan(v2_base)
    v3_files, v3_lines = scan(v3_base)

    res = {
        "v2.0_week3": {
            "py_files": len(v2_files),
            "py_lines": v2_lines,
            "config_constants": count_consts(v2_base / "config.py"),
            "file_names": sorted(p.name for p in v2_files),
        },
        "v3.0_week45": {
            "py_files": len(v3_files),
            "py_lines": v3_lines,
            "config_constants": count_consts(v3_base / "config.py"),
            "new_modules": ["rag/(ingest,embed,store,retriever)",
                            "memory/(manager,short_term,long_term,assembler,summarizer,extractor)"],
        },
    }
    print(f"    v2.0: {res['v2.0_week3']['py_files']} 文件 / {v2_lines} 行 / "
          f"{res['v2.0_week3']['config_constants']} 常量")
    print(f"    v3.0: {res['v3.0_week45']['py_files']} 文件 / {v3_lines} 行 / "
          f"{res['v3.0_week45']['config_constants']} 常量")
    return res


# ============================================================
# 入口
# ============================================================
ALL_STAGES = {
    "E1": exp_E1, "E2a": exp_E2a, "E2b": exp_E2b, "E2c": exp_E2c,
    "E3": exp_E3, "E4": exp_E4, "E5": exp_E5, "E6": exp_E6, "E7": exp_E7,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=str, default=",".join(ALL_STAGES),
                    help="逗号分隔，如 E1,E2a")
    args = ap.parse_args()
    want = [s.strip() for s in args.stages.split(",") if s.strip()]

    print(f"运行实验: {want}")
    for s in want:
        if s not in ALL_STAGES:
            print(f"  跳过未知实验 {s}")
            continue
        _stage(s, ALL_STAGES[s])

    print(f"\n{'='*64}\n  报告已保存: {REPORT_PATH}\n{'='*64}")
    # 简表
    for name, r in REPORT["results"].items():
        print(f"  {name}: {'OK' if r.get('_ok') else 'FAIL'} ({r.get('_seconds')}s)")


if __name__ == "__main__":
    main()
