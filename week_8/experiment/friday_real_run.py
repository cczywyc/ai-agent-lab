"""
friday_real_run.py — 第八周·周五真实跑（v6.0 supervisor 多 Agent 策略层验证）

桩测（experiment/e1–e7）离线验"框架接线"，本脚本接**真实 qwen 模型**压三条不同
路径的技术写作题，instrument 桩测锁不住的**策略层**信号：
  - per-round review 轨迹（每轮 verdict / score / parse 成功 / notes 长度）
  - 收口路径（accept 早退 vs 达上限 best-so-far）
  - worker 可见集 _*_visible（真实跑里隔离也只露设计视图、没串台）
  - reviewer 三行格式 VERDICT/SCORE/NOTES 的 parse 成功率（摔点③）
  - findings 丰度 / 600 字截断（writer 会不会被饿着，摔点②）
  - token 成本账（多 Agent 串接的代价，分角色）

三条题（压不同路径，对应今天的实验目标）：
  A = 易打回题  → 压打回循环 + Reflexion 返修 + best-so-far（盯摔点①：reviewer 太严/太松）
  B = 易 skip 题 → 压 skip-and-advance + 防御收口（含一条 week_8 不在库的子任务必 skip）
  C = 第七周 5 子任务"第二周到第六周演进" → 纵向对比 v5.0 机械拼接 vs v6.0 writer 成稿

跑法：
  ../../.venv/bin/python friday_real_run.py A     # 单条
  ../../.venv/bin/python friday_real_run.py all    # 三条
输出：friday_run_<T>.json（结构化）+ friday_drafts/<T>_*.md（draft / 交付全文，供人读）
"""

import sys
import os
import re
import json
import time
import logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
SA = HERE.parent / "search_agent"
sys.path.insert(0, str(SA))

import config as settings  # noqa: E402
import nodes               # noqa: E402
from graph import build_graph  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("friday")

# 真实跑当天的现实约束：qwen3.7-max-2026-05-17 免费额度在跑完 A/B/C + 纵向对比后耗尽（403
# AllocationQuota.FreeTierOnly）。用 FRIDAY_MODEL 环境变量切到尚有残量的旧模型跑 D / 复跑，
# 并把实际所用模型记进每条 rec（provenance：A/B/C=max，D/复跑=plus，跨题对比需带这个 caveat）。
_MODEL_OVERRIDE = os.getenv("FRIDAY_MODEL")
if _MODEL_OVERRIDE:
    settings.MODEL = _MODEL_OVERRIDE
    nodes.MODEL = _MODEL_OVERRIDE
    log.info(f"MODEL overridden -> {_MODEL_OVERRIDE}")

# ============================================================
# 三条压不同路径的写作题（语料 = 已重建的 weeks 2–7；week_8 故意不在库）
# ============================================================
TOPICS = {
    "A": {
        "id": "A", "path": "打回循环 + Reflexion + best-so-far",
        "query": "写一篇技术评估：我们项目第七周的 planner-executor-critic 外循环设计，"
                 "相比第六周的单步 LangGraph 控制流，工程上解决了什么、又引入了哪些新成本？"
                 "请给出明确的取舍判断和改进建议。",
        "why": "weeks 6/7 可检索（writer 有料），但'取舍判断/新成本/改进建议'诱发未接地论断 → reviewer 应判 reject。",
    },
    "B": {
        "id": "B", "path": "skip-and-advance + 防御收口",
        "query": "写一篇技术对比短文：我们项目第四五周的 RAG 检索设计，"
                 "和第八周多 Agent 里 researcher 的检索职责改造，各自怎么设计、有何取舍？",
        "why": "week_4&5 RAG 可检索（ok findings），但 week_8 不在库 → 该子任务必 skip；验 ok_findings>0 时 writer 仍能写。",
    },
    "C": {
        "id": "C", "path": "纵向对比 v5.0↔v6.0（5 子任务长链）",
        "query": "写一篇技术综述：我们项目从第二周到第六周，Agent 是怎么一步步演进的？"
                 "（工具调用 → Agent Loop → RAG → 记忆系统 → LangGraph 控制流）",
        "why": "weeks 2–6 全可检索、5 子任务长链；与 v5.0 同题对比'降一层复用'的成稿差异。",
    },
    "D": {
        "id": "D", "path": "强制 skip-and-advance（含杜撰模块必查不到）",
        "query": "写一篇技术对比：我们项目第六周的 LangGraph 控制流设计，"
                 "与我们项目里『量子化状态检查点压缩器（QSCC）』模块的设计取舍。",
        "why": "Topic B 的 week_8 子任务被相邻 week_7 内容救活、没 skip 成；QSCC 是杜撰模块、"
               "本地与联网都查不到 → 该子任务必 escalate→skip；week_6 部分给 ok findings → 验"
               "replan_count 正交计数 + ok_findings>0 时 writer 仍能写（防御式部分收口）。",
    },
}

# ============================================================
# Instrument：包装 5 个模型调用，记 token（分角色）+ reviewer per-round 轨迹
# ============================================================
TOK = {}                 # role -> {calls, prompt, completion, total}
REVIEW_TRACE = []        # 每轮 reviewer 调用一条
_CUR_ROLE = {"r": "?"}

_orig_create = settings.client.chat.completions.create


def _create_wrap(*a, **k):
    resp = _orig_create(*a, **k)
    u = getattr(resp, "usage", None)
    d = TOK.setdefault(_CUR_ROLE["r"], {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    d["calls"] += 1
    if u:
        d["prompt"] += getattr(u, "prompt_tokens", 0) or 0
        d["completion"] += getattr(u, "completion_tokens", 0) or 0
        d["total"] += getattr(u, "total_tokens", 0) or 0
    return resp


settings.client.chat.completions.create = _create_wrap


def _tag(role, fn):
    def wrapped(*a, **k):
        prev = _CUR_ROLE["r"]
        _CUR_ROLE["r"] = role
        try:
            return fn(*a, **k)
        finally:
            _CUR_ROLE["r"] = prev
    return wrapped


def _strict_format_ok(raw: str) -> bool:
    """reviewer 是否 follow 三行格式：VERDICT / SCORE / NOTES 三个标签都可解析（摔点③）。"""
    t = raw or ""
    has_v = bool(re.search(r"verdict[:：]\s*(accept|reject)", t, re.I))
    has_s = bool(re.search(r"score[:：]\s*([01](?:\.\d+)?|0?\.\d+)", t, re.I))
    has_n = bool(re.search(r"notes[:：]", t, re.I))
    return has_v and has_s and has_n


def _reviewer_tap(fn):
    inner = _tag("reviewer", fn)

    def wrapped(msgs):
        raw = inner(msgs)
        REVIEW_TRACE.append({
            "round": len(REVIEW_TRACE) + 1,
            "verdict": nodes._parse_review_verdict(raw),
            "score": nodes._parse_review_score(raw),
            "parse_ok": _strict_format_ok(raw),
            "notes_len": len(nodes._parse_review_notes(raw)),
            "raw_head": (raw or "")[:240],
        })
        return raw
    return wrapped


nodes.call_model = _tag("researcher", nodes.call_model)
nodes.call_supervisor_model = _tag("supervisor", nodes.call_supervisor_model)
nodes.call_critic_model = _tag("critic", nodes.call_critic_model)
nodes.call_writer_model = _tag("writer", nodes.call_writer_model)
nodes.call_reviewer_model = _reviewer_tap(nodes.call_reviewer_model)


# ============================================================
# 跑一条题 + 抓策略层信号
# ============================================================
def run_topic(key: str) -> dict:
    spec = TOPICS[key]
    TOK.clear()
    REVIEW_TRACE.clear()
    graph = build_graph()
    log.info(f"=== Topic {key} ({spec['path']}) ===")
    log.info(f"主题：{spec['query']}")

    t0 = time.time()
    state = graph.invoke({"user_message": spec["query"]},
                         {"configurable": {"thread_id": f"friday-{key}", "use_memory": False},
                          "recursion_limit": settings.RECURSION_LIMIT})
    dur = round(time.time() - t0, 1)

    plan = state.get("plan", [])
    findings = state.get("findings", [])
    ok_f = [f for f in findings if f.get("status") == "ok"]
    finding_lens = [len((f.get("point") or "")) for f in ok_f]
    best = state.get("best_draft", {}) or {}
    draft = state.get("draft", "")
    answer = state.get("answer", "")

    # 收口路径
    verdict = state.get("review_verdict", "")
    if verdict == "accept":
        closure = "accept（早退）"
    elif state.get("review_count", 0) >= settings.MAX_REVIEW:
        closure = "best-so-far（达上限收口）"
    elif not ok_f or not draft:
        closure = "defensive（无有效 finding/稿，占位收口）"
    else:
        closure = f"other（verdict={verdict!r}）"

    # 隔离断言（真实跑里也只露设计视图）
    iso = {
        "researcher": state.get("_researcher_visible", []),
        "writer": state.get("_writer_visible", []),
        "reviewer": state.get("_reviewer_visible", []),
    }
    iso_expected = {
        "researcher": ["boundary", "subtask_query"],
        "writer": ["findings", "outline", "review_notes"],
        "reviewer": ["draft", "rubric"],
    }
    iso_clean = all(iso.get(r) == exp for r, exp in iso_expected.items())

    tok_total = sum(d["total"] for d in TOK.values())
    parse_ok_n = sum(1 for r in REVIEW_TRACE if r["parse_ok"])

    rec = {
        "topic": key, "path": spec["path"], "query": spec["query"], "why": spec["why"],
        "model": nodes.MODEL,
        "duration_s": dur,
        "plan": [{"id": s.get("id"), "query": s.get("query"), "status": s.get("status")} for s in plan],
        "n_subtasks": len(plan),
        "n_findings": len(findings),
        "n_ok_findings": len(ok_f),
        "n_skipped_findings": sum(1 for f in findings if f.get("status") == "skipped"),
        "finding_point_lens": finding_lens,
        "finding_truncated_600": sum(1 for L in finding_lens if L >= 600),
        "has_draft": bool(draft), "draft_len": len(draft),
        "review_verdict": verdict,
        "review_count": state.get("review_count", 0),
        "replan_count": state.get("replan_count", 0),
        "best_score": best.get("score"),
        "termination_reason": state.get("termination_reason", ""),
        "closure": closure,
        "review_trace": list(REVIEW_TRACE),
        "review_rounds": len(REVIEW_TRACE),
        "review_parse_ok": parse_ok_n,
        "review_parse_total": len(REVIEW_TRACE),
        "isolation_visible": iso,
        "isolation_clean": iso_clean,
        "tokens_by_role": dict(TOK),
        "tokens_total": tok_total,
        "answer_len": len(answer),
    }

    # 存 draft / 交付全文供人读（纵向对比 + draft 质量评估）
    dd = HERE / "friday_drafts"
    dd.mkdir(exist_ok=True)
    (dd / f"{key}_v6_draft.md").write_text(draft or "(空)", encoding="utf-8")
    (dd / f"{key}_v6_delivery.md").write_text(answer or "(空)", encoding="utf-8")

    _print_summary(rec)
    out = HERE / f"friday_run_{key}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"已存 {out.name} + friday_drafts/{key}_v6_draft.md")
    return rec


def _print_summary(r: dict):
    print("\n" + "=" * 70)
    print(f"  Topic {r['topic']} — {r['path']}")
    print("=" * 70)
    print("  研究计划：")
    for s in r["plan"]:
        mark = {"done": "✓", "skipped": "⤬", "pending": "·"}.get(s["status"], "·")
        print(f"    {mark} {s['id'] + 1}. {s['query']}")
    print(f"  findings: {r['n_findings']}（ok {r['n_ok_findings']} / skip {r['n_skipped_findings']}）"
          f" | point 长度={r['finding_point_lens']} | 触顶600={r['finding_truncated_600']}")
    print(f"  draft: {r['draft_len']} 字 | replan={r['replan_count']} | review_count={r['review_count']}/{settings.MAX_REVIEW}")
    print(f"  收口: {r['closure']} | best_score={r['best_score']} | 终止={r['termination_reason']}")
    print(f"  review 轨迹（{r['review_rounds']} 轮，parse_ok {r['review_parse_ok']}/{r['review_parse_total']}）：")
    for t in r["review_trace"]:
        print(f"    轮{t['round']}: verdict={t['verdict']} score={t['score']} "
              f"parse_ok={t['parse_ok']} notes={t['notes_len']}字")
    print(f"  隔离 clean={r['isolation_clean']} | 可见集={r['isolation_visible']}")
    print(f"  token（分角色）: " + ", ".join(f"{k}={v['total']}(×{v['calls']})" for k, v in r["tokens_by_role"].items()))
    print(f"  token 合计: {r['tokens_total']} | 耗时 {r['duration_s']}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    sel = (sys.argv[1] if len(sys.argv) > 1 else "all").upper()
    keys = list(TOPICS) if sel == "ALL" else [sel]
    allrecs = {}
    for k in keys:
        allrecs[k] = run_topic(k)
    if len(keys) > 1:
        agg = {
            "topics": keys,
            "total_tokens": sum(r["tokens_total"] for r in allrecs.values()),
            "total_reject_rounds": sum(sum(1 for t in r["review_trace"] if t["verdict"] == "reject")
                                       for r in allrecs.values()),
            "total_review_rounds": sum(r["review_rounds"] for r in allrecs.values()),
            "parse_ok": sum(r["review_parse_ok"] for r in allrecs.values()),
            "parse_total": sum(r["review_parse_total"] for r in allrecs.values()),
            "isolation_all_clean": all(r["isolation_clean"] for r in allrecs.values()),
        }
        print("\n#### 汇总 ####")
        print(json.dumps(agg, ensure_ascii=False, indent=2))
        (HERE / "friday_run_summary.json").write_text(
            json.dumps({"aggregate": agg, "runs": allrecs}, ensure_ascii=False, indent=2),
            encoding="utf-8")
