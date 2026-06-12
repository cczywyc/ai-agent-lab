"""
第八周桩测共享底座：最小 supervisor 多 Agent 图 + 桩 worker。
被 e1_*.py ~ e7_*.py 各自 import，单独跑。零 API。栈：LangGraph 1.2.4。

设计依据：《第八周设计草稿 v0.1》决策 A–H。
- 除 messages 外一律替换语义、不上 reducer（v5.0 纪律）
- 节点干活、边做决策
- worker 不接 LLM，按 cfg 返回预设结果，并记录"实际读到的视图"（验隔离）
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

MAX_REVIEW = 2          # 决策 E：reviewer 打回上限
MAX_TURNS = 5           # researcher 内层（v5.0 原样）


class S(TypedDict, total=False):
    active_worker: str
    task_description: dict      # 委派契约四要素
    findings: list              # researcher 回传
    draft: str                  # writer 当前稿
    draft_score: float          # reviewer 给当前稿评分
    review_verdict: str         # '', 'accept', 'reject' —— 必须声明进 schema
    review_notes: str
    review_count: int           # 打回计数（独立于 replan_count）
    replan_count: int           # 子任务跳过计数（独立于 review_count）
    best_draft: dict            # {'draft','score'} best-so-far
    worker_result: dict         # worker 压缩回传信封
    turn_count: int             # researcher 内层
    _researcher_visible: list   # E5：researcher 实际读到的键
    _writer_visible: list
    _reviewer_visible: list
    _entered_researcher_turns: list  # E7
    _n_reviews: int             # E4：reviewer 被调次数


# --- supervisor：构造 task_description 四要素，路由交给边 ---
def make_supervisor(cfg):
    def supervisor(state: S) -> dict:
        if cfg.get("route_bug"):                      # E1 对照组：写歪的路由（恒派 writer、无视阶段）
            return {"active_worker": "writer", "task_description": {}}
        if not state.get("findings"):
            td = {"objective": "检索 LangGraph 多 Agent 资料并压缩要点",
                  "output_format": "findings: [{point, citations}]",
                  "tools_hint": ["retrieve_documents"]}
            if not cfg.get("drop_boundary"):          # 对照组：故意漏第 4 要素
                td["boundary"] = "只查多 Agent，别碰单 Agent loop（归 week_3）"
            return {"active_worker": "researcher", "task_description": td}
        if not state.get("draft"):
            td = {"objective": "把 findings 写成初稿", "output_format": "draft(含引用)",
                  "tools_hint": [], "boundary": "不自己检索、不自评"}
            return {"active_worker": "writer", "task_description": td}
        v = state.get("review_verdict", "")
        if v == "":
            td = {"objective": "审稿出 verdict+notes", "output_format": "verdict|notes",
                  "tools_hint": [], "boundary": "不改稿、只给意见"}
            return {"active_worker": "reviewer", "task_description": td}
        return {"active_worker": v}
    return supervisor


def route_supervisor(state: S) -> str:
    aw = state.get("active_worker")
    if aw in ("researcher", "writer", "reviewer"):
        return aw
    return "finalize"


def make_route_after_reviewer(cfg):
    gate_field = "replan_count" if cfg.get("shared_counter") else "review_count"

    def route(state: S) -> str:
        v = state.get("review_verdict", "")
        if v == "accept":
            return "finalize"                          # 早退
        if cfg.get("gate_on", True) and state.get(gate_field, 0) >= MAX_REVIEW:
            return "finalize"                          # 闸门收口 → best-so-far
        return "writer"                                # 返修
    return route


# --- 桩 worker ---
def make_researcher(cfg):
    def researcher(state: S) -> dict:
        visible = ["task_description"] if cfg.get("isolation_on", True) else list(state.keys())
        entered = state.get("_entered_researcher_turns", []) + [state.get("turn_count", 99)]
        out = {"_researcher_visible": visible, "turn_count": 3,
               "_entered_researcher_turns": entered,
               "findings": [{"point": "supervisor=中心编排", "citations": ["doc#A"]}],
               "worker_result": {"kind": "findings", "n": 1}}
        if cfg.get("researcher_skips_once") and not state.get("findings"):
            out["replan_count"] = state.get("replan_count", 0) + 1   # skip-and-advance
        return out
    return researcher


def make_writer(cfg):
    def writer(state: S) -> dict:
        visible = ["findings", "review_notes"] if cfg.get("isolation_on", True) else list(state.keys())
        n = state.get("review_count", 0)
        return {"draft": f"draft-v{n}", "review_verdict": "",
                "_writer_visible": visible, "worker_result": {"kind": "draft", "v": n}}
    return writer


def make_reviewer(cfg):
    scores = cfg.get("scores", [0.6, 0.8, 0.5])

    def reviewer(state: S) -> dict:
        visible = ["draft"] if cfg.get("isolation_on", True) else list(state.keys())
        rc = state.get("review_count", 0)
        score = scores[min(rc, len(scores) - 1)]
        draft = state.get("draft", "")
        best = state.get("best_draft", {"draft": "", "score": -1.0})
        if score > best["score"]:
            best = {"draft": draft, "score": score}
        accept_at = cfg.get("accept_at_review", None)
        verdict = "accept" if (accept_at is not None and rc >= accept_at) else "reject"
        out = {"review_verdict": verdict, "review_notes": f"notes@{rc}: 引用不全",
               "draft_score": score, "best_draft": best, "_reviewer_visible": visible,
               "_n_reviews": state.get("_n_reviews", 0) + 1,
               "worker_result": {"kind": "verdict", "v": verdict}}
        if verdict == "reject":
            if cfg.get("shared_counter"):
                out["replan_count"] = state.get("replan_count", 0) + 1
            else:
                out["review_count"] = rc + 1
        return out
    return reviewer


def make_finalize(cfg):
    def finalize(state: S) -> dict:
        if cfg.get("best_so_far", True):
            chosen = state.get("best_draft", {"draft": state.get("draft"), "score": state.get("draft_score")})
        else:                                          # 对照组：取最新稿
            chosen = {"draft": state.get("draft"), "score": state.get("draft_score")}
        return {"worker_result": {"kind": "final", "chosen": chosen}}
    return finalize


def build(cfg=None):
    cfg = cfg or {}
    g = StateGraph(S)
    g.add_node("supervisor", make_supervisor(cfg))
    g.add_node("researcher", make_researcher(cfg))
    g.add_node("writer", make_writer(cfg))
    g.add_node("reviewer", make_reviewer(cfg))
    g.add_node("finalize", make_finalize(cfg))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_supervisor,
                            {"researcher": "researcher", "writer": "writer",
                             "reviewer": "reviewer", "finalize": "finalize"})
    g.add_edge("researcher", "supervisor")
    g.add_edge("writer", "supervisor")
    g.add_conditional_edges("reviewer", make_route_after_reviewer(cfg),
                            {"writer": "writer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run(cfg, init=None, recursion_limit=50):
    init = init or {"review_count": 0, "replan_count": 0}
    return build(cfg).invoke(init, {"recursion_limit": recursion_limit})


def report(eid, name, ok, detail):
    """统一打印 + 退出码，供各 eN 文件单独跑用。"""
    import sys
    print(f"[{eid}] {name}: {'PASS' if ok else 'FAIL'}")
    print(f"      {detail}")
    sys.exit(0 if ok else 1)
