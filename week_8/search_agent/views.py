"""
views.py — v6.0 worker 视图投影（隔离的实现形态，设计草稿 §四/§六 + 桩测 E5）

E5 实测的一刀（本周唯一真·新机制）：LangGraph **不分区**——每个 worker 节点函数物理上收到的是
**全 state dict**；worker 越界读不属于它的字段（reviewer 私有 review_verdict / best_draft 等）
**不报错、只静默串台**，没有任何框架级信号。所以"上下文隔离"不是框架给的护栏、是**约定**：
每个 worker 喂给 LLM 的只能是**显式投影函数的结果**、绝不能是 `state` 本身。

这三个 *_view 就是那三个投影（设计 §三 角色 I/O 契约的 in 端）：
  researcher_view(state) → {subtask_query, boundary}              （只看 task_description）
  writer_view(state)     → {findings, outline, review_notes}      （不看 draft_score/review_verdict）
  reviewer_view(state)   → {draft, rubric}                        （不看 findings；各看各的）

与第七周 E7"executor 看局部靠每子任务重产 SYSTEM_PROMPT 锚"同构——隔离不是天生的，是每次刻意裁出来的。

`isolation=False`（对照组）时，投影退化成 `dict(state)`——把全 state 倒给 worker，复现 E5 那种
"writer 越界读到 reviewer 私有"的串台。worker 节点据 `proj.keys()` 记录可见集（trace `_*_visible`），
隔离开时可见集 = 设计视图、关时含 reviewer 私有键。
"""

import json

from config import REVIEW_RUBRIC


# ============================================================
# 三个 worker 的投影函数（喂 LLM 的只能是它们的返回值，不能是 state）
# ============================================================

def researcher_view(state: dict, isolation: bool = True) -> dict:
    """researcher 只看：当前研究子任务 query（来自 task_description.objective）+ boundary（别查什么、归谁）。"""
    if not isolation:
        return dict(state)                      # 对照组：全 state 倒给 worker（leaky）
    td = state.get("task_description", {}) or {}
    return {
        "subtask_query": td.get("objective", "") or state.get("user_message", ""),
        "boundary": td.get("boundary", ""),
    }


def writer_view(state: dict, isolation: bool = True) -> dict:
    """writer 只看：findings（researcher 压缩回传）+ outline（task_description.output_format）+
    返修时的 review_notes。**不读 step_results / draft_score / review_verdict / best_draft / 全 state**。
    review_notes 是 writer↔reviewer 合法的返修回传通道（不是越界），其余 reviewer 产出都不该到 writer。"""
    if not isolation:
        return dict(state)
    return {
        "findings": state.get("findings", []),
        "outline": (state.get("task_description", {}) or {}).get("output_format", ""),
        "review_notes": state.get("review_notes", ""),   # 返修时非空，初稿时空
    }


def reviewer_view(state: dict, isolation: bool = True) -> dict:
    """reviewer 只看：draft（当前待审稿）+ rubric（二元判据）。**不读 findings**（各看各的，E5）。"""
    if not isolation:
        return dict(state)
    return {
        "draft": state.get("draft", ""),
        "rubric": list(REVIEW_RUBRIC),
    }


def visible_keys(projection: dict) -> list:
    """worker 实际"看到"的键集 = 投影的键（trace 用，对应桩测 E5 的 _*_visible）。
    隔离开 → 设计视图的窄键集；隔离关（投影=dict(state)）→ 含 reviewer 私有键的全 state。"""
    return sorted(projection.keys())


# ============================================================
# 把投影渲染成喂 LLM 的 user 内容（worker 节点据此建 prompt）
# ============================================================

def _render_findings(findings: list) -> str:
    if not findings:
        return "（暂无研究要点）"
    lines = []
    for i, f in enumerate(findings, 1):
        cites = "".join(f"[{c}]" for c in (f.get("citations") or []))
        status = f.get("status", "ok")
        tag = "" if status == "ok" else f"（{status}）"
        lines.append(f"{i}. {f.get('point', '')} {cites}{tag}")
    return "\n".join(lines)


def render_writer_input(projection: dict) -> str:
    """writer 的 user 段：研究要点 findings + 产出格式 outline +（返修时）上一轮评审意见。"""
    parts = ["# 研究要点（findings，请据此组织初稿，保留每条的 [doc#section] 引用）",
             _render_findings(projection.get("findings", []))]
    outline = projection.get("outline", "")
    if outline:
        parts.append(f"\n# 期望产出格式\n{outline}")
    notes = projection.get("review_notes", "")
    if notes:
        from config import REVIEW_FEEDBACK_PREFIX
        parts.append("\n" + REVIEW_FEEDBACK_PREFIX + notes)
    return "\n".join(parts)


def render_reviewer_input(projection: dict) -> str:
    """reviewer 的 user 段：待审初稿 + 二元判据（逐条勾）。"""
    rubric = "\n".join(f"  {i}. {item}" for i, item in enumerate(projection.get("rubric", []), 1))
    draft = projection.get("draft", "") or "（空稿）"
    return f"# 待审初稿\n{draft}\n\n# 评审判据（逐条判 是/否）\n{rubric}\n\n请按 VERDICT/SCORE/NOTES 三行输出裁决。"
