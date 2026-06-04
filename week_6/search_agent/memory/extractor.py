"""
规则抽取 — 零成本写入通道

设计文档原话：
  写入双通道：规则抽取（主题/事实候选，零成本）+ 模型显式（偏好）
  ❌ 不要每轮让模型抽取记忆（成本/延迟，第三周假阳性教训的记忆版）

本模块全部用 regex/字符串规则，不调模型。

三个出口：
  extract_topics(message)         → list[str]，主题词
  extract_preference(message)     → dict | None，{key, value}
  extract_fact_candidates(answer) → list[(fact_text, source)]
"""

from __future__ import annotations

import re


# ============================================================
# 主题：项目相关关键词（命中即累加计数）
# ============================================================

TOPIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 主题词 → 匹配正则（大小写不敏感）
    ("rag", re.compile(r"\brag\b|检索增强|retrieve_documents", re.I)),
    ("agent_loop", re.compile(r"agent[_ ]?loop|agent 循环|agent循环", re.I)),
    ("memory", re.compile(r"记忆系统|memory system|长期记忆|短期记忆", re.I)),
    ("embedding", re.compile(r"embedding|向量|余弦", re.I)),
    ("chunking", re.compile(r"chunking|切块|分块", re.I)),
    ("tool_calling", re.compile(r"tool[_ ]?calling|工具调用|工具设计", re.I)),
    ("system_prompt", re.compile(r"system[_ ]?prompt|系统提示", re.I)),
    ("rule_engine", re.compile(r"规则引擎|should_have_", re.I)),
    ("fallback", re.compile(r"fallback|降级", re.I)),
    ("trace", re.compile(r"\btrace\b|追踪|可审计", re.I)),
    ("citation", re.compile(r"引用约束|\[doc#section\]|citation", re.I)),
    ("vector_store", re.compile(r"向量库|向量存储|vector store|faiss|chroma", re.I)),
]


def extract_topics(message: str) -> list[str]:
    """从消息里提取主题词。返回去重列表。"""
    found: list[str] = []
    seen: set[str] = set()
    for name, pat in TOPIC_PATTERNS:
        if name in seen:
            continue
        if pat.search(message):
            found.append(name)
            seen.add(name)
    return found


# ============================================================
# 偏好：显式信号触发
# ============================================================
# 设计：用规则而非模型——只在用户明确说"记住/请记得/我喜欢/I prefer"等
# 信号时才进入"偏好"通道，否则忽略。模型显式 = 用户显式 + 规则识别信号。

PREFERENCE_TRIGGERS = re.compile(
    r"(?P<verb>记住|请记得|请记住|记一下|帮我记|"
    r"我喜欢|我倾向于|我希望你|"
    r"remember that|i prefer|please remember|note that)"
    r"[，,：:\s]*(?P<value>.{2,200}?)(?:[。.！!？?]|$)",
    re.IGNORECASE,
)


def extract_preference(message: str) -> dict | None:
    """
    识别偏好信号，返回 {key, value} 或 None。

    key 由 value 的前 30 字简化得到（去标点空白）；
    重复存储时上层用 LongTermMemory.set_preference(key, value) 覆盖。
    """
    m = PREFERENCE_TRIGGERS.search(message)
    if not m:
        return None
    value = m.group("value").strip().strip("：:，,。.！!？?")
    if len(value) < 2:
        return None
    # key：取 value 前缀做粗略归一
    key = re.sub(r"\s+", "_", value[:30])
    return {"key": key, "value": value, "verb": m.group("verb")}


# ============================================================
# 事实候选：从助手回答里提取带 [doc#section] 引用的事实
# ============================================================
# 策略：只在回答带 [doc#section] 形式的引用时才抽取，引用即事实"已确认"。
# 没有引用的回答可能是模型自由发挥，不进入长期事实。

# 匹配 [doc#section] 或 [doc#section#chunk_id] 引用
CITATION_RE = re.compile(r"\[([^\[\]\n]+?#[^\[\]\n]+?)\]")

# 一句话边界（中英文标点）
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。.！!？?])\s*")


def extract_fact_candidates(answer: str) -> list[tuple[str, str]]:
    """
    从助手回答里提取（事实, 来源）对。

    规则：含 [doc#section] 引用的句子才算"已确认事实"，
    多个引用拼成 `src1; src2`。
    """
    if not answer or "[" not in answer:
        return []

    # 同时支持点 / 项目符号开头的"一行一点"格式
    candidates: list[tuple[str, str]] = []

    # 1) 行级切（针对 markdown 列表风格回答）
    line_units = [line.strip() for line in answer.splitlines() if line.strip()]
    # 2) 句子级切（针对长段回答）作为补充
    sentence_units = SENTENCE_SPLIT_RE.split(answer)

    units = line_units if any("[" in l and "#" in l for l in line_units) else sentence_units

    seen_facts: set[str] = set()
    for unit in units:
        sources = CITATION_RE.findall(unit)
        if not sources:
            continue
        # 去掉行内的引用括号后作为 fact 文本
        clean = CITATION_RE.sub("", unit).strip()
        # 去前导的 markdown bullet / 数字编号
        clean = re.sub(r"^[\-\*\d\.、\s]+", "", clean).strip()
        if len(clean) < 8:
            continue
        if clean in seen_facts:
            continue
        seen_facts.add(clean)
        candidates.append((clean, "; ".join(sources)))

    return candidates
