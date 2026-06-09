"""
工具使用检查 — 规则引擎 v3.0

两个核心函数：
  should_have_searched(message)   — 该不该联网搜索（v2.0 沿用）
  should_have_retrieved(message)  — 该不该查本地库（v3.0 新增）

设计原则：
  - 宁可误报也不漏报；漏报的代价是错答 / 编造
  - 检索纠正比搜索纠正更"贵"（多调一次 embedding），所以 should_have_retrieved
    的规则更收敛：只在明显涉及"本项目内部内容"时才触发
"""

import re


# ============================================================
# v3.0 新增：should_have_retrieved
# ============================================================

def should_have_retrieved(message: str) -> bool:
    """
    判断问题是否涉及"本项目内部知识"，需要先查本地库。

    触发场景（满足任一）：
      1. 明确指向本地资料：包含"笔记""复盘""设计""文档""我的""我们的""上次""之前"等
      2. 明确指向项目内部产物：search_agent、Agent Loop、第N周、tool 设计规范、
         记忆系统、RAG 设计、规则引擎、纠正注入、降级等
      3. 主语为"你"且涉及该项目自身能力："你是怎么做的"等

    不触发：纯外部知识、闲聊、数学、创作
    """
    msg = message.lower().strip()
    original = message.strip()

    # === 内部产物关键词（命中即触发） ===
    project_keywords = [
        # 项目自身工件
        "search_agent", "search-agent",
        "agent loop", "agent_loop", "agent-loop",
        "tool calling", "工具调用",
        "纠正注入", "降级注入", "降级指令", "fallback",
        "黑名单",
        # 周复盘
        "第一周", "第二周", "第三周", "第四周", "第五周", "第六周",
        "week 1", "week 2", "week 3", "week 4", "week 5", "week 6",
        "周复盘", "周笔记", "学习复盘",
        # 本项目设计概念
        "记忆系统", "memory system",
        "上下文装配", "装配策略", "装配顺序",
        "should_have_searched", "should_have_retrieved",
        "rule engine", "规则引擎",
        # RAG（本项目语境）— 配合下方"本地指向词"才稳
        # 单纯问"什么是 RAG" 不该走本地，所以这里不加 rag 关键词
    ]
    if any(kw in msg for kw in project_keywords):
        return True

    # === 本地指向词（命中即触发） ===
    local_pointers = [
        "我的笔记", "我们的笔记", "本地", "本项目",
        "我之前", "我们之前", "上次", "上回",
        "我的设计", "我们的设计", "我们的实现",
        "复盘里", "笔记里", "文档里", "设计里",
        "我做的", "我们做的",
        # 英文
        "my notes", "our notes", "our design", "our implementation",
        "previously", "earlier", "last time",
        "in my doc", "in the doc",
    ]
    if any(p in msg for p in local_pointers):
        return True

    # === 自指询问（询问 Agent 自身能力） ===
    self_ref_patterns = [
        r"你是怎么[实做]",
        r"你的实现",
        r"你怎么[处判决]",
        r"你的设计",
        r"how do you (handle|implement|decide)",
    ]
    if any(re.search(p, msg) for p in self_ref_patterns):
        return True

    return False


# ============================================================
# v2.0 沿用：should_have_searched
# ============================================================

def should_have_searched(message: str) -> bool:
    """判断问题是否需要联网搜索。v2.0 实现，原样保留。"""
    msg = message.lower().strip()

    # 排除规则
    creative_patterns = [
        r"写[一首个篇段]", r"创作", r"编[一个]", r"起[一个]*名",
        r"write\s+(a|me|an)", r"create\s+(a|an)", r"compose",
        r"翻译", r"translate", r"帮我润色", r"改写",
    ]
    if any(re.search(p, msg) for p in creative_patterns):
        return False

    math_patterns = [
        r"\d+\s*[\+\-\*\/\×\÷]\s*\d+", r"计算", r"算一下", r"等于多少",
        r"calculate", r"how much is \d+", r"求解", r"证明",
    ]
    if any(re.search(p, msg) for p in math_patterns):
        return False

    chat_patterns = [
        r"^(你好|hi|hello|hey|嗨|哈喽)",
        r"^(谢谢|thanks|thank you|感谢)",
        r"^(再见|bye|goodbye|拜拜)",
        r"^你是谁", r"^who are you",
        r"^你能做什么", r"^你叫什么",
    ]
    if any(re.search(p, msg) for p in chat_patterns):
        return False

    code_patterns = [
        r"写[一个]*代码", r"写[一个]*程序", r"写[一个]*脚本", r"写[一个]*函数",
        r"write\s+(code|a\s+function|a\s+script|a\s+program)",
        r"帮我实现", r"用\s*python", r"用\s*java\b",
    ]
    if any(re.search(p, msg) for p in code_patterns):
        return False

    # 搜索规则
    time_keywords = [
        "最新", "最近", "今年", "去年", "现在", "目前", "当前",
        "2023", "2024", "2025", "2026",
        "latest", "recent", "current", "new", "now", "today",
        "谁获得", "谁赢", "发布了", "上线了", "更新了",
    ]
    if any(kw in msg for kw in time_keywords):
        return True

    concept_patterns = [
        r"是什么", r"什么是", r"介绍[一下]*", r"解释[一下]*",
        r"what\s+is", r"what\s+are", r"explain",
        r"tell\s+me\s+about", r"how\s+does.*work", r"what\s+does.*mean",
    ]
    if any(re.search(p, msg) for p in concept_patterns):
        return True

    compare_keywords = [
        "对比", "比较", "区别", "不同", "优缺点",
        "哪个好", "推荐", "评价", "评测",
        "vs", "versus", "compare", "comparison",
        "difference", "which is better", "pros and cons",
    ]
    if any(kw in msg for kw in compare_keywords):
        return True

    original = message.strip()
    if re.search(r"[A-Z][a-z]+[A-Z]", original):
        return True
    if re.search(r"\b[A-Z]{2,}\b", original):
        return True
    if re.search(r"\b\w+[\-\.]\d", original):
        return True

    stripped = msg.replace("?", "").replace("？", "").strip()
    if len(msg) < 30 and len(stripped) >= 3 and ("?" in msg or "？" in msg):
        return True

    factual_patterns = [
        r"谁[是在]", r"在哪", r"多少", r"什么时候",
        r"when\s+did", r"where\s+is", r"how\s+many", r"how\s+much",
    ]
    if any(re.search(p, msg) for p in factual_patterns):
        return True

    return False
