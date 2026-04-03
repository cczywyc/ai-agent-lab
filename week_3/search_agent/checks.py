"""
工具使用检查 — 规则引擎

核心函数 should_have_searched() 判断用户问题是否属于"应该先搜索再回答"的类型。

设计原则：
  - 宁可误报（不该搜索的被判为该搜索）也不要漏报
  - 误报的代价是多一次搜索（浪费一轮）
  - 漏报的代价是给出未经验证的答案（危险）
  - 因此规则偏向宽松匹配
"""

import re


def should_have_searched(message: str) -> bool:
    """
    判断用户的问题是否属于"应该先搜索再回答"的类型。

    Args:
        message: 用户的原始输入

    Returns:
        True = 应该搜索（如果模型跳过了工具，需要纠正）
        False = 不需要搜索（模型直接回答是合理的）
    """
    msg = message.lower().strip()

    # ============================================================
    # 排除规则（优先级最高）— 命中则返回 False
    # ============================================================

    # 排除 1：明确的创意写作请求
    creative_patterns = [
        r"写[一首个篇段]",
        r"创作",
        r"编[一个]",
        r"起[一个]*名",
        r"write\s+(a|me|an)",
        r"create\s+(a|an)",
        r"compose",
        r"翻译",
        r"translate",
        r"帮我润色",
        r"改写",
    ]
    if any(re.search(p, msg) for p in creative_patterns):
        return False

    # 排除 2：数学和逻辑
    math_patterns = [
        r"\d+\s*[\+\-\*\/\×\÷]\s*\d+",  # 算式
        r"计算",
        r"算一下",
        r"等于多少",
        r"calculate",
        r"how much is \d+",
        r"求解",
        r"证明",
    ]
    if any(re.search(p, msg) for p in math_patterns):
        return False

    # 排除 3：闲聊
    chat_patterns = [
        r"^(你好|hi|hello|hey|嗨|哈喽)",
        r"^(谢谢|thanks|thank you|感谢)",
        r"^(再见|bye|goodbye|拜拜)",
        r"^你是谁",
        r"^who are you",
        r"^你能做什么",
        r"^你叫什么",
    ]
    if any(re.search(p, msg) for p in chat_patterns):
        return False

    # 排除 4：写代码
    code_patterns = [
        r"写[一个]*代码",
        r"写[一个]*程序",
        r"写[一个]*脚本",
        r"写[一个]*函数",
        r"write\s+(code|a\s+function|a\s+script|a\s+program)",
        r"帮我实现",
        r"用\s*python",
        r"用\s*java\b",
    ]
    if any(re.search(p, msg) for p in code_patterns):
        return False

    # ============================================================
    # 搜索规则 — 命中任一则返回 True
    # ============================================================

    # 规则 1：包含时间指示词
    time_keywords = [
        "最新", "最近", "今年", "去年", "现在", "目前", "当前",
        "2023", "2024", "2025", "2026",
        "latest", "recent", "current", "new", "now", "today",
        "谁获得", "谁赢", "发布了", "上线了", "更新了",
    ]
    if any(kw in msg for kw in time_keywords):
        return True

    # 规则 2：概念/定义查询
    concept_patterns = [
        r"是什么",
        r"什么是",
        r"介绍[一下]*",
        r"解释[一下]*",
        r"what\s+is",
        r"what\s+are",
        r"explain",
        r"tell\s+me\s+about",
        r"how\s+does.*work",
        r"what\s+does.*mean",
    ]
    if any(re.search(p, msg) for p in concept_patterns):
        return True

    # 规则 3：对比/评价类
    compare_keywords = [
        "对比", "比较", "区别", "不同", "优缺点",
        "哪个好", "推荐", "评价", "评测",
        "vs", "versus", "compare", "comparison",
        "difference", "which is better", "pros and cons",
    ]
    if any(kw in msg for kw in compare_keywords):
        return True

    # 规则 4：包含可能是专有名词的 token
    # 检测驼峰命名（LangGraph）、全大写缩写（MCP）、带版本号（GPT-4）
    original = message.strip()  # 用原始大小写检测
    if re.search(r"[A-Z][a-z]+[A-Z]", original):  # 驼峰：LangGraph, CrewAI
        return True
    if re.search(r"\b[A-Z]{2,}\b", original):  # 全大写缩写：MCP, RAG, API
        return True
    if re.search(r"\b\w+[\-\.]\d", original):  # 带版本号：GPT-4, v2.0
        return True

    # 规则 5：问句但不是闲聊（短问题且含问号，至少 3 个有效字符）
    stripped = msg.replace("?", "").replace("？", "").strip()
    if len(msg) < 30 and len(stripped) >= 3 and ("?" in msg or "？" in msg):
        return True

    # 规则 6：包含"谁""哪里""多少"等疑问词 + 具体实体特征
    factual_patterns = [
        r"谁[是在]",
        r"在哪",
        r"多少",
        r"什么时候",
        r"when\s+did",
        r"where\s+is",
        r"how\s+many",
        r"how\s+much",
    ]
    if any(re.search(p, msg) for p in factual_patterns):
        return True

    # ============================================================
    # 默认：不需要搜索
    # ============================================================
    return False
