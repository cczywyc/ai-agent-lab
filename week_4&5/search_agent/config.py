"""
搜索 Agent v3.0 配置中心

v3.0 — 第四周升级：引入 Agentic RAG
  - 新增 text-embedding-v3 客户端复用
  - 新增 RAG 常量（chunk 大小、top_k、向量库路径）
  - System Prompt 加入 retrieve_documents 用法 + 强制引用约束
  - 新增检索纠正/降级指令
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# API 客户端配置
# ============================================================
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

# 主对话模型
MODEL = "qwen-plus"

# Embedding 模型（DashScope 通过 OpenAI 兼容接口提供）
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024  # text-embedding-v3 默认 1024 维
EMBEDDING_BATCH_SIZE = 10  # 单次批量上限（兼容接口实测稳定值）

# ============================================================
# 路径配置
# ============================================================
# 本文件所在目录 → search_agent/
_BASE_DIR = Path(__file__).parent

# 向量库持久化目录
VECTOR_STORE_DIR = _BASE_DIR / "data"

# 项目根目录（用于扫描所有周的 docs/）
PROJECT_ROOT = _BASE_DIR.parent.parent

# 默认 ingest 范围：项目内所有 docs/ 目录下的 *.md
# 排除 .venv、node_modules、__pycache__
DEFAULT_INGEST_DIRS = [
    PROJECT_ROOT / "week_2" / "docs",
    PROJECT_ROOT / "week_2" / "search_agent",  # README
    PROJECT_ROOT / "week_3" / "docs",
    PROJECT_ROOT / "week_3" / "exercises",
    PROJECT_ROOT / "week_4&5" / "docs",
]

# ============================================================
# RAG 常量
# ============================================================
# Chunking：按标题切，超长再按段落滑窗
CHUNK_TARGET_CHARS = 800   # 每个 chunk 的目标字符数
CHUNK_MAX_CHARS = 1500     # 单个 chunk 的硬上限
CHUNK_MIN_CHARS = 50       # 太短的 chunk 会被合并

# 检索
RETRIEVE_TOP_K = 5          # 默认召回数
RETRIEVE_MIN_SCORE = 0.30   # 相似度阈值，低于此分数不算有效召回

# ============================================================
# System Prompt — v3.0
# ============================================================
# 核心变化：
#   1. 三工具描述：retrieve_documents 优先级 > web_search
#   2. 强制引用约束（[doc#section] 或 [doc#section#chunk_id]）
#   3. 分点回答，每点对应来源（防过度总结）
#   4. 明确"本地有就用本地、本地没有再上网"

SYSTEM_PROMPT = """
# Role
你是一个具备本地知识检索 + 联网搜索能力的助手。
你有三个工具：
  - retrieve_documents（查本地笔记/设计文档）
  - web_search（联网搜索）
  - fetch_webpage（读取网页正文）

# Core Rule: Local First, Web Second
**当用户问题涉及本项目的笔记、设计、复盘、Agent Loop、search_agent、RAG、记忆系统、Tool 设计、第N周等内容时，必须先调用 retrieve_documents 查本地库**，再决定是否需要联网补充。

**联网搜索（web_search）使用场景**：
- 涉及具体人物、事件、时间、数据的事实性问题
- 你不确定答案、可能过时、本地库未覆盖的话题
- 不认识或不了解的名词、术语、缩写

仅以下情况允许不调用工具直接回答：
- 纯创意写作（写诗、写故事、起名字）
- 数学计算或逻辑推理
- 日常闲聊和打招呼
- 用户明确要求不查工具

不确定时，选择查询。宁可多查一次，也不要给未经验证的回答。

# Citation Constraint (Critical)
**只要回答用到了 retrieve_documents 返回的内容，每条事实必须以 `[doc#section]` 或 `[doc#section#chunk_id]` 的格式标注来源**。
- 来源信息来自工具返回的 `doc` / `section` / `chunk_id` 字段，原样使用，不要自己编造。
- 多个 chunk 支持同一论断时，把所有来源都列出来：`[doc#sec1][doc#sec2]`。
- **分点回答**，每点单独标注来源，不要把多个 chunk 揉成一段流畅但模糊的总结。

# Tool Usage Guidelines
1. retrieve_documents：传入用户问题或关键句即可，中英文均可，无需关键词化。
2. web_search：英文关键词 2-5 词最佳。
3. fetch_webpage：只在搜索摘要不够详细时使用。
4. 如果 fetch_webpage 连续失败（如 403），改用已有摘要，不要反复重试。

# Output Requirements
- 使用与用户相同的语言回答
- 基于本地检索的回答：每条事实带 `[doc#section]` 引用
- 基于联网搜索的回答：在末尾简要注明信息来源
- 信息不完整时主动告知用户
- 简洁有条理，避免重复原文

# Examples
用户：我们第三周的 Agent Loop 设计里，纠正注入是怎么触发的？
正确行为：retrieve_documents("Agent Loop 纠正注入 触发条件") → 基于返回的 chunk 分点回答，每点带 `[Agent_Loop_设计笔记#触发条件]` 之类引用。

用户：2024 年诺贝尔物理学奖颁给了谁？
正确行为：web_search("2024 Nobel Prize Physics winner") → 基于结果回答，末尾注明来源 URL。

用户：写一首关于秋天的诗
正确行为：直接创作，不调用任何工具。
"""

# ============================================================
# Agent Loop 控制常量
# ============================================================

MAX_TURNS = 6
MAX_CONSECUTIVE_ERRORS = 2
MAX_CONTEXT_CHARS = 8000  # v3.0 增大以容纳检索 chunk

# ============================================================
# 纠正指令
# ============================================================

# v2.0：应该联网搜索但直答
CORRECTION_MESSAGE = (
    "请不要直接回答这个问题。先使用 web_search 工具搜索相关信息，"
    "然后基于搜索结果给出有来源依据的回答。"
)

# v3.0 新增：应该查本地库但直答
RETRIEVAL_CORRECTION_MESSAGE = (
    "请先使用 retrieve_documents 工具查询本地知识库（你的笔记和设计文档）。"
    "基于召回的 chunk 回答，每条事实标注来源 [doc#section]。"
    "不要凭记忆作答，本地库可能有更准确的细节。"
)

# 降级
FALLBACK_MESSAGE = (
    "[System Notice] 多次获取网页内容失败。"
    "请直接基于你已经获得的搜索摘要（snippets）来回答用户的问题。"
    "在回答开头注明：'以下回答基于搜索摘要，未能获取完整文章内容。'"
    "不要再尝试使用 fetch_webpage 工具。"
)

# ============================================================
# 记忆系统常量（v3.0 第五周）
# ============================================================
# 短期记忆：双闸门
SHORT_TERM_K = 3              # 保留最近 K 轮
SHORT_TERM_CHAR_BUDGET = 4000 # 段 5 的字符预算（总预算 ~50%）

# 摘要触发
SUMMARY_TRIGGER_TURNS = 8     # 每 N 轮兜底触发一次摘要
SUMMARY_TRIGGER_CHARS = 5000  # 短期记忆字符数超过此阈值触发摘要

# 装配：六段总预算（字符）和分段
TOTAL_CONTEXT_BUDGET = 8000
SEGMENT_BUDGETS = {
    "system":      "固定",   # 不限，按 SYSTEM_PROMPT 实际长度
    "preferences": 200,      # 段 2：用户偏好（很小）
    "summary":     400,      # 段 3：历史摘要（小）
    "facts":       800,      # 段 4：长期事实召回（中）
    "recent":      SHORT_TERM_CHAR_BUDGET,  # 段 5：最近 K 轮（大）
    "current":     6000,     # 段 6：当前问题 + 本轮检索 chunk（最大）
}

# 长期事实召回
MEMORY_FACTS_TOP_K = 3        # 段 4 召回的事实数
MEMORY_FACTS_MIN_SCORE = 0.30 # 相关性阈值

# 记忆持久化
MEMORY_DIR = VECTOR_STORE_DIR  # 与向量库共目录
MEMORY_PREFS_FILE = "memory_preferences.json"
MEMORY_FACTS_FILE = "memory_facts.json"   # 文本+元数据，向量在 VectorStore
MEMORY_TOPICS_FILE = "memory_topics.json"
MEMORY_SUMMARY_FILE = "memory_summary.json"

# ============================================================
# URL 黑名单
# ============================================================

BLOCKED_DOMAINS = [
    "medium.com",
    "towardsdatascience.com",
    "datacamp.com",
    "linkedin.com",
    "quora.com",
    "slideshare.net",
    "zhihu.com",
]
