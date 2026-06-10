"""
搜索 Agent v4.0 配置中心

v4.0 — 第六周升级：控制流迁移到 LangGraph（StateGraph + checkpointer）
  - 新增 INTERRUPT_ENABLED（决策 F：human_review 开关，默认关）
  - 新增 RECURSION_LIMIT（框架兜底，正常终止靠 turn_count 闸门——E4 实证）
  - 其余常量沿 v3.0 不变（决策 A：本周不动存储层与 RAG）

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
MODEL = "qwen3.7-plus-2026-05-26"

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
    PROJECT_ROOT / "week_6" / "docs",          # v4.0：第六周设计草稿
    PROJECT_ROOT / "week_6" / "experiment",    # v4.0：E1-E5 实验结论
    PROJECT_ROOT / "week_7" / "docs",          # v5.0：第七周设计草稿 / 职责边界
    PROJECT_ROOT / "week_7" / "experiment",    # v5.0：E1-E7 实验结论
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

MAX_TURNS = 6              # 内层（单子任务）tool-use 上限（v4.2 原样）
MAX_CONSECUTIVE_ERRORS = 2
MAX_CONTEXT_CHARS = 8000  # v3.0 增大以容纳检索 chunk

# ============================================================
# 外循环控制常量（v5.0 第七周 · planner-executor-critic）
# ============================================================
# 双闸门各兜一个正交维度（E4 实证、非冗余）：
#   MAX_STEPS  兜"前进步数走太长"（plan 子任务上限；职责边界 §5 建议 5–8）
#   MAX_REPLAN 兜"原地反复 re-plan 不前进"（决策 D；E4：escalate 恒压下它先收口）
# MAX_RETRY 是业务层重试额度（per-subtask，critic 驱动"换措辞重做该步"），与
#   传输层 empty_retries（agent 节点内、不进拓扑）分开计数（决策 B / E5）。
MAX_STEPS = 6
MAX_REPLAN = 2
MAX_RETRY = 2

# critic 引用"接地下限"（v0.5 实测调整）：单步引用里能溯源到本步真实召回的比例 ≥ 此值才放行
# 进 LLM 质量裁决；低于此值判 retry。不再要求"全部引用精确命中"——真实跑暴露：模型写富报告
# 会引很多子节/相关节（多于本步召回集），全命中是奢望；过严会把合法步全误杀成 retry→级联。
# 长期记忆持久化侧仍由 extract_fact_candidates(allowed_sources) 严格白名单兜底（S13/S14）。
CITATION_MIN_GROUNDING = 0.5

# ============================================================
# LangGraph 控制常量（v4.0 第六周）
# ============================================================

# 决策 F：human_review 节点的 interrupt 开关。
# 默认关——测试和批量评测保证可复现（沿用"记忆可插拔、测试不带记忆"的思路）。
# 交互模式可用 --review 打开。节点内动态读取（import config 后取属性），
# 所以运行时改 config.INTERRUPT_ENABLED 即时生效。
INTERRUPT_ENABLED = False

# 框架递归上限（兜底，不该靠它正常终止——E4/E6 实证它的终止方式是抛异常）。
# v5.0 外循环把 super-step 数抬高一个量级：MAX_STEPS×(1+MAX_RETRY) 次执行 ×
# 每次最坏 MAX_TURNS 轮（agent+tools=2/轮）+ planner/critic/step_init，最坏约 300。
# E6 实测 LangGraph 1.2.4 默认 recursion_limit=10007（远超有界任务）、收口靠显式
# 闸门而非它——这里放宽到 500 作生成式兜底（仍只兜底，正常终止靠 MAX_STEPS/REPLAN/TURNS）。
RECURSION_LIMIT = 500

# ============================================================
# 占位符回答（v4.2 第七周前重构：写入方与判读方共用同一份名单）
# ============================================================
# 写入方：nodes.agent（[错误]）、nodes.finalize（[达到最大轮次] / [模型返回空回答]）
# 判读方：main.run_test 的 has_answer 判据、nodes.update_memory 的记忆跳过逻辑
# 教训（06-05 复跑 Case 5）：判据漏一个前缀，空回答就成了 PASS——
# 新增占位符必须进这份名单，判据和跳过逻辑自动跟上。

PLACEHOLDER_MAX_TURNS = "[达到最大轮次]"
PLACEHOLDER_ERROR = "[错误]"
PLACEHOLDER_EMPTY = "[模型返回空回答]"

PLACEHOLDER_PREFIXES = (
    PLACEHOLDER_MAX_TURNS,
    PLACEHOLDER_ERROR,
    PLACEHOLDER_EMPTY,
)


def is_placeholder_answer(answer: str) -> bool:
    """answer 为空或以任一占位符前缀开头 → 不是给用户的有效回答。"""
    return not answer or any(answer.startswith(p) for p in PLACEHOLDER_PREFIXES)


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
# 外循环提示词（v5.0 第七周 · planner / critic）
# ============================================================
# planner 看全局（原问题 + plan + 各步结论摘要），不碰检索原文（职责边界 §2/§5）。
# 拆解粒度：一个子任务 = 一个可独立检索 + 总结的子问题，不要细到"一次检索调用"。
# 输出**严格 JSON 字符串数组**，便于 nodes._parse_plan 解析（解析失败有兜底）。

PLANNER_PROMPT = """\
你是技术调研 Agent 的**规划器（planner）**。你的职责是把用户的调研问题拆成有序的子任务，
不亲自检索、不调工具——只决定"做什么、分几步"。

拆解原则：
- 一个子任务 = 一个可以独立检索 + 总结的子问题（不要细到"调一次工具"那种粒度）。
- 子任务串行推进，后面的可依赖前面的结论。
- 控制数量：最多 {max_steps} 个子任务；问题简单时 1–2 个即可，不要硬凑。
- 每个子任务用一句自然语言问题描述，能直接交给执行器去检索。

**只输出一个 JSON 字符串数组**，每个元素是一个子任务问题，不要任何额外解释或 Markdown 代码块。
示例：["子任务问题一", "子任务问题二", "子任务问题三"]
"""

# re-plan：某子任务被 critic escalate（方向不对/重试用尽）后，重新规划"从当前步起的剩余子任务"。
PLANNER_REPLAN_PROMPT = """\
你是技术调研 Agent 的规划器。当前子任务方向不对、已被打回（escalate）。
请结合"原问题 + 已完成步的结论摘要 + 被打回的子任务"，**重新规划从当前步起的剩余子任务**
（可以换一种表述、拆得更细、或调整方向）。同样的输出约束：

**只输出一个 JSON 字符串数组**（剩余子任务，最多 {max_steps} 个），不要任何额外解释。
"""

# critic 审单步：比执行器输出更严，默认怀疑"全部 accept"（职责边界 §8）。
# 输出首行 VERDICT: accept|retry|escalate，其后可给一行 FEEDBACK 供 executor 重做参考。
CRITIC_PROMPT = """\
你是技术调研 Agent 的**评审器（critic）**。你只审**当前这一步**执行器的产出质量，
不改计划、不重写结果——只发裁决信号。判据要比执行器更严，默认怀疑、不要轻易放行。

裁决三选一：
- accept：结论扣题、有依据（基于检索/搜索而非凭空）、引用合法（只引本步真实召回的来源）。
- retry：本步可救——空回答/跑题/引用对不上来源/总结过度但方向对——值得换措辞重做该步。
- escalate：本步方向本身不对（子任务问得不对、本地与联网都查不到），需要规划器改计划。

输出格式（严格）：
第一行：VERDICT: accept    （或 retry / escalate）
第二行（可选）：FEEDBACK: 一句话告诉执行器该怎么改（retry 时务必给）
"""

# critic 判 retry 时，executor 重做该步时看到的提示前缀（assemble 带进窗口）。
RETRY_FEEDBACK_PREFIX = "[上一次产出被评审打回，请针对性改进] "

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
