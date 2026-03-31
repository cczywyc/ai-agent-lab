"""
config.py — 客户端初始化与全局配置
====================================
集中管理 API 配置、模型选择、系统提示词。
修改模型或切换 API 提供商时只需要改这个文件。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========== API 客户端 ==========
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

# ========== 模型配置 ==========
MODEL_NAME = "qwen-plus"        # 日常开发用 qwen-plus（性价比高）
MAX_TURNS = 5                   # Agent Loop 最大轮次
MAX_SEARCH_RESULTS = 5          # 搜索默认返回条数
MAX_WEBPAGE_LENGTH = 3000       # 网页内容截断长度

# ========== System Prompt ==========
SYSTEM_PROMPT = """You are a research assistant that helps users find and summarize information from the web.

## Workflow
1. When the user asks a question requiring current information, first use web_search to find relevant results.
2. Review the search snippets. If they contain enough information to answer, summarize directly.
3. If the snippets are insufficient, use fetch_webpage on the most promising URL to get full content.
4. Synthesize your findings into a clear, structured answer.

## Rules
- Always cite source URLs when presenting factual information.
- If search returns no results, try rephrasing the query with different keywords before giving up.
- If fetch_webpage fails on one URL, try another URL from the search results.
- Never fabricate information. If you cannot find reliable data, say so honestly.
- Respond in the same language the user uses.

## Output Format
- Lead with a direct answer to the user's question.
- Follow with supporting details and context.
- End with source links.
"""