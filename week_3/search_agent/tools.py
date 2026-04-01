"""
tools.py — 工具实现与注册
==========================
包含两个 Data Tool：
  1. web_search    — 网页搜索（DuckDuckGo）
  2. fetch_webpage — 网页内容提取（readability）

设计原则（来自《Tool 设计规范 v1.0》）：
  - 返回值高信号、低噪音，只包含模型推理所需字段
  - 错误返回包含 error_type + message + recoverable + suggestion
  - 工具内部做容错（如中英文映射），不依赖模型总是传完美参数
"""
import json
import re

import httpx
from ddgs import DDGS
from readability import Document

from config import MAX_SEARCH_RESULTS, MAX_WEBPAGE_LENGTH


# ===================================================================
# 工具 1：网页搜索
# ===================================================================

def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> dict:
    """
    使用 DuckDuckGo 搜索网页，返回精简的结果列表。

    参数:
        query: 搜索关键词
        max_results: 返回结果数量，范围 1-10

    返回:
        成功: {"results": [...], "total": N, "query": "..."}
        失败: {"error": True, "error_type": "...", "message": "...", ...}
    """
    # 参数边界保护
    max_results = max(1, min(10, max_results))

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        # 空结果处理 — 永远不返回裸空数组
        if not raw_results:
            return {
                "results": [],
                "total": 0,
                "query": query,
                "message": f"No results found for '{query}'.",
                "suggestion": "Try different keywords, a broader query, or switch language (Chinese/English)."
            }

        # 只保留模型需要的字段（精简原则）
        clean_results = []
        for r in raw_results:
            clean_results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:200]
            })

        return {
            "results": clean_results,
            "total": len(clean_results),
            "query": query
        }

    except Exception as e:
        error_msg = str(e)

        # 区分常见错误类型，给出针对性建议
        if "ConnectError" in error_msg or "ProxyError" in error_msg:
            return {
                "error": True,
                "error_type": "NetworkError",
                "message": f"Cannot connect to search service. Proxy or network issue: {error_msg[:100]}",
                "recoverable": False,
                "suggestion": "Check network/proxy configuration."
            }
        elif "RatelimitE" in error_msg or "429" in error_msg:
            return {
                "error": True,
                "error_type": "RateLimitError",
                "message": "Search rate limit reached.",
                "recoverable": True,
                "suggestion": "Wait a moment and try again with fewer results."
            }
        else:
            return {
                "error": True,
                "error_type": "APIFailure",
                "message": f"Search failed: {error_msg[:150]}",
                "recoverable": True,
                "suggestion": "Try again with simpler or different keywords."
            }


# ===================================================================
# 工具 2：网页内容提取
# ===================================================================

def fetch_webpage(url: str, max_length: int = MAX_WEBPAGE_LENGTH) -> dict:
    """
    获取网页正文内容，使用 readability 提取主要内容，过滤导航栏/广告等噪音。

    参数:
        url: 完整的网页 URL

    返回:
        成功: {"title": "...", "url": "...", "content": "...", "truncated": bool}
        失败: {"error": True, "error_type": "...", "message": "...", ...}
    """
    try:
        resp = httpx.get(
            url,
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        )
        resp.raise_for_status()

        doc = Document(resp.text)
        title = doc.title()

        # 提取纯文本，去掉 HTML 标签
        content = re.sub(r'<[^>]+>', '', doc.summary())
        content = re.sub(r'\s+', ' ', content).strip()

        if not content:
            return {
                "error": True,
                "error_type": "NotFoundError",
                "message": f"No readable content extracted from {url}. The page may require JavaScript or login.",
                "recoverable": False,
                "suggestion": "Try a different URL from the search results."
            }

        return {
            "title": title,
            "url": url,
            "content": content[:max_length],
            "truncated": len(content) > max_length,
            "content_length": len(content)
        }

    except httpx.TimeoutException:
        return {
            "error": True,
            "error_type": "TimeoutError",
            "message": f"Request to {url} timed out after 10s.",
            "recoverable": True,
            "suggestion": "Try a different URL from the search results."
        }
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        return {
            "error": True,
            "error_type": "HTTPError",
            "message": f"HTTP {code} when fetching {url}.",
            "recoverable": code >= 500,
            "suggestion": "Try a different URL." if code == 403 or code == 404 else "The server may be temporarily unavailable, try again later."
        }
    except Exception as e:
        return {
            "error": True,
            "error_type": "ToolInternalError",
            "message": f"Failed to fetch {url}: {str(e)[:100]}",
            "recoverable": False,
            "suggestion": "Try a different URL from the search results."
        }


# ===================================================================
# 工具定义（OpenAI/千问 格式）
# ===================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo and return relevant results. "
                "Use this when the user asks a question that requires up-to-date "
                "information, facts you're unsure about, or current events. "
                "Returns a list of results with title, URL, and snippet. "
                "Does NOT fetch the full content of web pages — use "
                "fetch_webpage for that. "
                "For best results, use concise English keywords as the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query string. Use concise keywords for best results. "
                            "Prefer English keywords even if the user asks in Chinese. "
                            "Example: '2024 Nobel Prize Physics winner'"
                        )
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return. Range: 1-10. Default: 5 if omitted."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch and extract the main text content from a web page URL. "
                "Use this AFTER web_search when you need the full article content "
                "rather than just the search snippet. "
                "Returns the page title and extracted main content (max 3000 chars). "
                "Does NOT work on pages requiring login, JavaScript rendering, or PDF files. "
                "If this tool fails, try a different URL from the search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch, e.g. 'https://example.com/article'. Must start with http:// or https://."
                    }
                },
                "required": ["url"]
            }
        }
    }
]


# ===================================================================
# 工具注册表 — 解耦工具定义和工具实现
# ===================================================================

TOOL_REGISTRY = {
    "web_search": lambda args: web_search(
        query=args["query"],
        max_results=args.get("max_results", MAX_SEARCH_RESULTS)
    ),
    "fetch_webpage": lambda args: fetch_webpage(
        url=args["url"]
    ),
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    统一的工具执行入口。

    返回值始终是 JSON 字符串（符合 OpenAI tool result 格式要求）。
    """
    if tool_name in TOOL_REGISTRY:
        result = TOOL_REGISTRY[tool_name](tool_args)
    else:
        result = {
            "error": True,
            "error_type": "ToolNotFound",
            "message": f"Unknown tool: '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}",
            "recoverable": False
        }

    return json.dumps(result, ensure_ascii=False)