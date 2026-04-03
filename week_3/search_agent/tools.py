"""
搜索 Agent 工具集

工具清单：
  1. web_search  — 搜索引擎查询（Data Tool，只读）
  2. fetch_webpage — 网页正文提取（Data Tool，只读）

v2.0 变更：
  - fetch_webpage 增加 URL 黑名单预过滤
  - 统一错误返回格式（error_type + message + recoverable + suggestion）
  - 返回值精简，控制 token 消耗
"""

import json
import time
from urllib.parse import urlparse

from config import BLOCKED_DOMAINS


# ============================================================
# 工具 1：web_search — 搜索引擎查询
# ============================================================


def web_search(query: str, max_results: int = 5) -> dict:
    """
    使用 DuckDuckGo 搜索引擎查询。

    返回精简的搜索结果：title + url + snippet（每条截断 200 字符）。
    """
    try:
        from ddgs import DDGS

        with DDGS(proxy=None) as ddgs:  # 如需代理在此配置
            raw_results = list(ddgs.text(query, max_results=max_results))

        if not raw_results:
            return {
                "results": [],
                "total": 0,
                "message": f"No results found for '{query}'.",
                "suggestion": "Try different keywords or a broader search term.",
            }

        # 精简返回值：只保留 title, url, snippet
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": (r.get("body", r.get("snippet", "")))[:200],
            })

        return {
            "results": results,
            "total": len(results),
            "query": query,
        }

    except Exception as e:
        return {
            "error": True,
            "error_type": "SearchError",
            "message": f"Search failed: {str(e)}",
            "recoverable": True,
            "suggestion": "Try a simpler or different query.",
        }


# ============================================================
# 工具 2：fetch_webpage — 网页正文提取
# ============================================================


def fetch_webpage(url: str) -> dict:
    """
    获取网页正文内容。

    包含 URL 黑名单预过滤 + 内容截断（3000 字符）。
    """
    # === 黑名单预过滤 ===
    try:
        domain = urlparse(url).netloc.lower()
        # 检查域名是否在黑名单中（支持子域名匹配）
        if any(blocked in domain for blocked in BLOCKED_DOMAINS):
            return {
                "error": True,
                "error_type": "BlockedDomain",
                "message": f"'{domain}' is known to block automated access.",
                "recoverable": True,
                "suggestion": "Try a different URL from the search results, or answer based on available snippets.",
            }
    except Exception:
        pass  # URL 解析失败，让后续请求来报错

    # === 实际抓取 ===
    try:
        import httpx

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(timeout=10, follow_redirects=True) as http_client:
            resp = http_client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        # 尝试用 readability 提取正文
        try:
            from readability import Document
            doc = Document(html)
            title = doc.title()
            # 简单清理 HTML 标签
            import re
            content = re.sub(r"<[^>]+>", "", doc.summary())
            content = re.sub(r"\s+", " ", content).strip()
        except ImportError:
            # 没有 readability，做基础提取
            import re
            content = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.S)
            content = re.sub(r"<[^>]+>", "", content)
            content = re.sub(r"\s+", " ", content).strip()
            title = ""

        # 截断到 3000 字符
        max_len = 3000
        truncated = len(content) > max_len
        content = content[:max_len]

        if not content or len(content) < 50:
            return {
                "error": True,
                "error_type": "EmptyContent",
                "message": f"Page at '{url}' returned no meaningful content.",
                "recoverable": True,
                "suggestion": "Try a different URL from the search results.",
            }

        return {
            "url": url,
            "title": title,
            "content": content,
            "truncated": truncated,
            "char_count": len(content),
        }

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        error_type = {
            403: "Forbidden",
            404: "NotFound",
            429: "RateLimited",
        }.get(status, "HTTPError")

        return {
            "error": True,
            "error_type": error_type,
            "message": f"HTTP {status} when fetching '{url}'.",
            "recoverable": status != 404,  # 404 不可恢复
            "suggestion": "Try a different URL from the search results.",
        }

    except httpx.TimeoutException:
        return {
            "error": True,
            "error_type": "Timeout",
            "message": f"Request timed out for '{url}'.",
            "recoverable": True,
            "suggestion": "Try a different URL, or answer based on available snippets.",
        }

    except Exception as e:
        return {
            "error": True,
            "error_type": "FetchError",
            "message": f"Failed to fetch '{url}': {str(e)}",
            "recoverable": True,
            "suggestion": "Try a different URL from the search results.",
        }


# ============================================================
# 工具定义（OpenAI 格式，传给 API 的 tools 参数）
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using a search engine. Returns up to 5 results, "
                "each with title, URL, and a brief snippet (max 200 chars). "
                "Use this FIRST for any factual question, technical topic, "
                "or when you need up-to-date information. "
                "Use concise English keywords for best results (2-5 words). "
                "Does NOT fetch full page content — use fetch_webpage for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query in concise English keywords. "
                            "Example: 'MCP Model Context Protocol', "
                            "'LangGraph vs CrewAI comparison'. "
                            "2-5 words works best."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch and extract the main text content from a webpage URL. "
                "Use this AFTER web_search when you need more detailed content "
                "than the search snippets provide. "
                "Returns the article body text, truncated to 3000 characters. "
                "Some websites may block access (403 error) — if this happens, "
                "try a different URL from the search results. "
                "Do NOT guess URLs; only use URLs returned by web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The full URL to fetch. Must be a URL from "
                            "web_search results. Example: 'https://example.com/article'."
                        ),
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]

# ============================================================
# 工具注册表（name → function 映射）
# ============================================================

TOOL_REGISTRY = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
}


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """
    统一工具执行入口。

    通过注册表查找并执行工具函数。
    新增工具只需：1.写函数 2.加定义 3.注册。
    """
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        return {
            "error": True,
            "error_type": "UnknownTool",
            "message": f"Tool '{tool_name}' is not registered.",
            "recoverable": False,
        }

    try:
        return func(**tool_args)
    except TypeError as e:
        return {
            "error": True,
            "error_type": "InvalidArguments",
            "message": f"Invalid arguments for '{tool_name}': {str(e)}",
            "recoverable": True,
            "suggestion": "Check the parameter names and types.",
        }
    except Exception as e:
        return {
            "error": True,
            "error_type": "ToolExecutionError",
            "message": f"Tool '{tool_name}' failed: {str(e)}",
            "recoverable": False,
        }
