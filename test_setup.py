"""
环境验证脚本 — 确认千问 API 连通性
运行: python test_setup.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

resp = client.chat.completions.create(
    model="qwen3-max-2026-01-23",
    messages=[{"role": "user", "content": "用一句话跟我打个招呼"}],
    max_tokens=50
)

print("[Qwen]", resp.choices[0].message.content)
print("\n✅ 千问 API 连通正常")