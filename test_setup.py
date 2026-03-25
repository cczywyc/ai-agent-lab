import os
from dotenv import load_dotenv
load_dotenv()

# --- 测试 Anthropic api ---
import anthropic
client_ant = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL")
)
resp = client_ant.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("[Anthropic]", resp.content[0].text)