# Search Agent — Week 2 Demo

基于千问 + DuckDuckGo 的最小搜索 Agent。

## 功能
- 🔍 网页搜索（DuckDuckGo）
- 📄 网页内容提取（readability）
- 🔄 自动多轮工具调用（Agent Loop）
- 📋 完整 trace 记录
- ⚠️ 失败兜底（搜索为空/网页超时/格式错误）

## 工具设计
| 工具 | 类型 | 说明 |
|------|------|------|
| web_search | Data Tool | 搜索网页，返回标题+摘要+URL |
| fetch_webpage | Data Tool | 提取网页正文内容 |

## 运行
（安装和运行命令）

## 已知问题 & 待改进
### 已知问题（基于 2026-03-30 实测）

#### Agent 实现层面
- **Agent Loop 退出条件过于简单**：`agent.py` 中只要 `finish_reason == “stop”` 或无 `tool_calls` 就直接退出循环，没有检查模型是否应该先搜索再回答。实测 7 个 case 中有 3 个（诺贝尔奖、RAG、xyzabc123）模型直接凭已有知识回答，跳过了工具调用。
- **System Prompt 对工具调用的约束力不足**：Prompt 说”when the user asks a question requiring current information, first use web_search”，但”current information”的界定模糊，模型对于自身训练数据已覆盖的问题倾向于直接回答，导致工具使用率偏低（7 个 case 只有 3 个调用了工具）。
- **缺少从工具失败到 snippet 回退的引导机制**：当 `fetch_webpage` 连续失败时，Agent 只是继续循环让模型自行决策，没有主动引导”放弃抓取，用已有 snippet 总结回答”。实测 case 3（LangGraph vs CrewAI）因连续遇到 403 耗尽 5 轮次，最终无答案输出。
- **MAX_TURNS = 5 对多页面比对类问题偏小**：需要搜索 + 多次抓取 + 对比的问题容易因网页抓取失败而耗尽轮次。实测中对比类问题 5 轮调用了 5 次工具仍未完成。
- **多轮对话的上下文窗口未管理**：messages 列表随工具调用不断累积（每次工具结果最多 3000 字符），没有对历史工具结果做裁剪或摘要，长对话可能逼近或超出模型上下文限制。
- **工具错误信号未被 Agent 利用**：工具返回了结构化的 `recoverable` / `suggestion` 字段，但 Agent Loop 本身不读取这些字段，完全依赖模型自行解析 JSON 并决定是否重试，降低了错误恢复的可靠性。

#### 工具设计层面
- **web_search 缺少 query 预处理**：中文 query 直接传给 DuckDuckGo，没有翻译或关键词改写。虽然 tool description 提示”prefer English keywords”，但模型不一定遵守，影响中文问题的搜索质量。
- **web_search 对无意义查询缺少结果相关性过滤**：DuckDuckGo 对无意义字符串（如 `xyzabc123`）仍可能返回关联词的页面，工具层面没有对结果与原始 query 的相关性做基本过滤，导致”搜索为空”的兜底路径难以稳定触发。
- **fetch_webpage 缺少 URL 预过滤机制**：已知 Medium、DataCamp 等站点必然返回 403，但工具没有黑名单或预检测，每次无效抓取浪费一轮 Agent 循环。
- **fetch_webpage 缺少内容类型检测**：没有在下载前通过 HEAD 请求或 Content-Type 判断资源是否为 HTML，可能浪费时间在 PDF、图片等非 HTML 资源上。
- **工具间缺少协作设计**：`web_search` 和 `fetch_webpage` 完全独立，对于”搜索 + 读取第一条结果”这种高频模式，模型必须分两轮调用，增加了不必要的轮次消耗和延迟。

### 待改进

#### Agent 实现
1. **增强 System Prompt 的工具调用约束**：更明确地区分”常识问答”和”需要搜索验证的问答”，对事实性问题、时效性问题强制要求先搜索，减少模型直接跳过工具的情况。
2. **在 Agent Loop 中加入 fallback 策略**：当 `fetch_webpage` 连续失败达到阈值时，在下一轮 messages 中注入提示，引导模型基于已有 snippet 给出回答，避免耗尽轮次后无输出。
3. **适当增大 MAX_TURNS 并加入动态调整**：对需要多步操作的问题（对比类、深度研究类）允许更多轮次；同时在连续失败时提前终止，避免无效循环。
4. **Agent 层面利用工具返回的错误元数据**：读取工具结果中的 `recoverable` 和 `suggestion` 字段，在 Agent Loop 中做自动重试或策略切换，而不是完全交给模型自由发挥。
5. **增加上下文窗口管理**：对历史 messages 中的长工具结果做摘要裁剪，防止多轮对话累积超出模型上下文限制。
6. **提升 Trace 结构化程度**：在 trace 中增加”模型为何未调用工具””工具失败后模型的恢复决策”等结构化字段，便于复盘 Agent 行为。

#### 工具设计
1. **web_search 增加 query 改写能力**：对中文问题、对比类问题自动生成更适合搜索引擎的英文关键词，而不是完全依赖模型在 prompt 层面的遵从。
2. **web_search 增加结果相关性基础过滤**：对搜索结果与原始 query 做简单的关键词匹配或相似度判断，过滤明显无关的结果，让”空结果”兜底路径更可靠。
3. **fetch_webpage 增加 URL 预检测和黑名单**：维护已知反爬站点列表，抓取前先做 HEAD 请求检查 Content-Type 和状态码，快速跳过不可访问的 URL，减少轮次浪费。
4. **考虑增加组合工具**：提供 `search_and_read` 类型的组合工具，一次调用完成”搜索 + 抓取最相关结果”，减少简单问题的轮次消耗。