# 多 Agent 概念

> **定位**：第八周·周一·概念精读日的笔记。读完 Anthropic《How we built our multi-agent research system》和 OpenAI Agents SDK 文档（Agents / Handoffs / Guardrails / Sessions）之后，把这些概念逐个落到第七周已有的家具上，而不是孤立地背名词。</br>
> **带着读的那根刺**：真·多 agent（独立 agent + handoff，各自独立 messages / 上下文）比第七周"一张图里的角色节点"到底**多买到什么**？——带着这个怀疑读，别一上来就堆 agent。</br>
> **基线**：第七周 v5.0（一张 LangGraph 图，planner / executor / critic + 条件边；全员共享同一 state / messages；§四分层装配按角色挑段；escalate / skip-and-advance；max_replan + synthesis-reserve 双闸门；planner 非确定地拆 5/3/5/4/5 步）。

---

## 0. 这份文档要想清的一件事

一句话主轴：**从第七周的"一张图、共享 state"跨到真·多 agent，实打实多买到的只有"独立上下文 + 真并行"两样，代价是"约 15x token + 协调成本 + 隔离副作用"；它只在子任务能切成互相独立的并行分支时才划算。** 反过来，顺序依赖、要共享同一份上下文的活，留在一张图里反而更省更稳。

其余几个概念——handoff 和 tool call 差在哪、中心编排 vs 去中心化怎么选、subagent 的委派契约是什么、角色按什么拆、上下文怎么跨独立窗口路由——都是为了把上面这条主轴判得更准而铺的底，下面逐节展开。

---

## 一、三种多 Agent 拓扑（先把地图画出来）

后面所有概念都落在这三种拓扑里，先建坐标系。

| 拓扑 | 控制权 | 上下文 | 代表实现 | 对应第七周的什么 |
|---|---|---|---|---|
| **A. 一张图里的角色节点**（图内编排） | 始终在图的拓扑手里 | **共享同一 state / messages** | LangGraph（我的 v5.0） | 就是现在的 planner / executor / critic |
| **B. 中心编排 / orchestrator-worker**（agents as tools） | 始终在 orchestrator 手里，subagent 当工具被调 | orchestrator 与每个 subagent **各自独立上下文** | Anthropic Research、OpenAI SDK 的 manager / agents-as-tools | planner 升格成独立 orchestrator、executor 升格成独立 subagent |
| **C. agent-to-agent / handoff / group chat**（去中心化） | **移交**——接手方接管对话 | 各 agent 独立上下文，靠 handoff 协议传递 | OpenAI SDK 的 handoff、AutoGen / CrewAI group chat | 没有直接对应物（这是本周真正"新"的东西） |

我第七周做的是 **A**。本周要体会的，是从 A 跨到 **B / C** 到底跨过了什么——不是"角色更多了"，而是**上下文从"共享"变成了"隔离"，控制流从"图的边"变成了"工具调用（B）"或"控制权移交（C）"**。这条认识贯穿全篇。

---

## 二、handoff、普通 tool call、图内条件边——到底差在哪

handoff 是"把对话控制权交给另一个专门 agent，由它接管后续回应"的机制。在 OpenAI Agents SDK 里，handoff 对模型而言**就表现为一个工具**——把任务交给 Refund Agent，模型看到的就是一个名叫 `transfer_to_refund_agent` 的工具调用。所以 handoff 在调用形态上确实"长得像 tool call"，但语义完全是两回事：

| 维度 | 普通 tool call | handoff | 第七周的图内条件边 |
|---|---|---|---|
| 调用后谁说下一句 | **还是当前 agent**（拿到结果继续推理） | **接手的 agent**（它接管对话、自己产出下一条回应） | 没有"谁说话"的概念，只是 state 流到下一个节点 |
| 上下文 | 共享当前 agent 的 messages | 接手方**用自己的上下文**（可经 input_filter 裁剪后接收） | 全程共享同一 state |
| 控制权 | 不转移 | **转移**（C）；agents-as-tools（B）则不转移 | 由图拓扑持有，从不"转移给某个 agent" |

一句话：tool call 是"我调个函数拿结果继续干"；handoff 是"这活不归我了，你来接"。这正是 A→C 的本质跃迁——我第七周从来没有"把活交出去、自己退场"这件事，escalate / skip-and-advance 始终是**图**在做决策，agent 自己没有"退场"的语义。

---

## 三、中心编排 vs 去中心化 handoff：谁始终握着控制权

这是 B 和 C 的分界，核心就一个问题：**谁始终持有对话控制权。**

- **中心编排（B，agents as tools）**：一个中心 orchestrator 把专门 subagent 当工具来调，**自己始终保留对话控制权、负责最终回应**。subagent 干完把结论回传，orchestrator 决定下一步。OpenAI SDK 的指引很直白——当"主 agent 应当对最终答案负责、专家只是幕后帮手"时，用 agents-as-tools。
- **去中心化 handoff（C）**：平级 agent 之间**移交控制权**，接手方接管对话、由它产出下一条用户可见的回应。SDK 的指引是——当"某个专家应当拥有下一条回应、而不只是幕后帮忙"时，用 handoff。

取舍：

| | B 中心编排 | C 去中心化 handoff |
|---|---|---|
| 可控性 / 可观测 | **高**（一切经过 orchestrator） | 低（控制权在 agent 间流转，trace 更难追） |
| 灵活性 | 低（orchestrator 是瓶颈也是单点） | 高（专家自治） |
| 适合 | 需要统一收口、统一对最终结果负责 | 客服转接式、各专家独立拥有一段对话 |
| 失控风险 | 小 | **大**（容易绕圈、互相 handoff 不收敛） |

OpenAI 文档那条设计建议可以当口诀记：给每个专家一个**窄**的职责；**只有当下一个分支真的需要不同的指令 / 工具 / 策略时才拆**。能用 B 收口就别急着上 C——这跟 roadmap 那句"不是越多 agent 越好"是同一条。

---

## 四、从 A 跨到 B/C，到底多买到什么（主轴）

把名词剥掉，从 A 跨到 B/C 实打实多买到的只有两样，且都带明确代价。

**买到的（仅两样）**：

1. **独立上下文窗口**。每个 subagent 有自己的上下文、工具和探索轨迹，互不污染。它解决的是单一上下文窗口装不下的问题——Anthropic 那篇里，lead agent 在上下文逼近 ~200K token 时，得把研究计划写进 memory 才不丢；而 subagent 各自独立窗口，天然把这种压力分摊掉了。
2. **真正的并行**。多个 subagent 同时跑、各查一摊，这是顺序执行做不到的。Anthropic 内部评测里，Opus 4 当 lead + Sonnet 4 当 subagent 这套，比单 agent 的 Opus 4 高出约 **90.2%**——而这个提升被明确归因于 token 用量，以及"把推理摊到多个独立上下文窗口上"的能力。

**付出的代价**：

- **约 15x token**（相对普通对话）。所以只有"结果价值压得过成本"才划算。
- **协调成本**。subagent 之间会重复劳动——Anthropic 早期就出过事故：一个 subagent 在查 2021 芯片危机，另两个重复去查 2025 供应链。这得靠 orchestrator 的委派 prompt 把边界写清才压得住。
- **隔离的副作用**。隔离的另一面是"互相不知道对方在干嘛"。在强依赖任务里，这是 bug 不是 feature（见 §五）。

所以主轴的答案是：真·多 agent 多买到的是"**独立上下文 + 真并行**"，代价是"**15x token + 协调 + 隔离**"。它**只在子任务能被切成互相独立的并行分支时才划算**；如果任务本来就共享上下文、或子任务强依赖，A（一张图、共享 state）反而更省更稳。我第七周的 v5.0 之所以做成 A 是对的——研究步骤之间有顺序依赖、又要共享检索到的 chunk，这正是 A 的甜区。

---

## 五、什么时候该上多 Agent，什么时候不该

这是本周要交的《什么时候该用多 Agent，什么时候不要用》的核心。我把判据收成**一条主轴 + 三个体征**。

**主轴判据**（Anthropic 博客的原则）：**需要所有 agent 共享同一上下文、或 agent 之间存在大量依赖的领域，目前不适合多 Agent。** 并行只在子任务真正互相独立时才有收益——**如果 subagent B 要等 A 的产出才能开工，"并行"就退化成了带额外开销的串行。**

**三个该用的体征**：

| 体征 | 说明 | 反例（不该用） |
|---|---|---|
| **广度优先（breadth-first）** | 答案需要横扫一大片来源 / 候选（如"S&P 500 IT 板块所有董事会成员"），单上下文装不下 | 深度优先、单线推理 |
| **子任务可独立并行** | 各分支互不依赖，能同时跑 | 强顺序依赖的流水线 |
| **结果价值 > 15x 成本** | 高价值产出（深度调研报告）才值这个 token 账单 | 日常对话、快速问答 |

**明确不该用的场景**：编码 / 调试这类**强依赖、需共享同一份上下文**的任务——多 agent 在这里几乎必亏。还有简单事实查询——早期系统的教训就是，给简单问题也猛拆 subagent，只会徒增冗余检索。

**配套：努力分级（effort-scaling）**。Anthropic 把"派几个 agent"直接写进了 orchestrator 的 prompt 规则：简单事实查证 1 个；直接对比 2–4 个；复杂研究 10+ 个。这条恰好呼应我第七周 planner 拆 5/3/5/4/5 步的非确定性——"派几个"本身要由任务复杂度驱动，不是常数。

---

## 六、真正的难点是 orchestrator 的委派 prompt

多 agent 工程的难点不在"怎么并行"，而在"orchestrator 怎么把任务讲清楚"。Anthropic 给每个 subagent 的委派契约有**四要素**，缺一不可：

1. **objective**：这个 subagent 要达成什么目标；
2. **output format**：产出长什么样（好让 orchestrator 能消费、能拼装）；
3. **工具 / 来源指引**：用哪些工具、查哪些源；
4. **明确的任务边界**：明确说"**别碰 X，那是另一个 subagent 的活**"。

**少任何一条，subagent 就会漂**——不是模型不听话，而是 orchestrator 没把"做到什么算完"讲清楚。前面那个"两个 subagent 重复查 2025 供应链"的事故，根因就是边界没写清；修法也很朴素，就是把 objective / 边界 / "别查 X"写进委派 prompt。

接回第七周：这四要素我其实已经摸到三条了——executor 的子任务 `query`（≈objective）、结构化报告的段落契约（≈output format）、`retrieve_documents` 工具（≈工具指引）。**唯一系统性缺的是第 4 条"边界 / 别碰 X"**——因为在 A 拓扑里子任务共享 state、天然不会"撞车"，而一旦上 B/C、subagent 各自独立上下文，"边界"就从隐式变成必须显式写。这是本周最值得补的认知缺口。

---

## 七、角色怎么拆——按"下一段要不要换指令/工具/策略"，不按职业

拆角色不是按"听起来像不同职业"拆，而是按**下一段是否真的需要不同的指令 / 工具 / 策略**拆（OpenAI 文档的原则）。

- **planner / orchestrator**：拆任务、定策略、派活、收口。对应我的 planner，升一层就是 lead agent。
- **researcher**：查资料、做检索、压缩结论回传。对应我的 executor。
- **writer**：把研究结论组织成初稿。**我第七周没有独立的 writer**——finalize 直接装配报告，没有一个"专门写"的角色。
- **reviewer**：审阅、打回。对应我的 critic（critic ≈ reviewer，escalate 通道 ≈ "打回上游"），这条几乎一一对应。

拆分纪律就一句：给每个角色**窄**职责；**只有当下一个分支真需要不同指令/工具/策略时才拆出独立 agent**，否则就让它留在同一 agent / 同一节点里。把"writer"拆成独立 agent 之前先问一句：它需要的指令 / 工具，和 researcher 真的不同到值得隔离上下文吗？

---

## 八、上下文怎么跨独立窗口路由（接第七周 §四）

这是第七周 §四"分层装配 = 按角色路由上下文"的**直接升级**。

- **第七周（A 拓扑）**：所有角色共享一份 state，"路由"是指**在装配时按角色挑段**——planner 看摘要、executor 看局部召回。本质是"在共享池里挑给谁看什么"。
- **本周（B/C 拓扑）**：上下文**物理隔离**——每个 subagent 一个独立窗口。"路由"从"挑段"升级成两件事：① **委派时传什么**（orchestrator 在 handoff / 委派里塞进哪些上下文，OpenAI SDK 的 `input_filter` 就是干这个的——裁剪交给下一个 agent 的输入）；② **回收时收什么**（subagent 把结论**压缩**后回传，而不是把整个探索轨迹倒给 orchestrator）。

一句话：第七周是"时间维度按轮挑段 + 空间维度按角色挑段"，但都在**一个共享池**里；本周是把"按角色挑段"升级成**跨独立上下文的"交接什么 / 回收什么"协议**。这正是 §四埋下的那颗种子开花的地方。

---

## 九、新概念 → 第七周 映射表（认知锚点）

| 多 Agent 新概念 | 第七周已有家具 | 跨过去多了什么 |
|---|---|---|
| orchestrator / lead agent | planner（拆计划、判 done） | 从"图内节点、共享 state"→"独立 agent、独立上下文、负责派活收口" |
| subagent / worker | executor（单步检索总结） | 从"复用同一 messages"→"独立 context 窗口、可真并行" |
| reviewer | critic（已有，≈一一对应） | 几乎不变；唯一变化是 C 拓扑下它能"接管对话"而非只走条件边 |
| handoff | escalate / skip-and-advance（图内条件边） | 从"图做决策、agent 不退场"→"控制权真移交、接手方接管对话" |
| 委派契约四要素 | 子任务 query + 报告段落契约 + retrieve 工具 | **新增第 4 条"任务边界 / 别碰 X"**（A 里隐式、B/C 里必须显式） |
| 上下文路由（input_filter / 压缩回传） | §四 分层装配（按角色挑段） | 从"共享池里挑段"→"跨独立上下文的交接 + 回收协议" |
| 努力分级（1 / 2–4 / 10+ agent） | planner 非确定拆 5/3/5/4/5 步 | 把"派几个"显式写进 orchestrator prompt、由复杂度驱动 |

---

## 十、把本周实战放到判据上称一称（带进周二设计日的那根刺）

这是今天最值得想清的一点，正好契合"带着怀疑去学"。

roadmap 本周实战是 **researcher → writer → reviewer** 三角色。但拿 §五 的判据一称：**这是一条强顺序依赖的流水线**——writer 必须拿 researcher 的产出才能写、reviewer 必须拿 writer 的草稿才能审。按"子任务强依赖 → 不适合多 Agent / 并行退化成串行"的原则，**它恰恰落在多 Agent 的非甜区**：这里没有可并行的独立分支，多 agent 买不到 §四的"真并行"，只买到"角色专精 + 干净的 handoff 契约"。

所以周二设计日要回答的真问题，不是"怎么搭三个 agent"，而是这个 demo 该落在 §一 的哪个拓扑：

| 选项 | 怎么做 | 代价 / 收益 |
|---|---|---|
| **A. 沿用一张图 + 加 writer/reviewer 节点** | 复用 v5.0 的 executor 引擎，writer / reviewer 当新节点，共享 state | 最省、最稳；但学不到 handoff / 上下文隔离 |
| **B. 中心编排（orchestrator + 3 个 agent as tools）** | 一个 orchestrator 顺序调 researcher / writer / reviewer | 学到独立上下文 + 委派契约；收口清晰、可控 |
| **C. 去中心化 handoff（researcher→writer→reviewer，reviewer 打回 writer）** | 控制权沿链移交，reviewer 可 handoff 回 writer | 最贴 roadmap 的"handoff"主题；但顺序链上 handoff 收益有限、且要防打回不收敛 |

诚实的判断：对**这条顺序流水线**而言，独立 agent 的隔离成本换不回多少能力。所以本周 demo 的价值点，应当刻意放在 roadmap 要求的"**明确每个角色的输入输出格式**"——也就是 **handoff 协议 / 角色 I/O 契约本身**，而不是"证明多 agent 更强"。这样那份《什么时候该用多 Agent》就不是抄观点，而是**拿自己的 demo 当反例论证出来的**——比正面堆一个"看，多 agent 能跑"的 demo 有价值得多。

---

## 十一、未解决的问题（留给周二设计日 / 后续周次）

**Q1：这个 demo 选 A / B / C 哪个拓扑？**（周二必须定）
判据见 §十。倾向：若本周学习目标是"体会 handoff 与上下文隔离"，选 B 或 C；若目标是"最稳地交付报告"，A 就够。我倾向选 **B（中心编排）**——既能真正引入"独立上下文 + 委派契约"这两个新概念，又用 orchestrator 收口、可控性比 C 高，避免一上来就踩 handoff 不收敛的坑。

**Q2：reviewer 打回的终止与回滚怎么设计？**
第七周的 escalate / skip-and-advance / `replan_count` 闸门能不能降一层复用到"reviewer 打回 writer"的循环上？打回几次封顶？打回时 writer 拿到的是"原稿 + reviewer 意见"还是重新来过？（这是第七周三级重试的再一次降层。）

**Q3：跨 agent 的控制流，纠正 / 降级在哪一层生效？**（承第三周《Agent Loop 设计笔记》Q3）
如果 orchestrator 调 researcher，researcher 自己也有内层 tool-use loop，那"该搜没搜""连续失败降级"在哪一层兜？subagent 的失败要不要上报给 orchestrator？——这正是第三周留下、单 agent 场景答不了的那个问题，本周该有答案了。

**Q4：委派契约的第 4 条"任务边界"在只有 3 个固定角色时还重要吗？**
Anthropic 的边界问题源于"多个同质 subagent 并行、会撞车"。本周是 3 个**异质**角色、且顺序执行，天然不撞车——所以"边界"这条在本周 demo 里可能弱化。但要在笔记里点明：**它在"同质 + 并行"时才是刚需**，本周用不上不代表不重要（第九周接 MCP / 后续做并行 subagent 时会回来）。

---

*完成于 2026-06-15（第八周·周一·概念精读日）。读的两份材料：Anthropic《How we built our multi-agent research system》、OpenAI Agents SDK 官方文档（Agents / Handoffs / Guardrails / Sessions）。文中所有结论都接回第七周 v5.0（planner/executor/critic、§四分层装配、escalate/skip-and-advance、双闸门）与第三周《Agent Loop 设计笔记》Q3。*
