

# 背景信息

1. 消费金融场景，智能客服的路由模块。当前线上可用的 agent 共 6 个，每个 agent 负责处理一个方面的业务，保存在 `agent_info.json` 文件中，文件包含每个 agent 的职责与编码。线上路由模块由提示词驱动。
2. 路由模块已经在线上运行一段时间了，每周由标注人员随机抽取一部分数据进行标注，标注结果参考 `data_sample.json`。`router_agent` 是线上路由模块输出结果，`human_annotate_agent` 是人工标注结果，badcase 就是二者不一致的 case。`router_reason` 是线上路由的原因，`human_note` 是人工简单标注的原因

---

# 当前提示词

根据agent职责、当前 query、对话历史、当前 agent，去预测接下来应该由哪个 agent 处理用户query：

```
# 任务：智能路由
你是一个客服系统的智能路由器。你的任务是分析用户的最新问题，并从下面的Agent列表中，选择一个最匹配的来处理请求。
如果没有任何Agent能够符合用户的需求，那么返回"None"

# 可用Agent列表
{agent_info}

# 对话历史
{history}

# 当前agent
{current_agent}

# 用户最新问题
{current_query}

# 输出格式（严格遵守 JSON）
{{
 "分发目标": "在此填写最匹配的Agent全名",
 "理由": "xxxx"
}}
```

# 任务
1. 分析一些badcase之后，我发现大模型之所以会判断错误，跟人相比，他缺少业务知识积累、经验积累。所以我希望能够从数据中提取出业务经验喂给大模型从而提升判断准确性
2. Agentic Context Engineering Evolving Contexts for Self-Improving Language Models.pdf 这篇论文是否可以作为参考来实现这一任务，论文对应代码已经下载到根目录了，从README.md结合论文来理解这篇论文
