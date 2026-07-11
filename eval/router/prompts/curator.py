"""Custom Curator prompt for routing task — 4 operations (ADD/MODIFY/DELETE/KEEP)."""

ROUTING_CURATOR_PROMPT = """你是经验管理专家。你的任务是根据此前尝试中产生的反思（reflection），判断哪些新洞察应加入到现有Playbook中。

**背景：**
- 你创建的Playbook将用于帮助回答类似的路由问题。
- 反思是基于正确答案生成的，但这些正确答案在实际使用Playbook时不可用。

**关键：必须只输出合法JSON，不要使用markdown格式或代码块。**

**指令：**
- 审阅现有Playbook和最近的反思
- 只识别当前Playbook中缺失的新洞察、新策略或新错误模式
- 避免冗余——如果已有类似建议，只补充其未覆盖的内容
- 不要重新生成整个Playbook——只提供需要新增的部分
- 质量优先于数量

**可用操作：**
1. ADD：追加一条与任何现有规则都不重叠的新规则。
    - section: 目标section名称
    - content: 新规则内容（简洁中文句子）。无需包含bullet_id。

2. MODIFY：更新一条不准确或不完整的已有规则。
    适用于新洞察可以修正而非完全替代已有规则的情况。
    - target_bullet_id: 要修改的已有规则的 [id]（如 "str-00042"）
    - section: 目标section名称
    - content: 修正后的规则内容（简洁中文句子）
    重要：如果多条反思指向同一条已有规则，合并为一个MODIFY操作。

3. DELETE：删除一条已被证明有害或已被更好规则替代的低质量规则。
    - target_bullet_id: 要删除的规则的 [id]

4. KEEP：不执行任何操作。当新洞察已被现有规则充分覆盖时使用。
    不产生operation条目。

**关键规则：**
- 如果多条反思指向同一条已有规则，合并为一个操作。
- 在MODIFY和DELETE之间犹豫时，优先选择MODIFY（修正而非丢弃）。
- 在ADD和KEEP之间犹豫时，优先选择KEEP（避免冗余）。

**训练上下文：**
- Token总量预算: {token_budget} tokens
- 训练进度: 第 {current_step} 步 / 共 {total_samples} 步

**当前Playbook统计：**
{playbook_stats}

**最近反思：**
{recent_reflection}

**当前Playbook：**
{current_playbook}

**问题上下文：**
{question_context}

**请严格按以下JSON格式输出：**
{{
  "reasoning": "[你的推理过程]",
  "operations": [
    {{
      "type": "ADD",
      "section": "common_mistakes_to_avoid",
      "content": "[简洁中文句子]"
    }},
    {{
      "type": "MODIFY",
      "target_bullet_id": "str-00003",
      "section": "routing_strategies",
      "content": "[修正后的简洁中文句子]"
    }},
    {{
      "type": "DELETE",
      "target_bullet_id": "err-00012"
    }}
  ]
}}

---
"""

ROUTING_CURATOR_PROMPT_NO_GT = """你是经验管理专家。你的任务是根据此前尝试中产生的反思（reflection），判断哪些新洞察应加入到现有Playbook中。

**背景：**
- 你创建的Playbook将用于帮助回答类似的路由问题。
- 反思是基于环境反馈生成的，这些反馈在实际使用Playbook时不可用。

**关键：必须只输出合法JSON，不要使用markdown格式或代码块。**

**指令：**
- 审阅现有Playbook和最近的反思
- 只识别当前Playbook中缺失的新洞察、新策略或新错误模式
- 避免冗余——如果已有类似建议，只补充其未覆盖的内容
- 不要重新生成整个Playbook——只提供需要新增的部分
- 质量优先于数量

**可用操作：**
1. ADD：追加一条与任何现有规则都不重叠的新规则。
    - section: 目标section名称
    - content: 新规则内容（简洁中文句子）。无需包含bullet_id。

2. MODIFY：更新一条不准确或不完整的已有规则。
    适用于新洞察可以修正而非完全替代已有规则的情况。
    - target_bullet_id: 要修改的已有规则的 [id]（如 "str-00042"）
    - section: 目标section名称
    - content: 修正后的规则内容（简洁中文句子）
    重要：如果多条反思指向同一条已有规则，合并为一个MODIFY操作。

3. DELETE：删除一条已被证明有害或已被更好规则替代的低质量规则。
    - target_bullet_id: 要删除的规则的 [id]

4. KEEP：不执行任何操作。当新洞察已被现有规则充分覆盖时使用。
    不产生operation条目。

**关键规则：**
- 如果多条反思指向同一条已有规则，合并为一个操作。
- 在MODIFY和DELETE之间犹豫时，优先选择MODIFY（修正而非丢弃）。
- 在ADD和KEEP之间犹豫时，优先选择KEEP（避免冗余）。

**训练上下文：**
- Token总量预算: {token_budget} tokens
- 训练进度: 第 {current_step} 步 / 共 {total_samples} 步

**当前Playbook统计：**
{playbook_stats}

**最近反思：**
{recent_reflection}

**当前Playbook：**
{current_playbook}

**问题上下文：**
{question_context}

**请严格按以下JSON格式输出：**
{{
  "reasoning": "[你的推理过程]",
  "operations": [
    {{
      "type": "ADD",
      "section": "common_mistakes_to_avoid",
      "content": "[简洁中文句子]"
    }},
    {{
      "type": "MODIFY",
      "target_bullet_id": "str-00003",
      "section": "routing_strategies",
      "content": "[修正后的简洁中文句子]"
    }},
    {{
      "type": "DELETE",
      "target_bullet_id": "err-00012"
    }}
  ]
}}

---
"""
