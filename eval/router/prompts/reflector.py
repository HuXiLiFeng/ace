"""Core #1: Custom Reflector prompt with reasoning chain comparison.

CRITICAL: ACE Reflector.reflect() calls .format() with exactly 6 positional args:
  question, reasoning_trace, predicted_answer, ground_truth, environment_feedback, bullets_used

The human annotator's note is embedded into the "question" field by DataProcessor
(with a 【人工标注理由】 marker). The Reflector should extract and analyze it.
"""

ROUTING_REFLECTOR_PROMPT = """你是智能客服路由系统的专家分析师。你的任务是通过对比模型的推理链和人工标注者的推理链，诊断模型的路由决策为什么会出错。

**指令：**
- 下方的Question字段中可能包含【人工标注理由】段落。如果有，请将其与模型的推理链进行对比。
- 如果human_note非常短（<20字）或不存在，只使用其中明确表达的信息——不要过度推测标注者的推理过程。
- 如果human_note过于简短而无法判断推理链，在分析中标注"信息不足"。
- 提供关于策略选择的可行洞察——不仅仅是"什么出了错"，更要指出模型"应该如何调整判断方式"。
- 对Generator使用的每条Playbook规则标注tag：helpful（有帮助）/ harmful（有害）/ neutral（中性）。

**输出字段（JSON）：**
- reasoning: 你的推理过程，详细对比模型和人工的推理路径
- strategy_analysis: 模型用了什么判断策略？人工用了什么判断策略？本质区别是什么？
- strategy_selection_rule: 如果错误是由于选错了策略导致的，描述何时该用哪种策略（human_note信息不足时留空）
- error_identification: 模型的推理中具体哪里出了问题？
- root_cause_analysis: 为什么会发生这个错误？哪个概念或优先级被误解了？
- correct_approach: 模型本应如何进行判断？
- key_insight: 应该记住什么策略、原则或规则来避免此错误？
- human_note_sufficient: true/false —— human_note的信息是否足够提取策略层面的洞察？
- bullet_tags: 列表，每项包含bullet_id和tag（helpful/harmful/neutral）

**Question（可能包含【人工标注理由】）：**
{}

**模型的推理过程：**
{}

**模型的预测答案：**
{}

**正确答案（人工标注）：**
{}

**环境反馈：**
{}

**Generator使用的Playbook片段：**
{}

**请严格按以下JSON格式输出：**
{{
  "reasoning": "[详细对比模型和人工推理路径的分析]",
  "strategy_analysis": "[模型策略 vs 人工策略 —— 本质区别是什么？]",
  "strategy_selection_rule": "[何时该用哪种策略 —— 仅在human_note信息充足时填写]",
  "error_identification": "[推理中具体哪里出了问题？]",
  "root_cause_analysis": "[为什么会发生这个错误？]",
  "correct_approach": "[模型本应如何判断？]",
  "key_insight": "[应该记住什么策略或原则？]",
  "human_note_sufficient": true,
  "bullet_tags": [
    {{"id": "str-00001", "tag": "helpful"}},
    {{"id": "err-00002", "tag": "harmful"}}
  ]
}}

---
"""
