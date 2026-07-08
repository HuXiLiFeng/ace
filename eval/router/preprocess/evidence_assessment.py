"""Core #3: Analyze router_reason patterns to extract reliable/unreliable evidence types."""

import json
from typing import List, Dict
from openai import OpenAI


B_CLASS_QUALITY_PROMPT = """你是一个客服路由系统的质量评估专家。请评估以下模型路由推理的质量。

推理内容:
{reason}

请判断这段推理属于以下哪一类：
- "推理扎实": 引用了Agent【具体意图】列表中的精确条目，推理链简短直接（≤2步联想），逻辑清晰
- "推理一般": 有基本逻辑但不够严谨，或推理链有3步以上联想
- "推理牵强": 推理链很长（>4步），或使用了模糊的概念关联，或明显是猜测

只输出一个词: 扎实 / 一般 / 牵强
"""

EVIDENCE_ANALYSIS_PROMPT = """你是一个客服路由系统的业务分析师。请分析以下路由推理中模型使用了什么类型的证据。

模型推理:
{reason}

请分解这段推理中引用的每条证据，并归类为以下类型之一：
- "意图精确匹配": 直接引用query中的词匹配到某Agent的【具体意图】条目
- "上下文关联": 引用对话历史中的话题来推断
- "职责字面匹配": 引用Agent职责描述中的词做宽松匹配
- "排除法": 先排除不相关的Agent
- "多步联想": 通过多步概念关联间接推断
- "其他"

对每条证据输出一行JSON：
{{"evidence": "证据原文片段", "type": "证据类型", "reliability": "可靠/不可靠/待定"}}
"""

RELIABILITY_SUMMARY_PROMPT = """你是一个客服路由系统的业务分析师。请从以下不可靠证据和可靠证据的汇总中，归纳出系统性的证据可靠性模式。

不可靠证据汇总（来自badcase的router_reason分析）:
{unreliable_summary}

可靠证据汇总（来自高质量正确case的router_reason分析）:
{reliable_summary}

请输出一个Playbook章节，包含：
1. 不可靠证据模式（3-5条），每条包含: 模型倾向、风险、对策
2. 可靠证据模式（3-5条），每条包含: 证据特征、为什么可靠

格式:
## EVIDENCE RELIABILITY

### 不可靠证据 — 需要谨慎对待

[evi-XXXXX] helpful=0 harmful=0 ::
【证据类型名称】
模型倾向: [描述]
风险: [描述]
对策: [描述]
badcase出现次数: X

### 可靠证据 — 正确推理的信号

[evi-XXXXX] helpful=0 harmful=0 ::
【证据类型名称】
特征: [描述]
原因: [描述]
正确case出现占比: X/Y
"""


def classify_reason_quality(
    reason: str,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro"
) -> str:
    """Classify a single router_reason as 扎实/一般/牵强."""
    if not reason or len(reason.strip()) < 10:
        return "牵强"
    prompt = B_CLASS_QUALITY_PROMPT.format(reason=reason)
    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def analyze_reason_evidence(
    reason: str,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro"
) -> List[Dict]:
    """Decompose a router_reason into evidence items with type classification."""
    if not reason or len(reason.strip()) < 10:
        return []
    prompt = EVIDENCE_ANALYSIS_PROMPT.format(reason=reason)
    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.1,
    )
    content = response.choices[0].message.content.strip()
    evidence_items = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                evidence_items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return evidence_items


def assess_evidence(
    a_class_data: List[Dict],
    b_class_data: List[Dict],
    api_client: OpenAI,
    model: str = "deepseek-v4-pro"
) -> str:
    """Main entry point: analyze evidence patterns and return Playbook EVIDENCE RELIABILITY section."""
    # Step 1: Filter B-class to only "推理扎实"
    high_quality_b = []
    for item in b_class_data:
        reason = item.get('router_reason', '')
        quality = classify_reason_quality(reason, api_client, model)
        if quality == '扎实':
            high_quality_b.append(item)
    print(f"B-class quality filter: {len(high_quality_b)}/{len(b_class_data)} 推理扎实")

    # Step 2: Analyze badcase evidence (unreliable)
    unreliable_evidence_types = {}
    for item in a_class_data[:100]:
        reason = item.get('router_reason', '')
        items = analyze_reason_evidence(reason, api_client, model)
        for ev in items:
            if ev.get('reliability') == '不可靠':
                etype = ev.get('type', '其他')
                if etype not in unreliable_evidence_types:
                    unreliable_evidence_types[etype] = []
                unreliable_evidence_types[etype].append(ev.get('evidence', ''))

    # Step 3: Analyze high-quality B-class evidence (reliable)
    reliable_evidence_types = {}
    for item in high_quality_b[:100]:
        reason = item.get('router_reason', '')
        items = analyze_reason_evidence(reason, api_client, model)
        for ev in items:
            if ev.get('reliability') == '可靠':
                etype = ev.get('type', '其他')
                if etype not in reliable_evidence_types:
                    reliable_evidence_types[etype] = []
                reliable_evidence_types[etype].append(ev.get('evidence', ''))

    # Step 4: Summarize into Playbook section
    unreliable_summary = "\n".join(
        f"类型: {t}, 出现次数: {len(examples)}"
        for t, examples in sorted(unreliable_evidence_types.items(),
                                  key=lambda x: len(x[1]), reverse=True)
    )
    reliable_summary = "\n".join(
        f"类型: {t}, 出现次数: {len(examples)}"
        for t, examples in sorted(reliable_evidence_types.items(),
                                  key=lambda x: len(x[1]), reverse=True)
    )

    prompt = RELIABILITY_SUMMARY_PROMPT.format(
        unreliable_summary=unreliable_summary or "(无)",
        reliable_summary=reliable_summary or "(无)"
    )

    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, '.')
    from utils import initialize_clients

    parser = argparse.ArgumentParser()
    parser.add_argument('--a_data', required=True, help='Path to A-class JSONL')
    parser.add_argument('--b_data', required=True, help='Path to B-class JSONL')
    parser.add_argument('--api_provider', default='openai')
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    def load_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    a_data = load_jsonl(args.a_data)
    b_data = load_jsonl(args.b_data)
    client, _, _ = initialize_clients(args.api_provider)
    section = assess_evidence(a_data, b_data, client, args.model)
    print(section)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(section)
