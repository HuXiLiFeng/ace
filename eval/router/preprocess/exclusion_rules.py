"""Core #4: Extract soft exclusion rules from agent_info.json [注意事项]."""

import json
from openai import OpenAI


EXTRACTION_PROMPT = """你是一个消费金融客服系统的业务分析师。请分析以下Agent的【注意事项】，提取所有"不属于本Agent职责范围"的排除规则。

对每条排除规则，请指明：
1. 什么意图/操作被排除
2. 从哪个Agent排除
3. 应该归属于哪个Agent（如果能从其他Agent的职责中推断）
4. 排除原因

输出格式（每条一行，纯文本）:
[exc-XXXXX] :: 意图 "[意图描述]" → 降低 "[被排除Agent]" 优先级（阈值+0.3），归属 "[正确Agent]" → 原因: [简短原因]

如果无法确定正确归属，归属写"待确认"。

Agent信息:
{agent_info}
"""


def extract_exclusion_rules(
    agent_info_path: str,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro",
    max_tokens: int = 4096
) -> str:
    """Extract soft exclusion rules from agent_info.json. Returns EXCLUSION THRESHOLDS section text."""
    with open(agent_info_path, 'r', encoding='utf-8') as f:
        agents = json.load(f)

    agent_text_parts = []
    for agent in agents:
        desc = agent['agentDescription']
        if '【注意事项】' in desc:
            notes_start = desc.index('【注意事项】')
            notes = desc[notes_start:]
        else:
            notes = '(无注意事项)'
        agent_text_parts.append(
            f"Agent: {agent['agentName']}\n注意事项: {notes}\n"
        )
    agent_info_text = "\n".join(agent_text_parts)

    prompt = EXTRACTION_PROMPT.format(agent_info=agent_info_text)

    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )

    rules_text = response.choices[0].message.content.strip()
    section = "## EXCLUSION THRESHOLDS\n\n" + rules_text
    return section


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, '.')
    from utils import initialize_clients

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_info', default='task_info/agent_info.json')
    parser.add_argument('--api_provider', default='openai')
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--output', default=None, help='Save to file')
    args = parser.parse_args()

    client, _, _ = initialize_clients(args.api_provider)
    section = extract_exclusion_rules(args.agent_info, client, args.model)
    print(section)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(section)
        print(f"\nSaved to {args.output}")
