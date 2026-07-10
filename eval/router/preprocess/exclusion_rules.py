"""Core #4: Extract soft exclusion rules from agent_info.json [注意事项]."""

import json
from openai import OpenAI


EXTRACTION_PROMPT = """你是一个文本校对工具。请对以下每个Agent的【注意事项】做两件事：

1. 指代消解：把"该agent""本agent""不在承接范围"等模糊引用，替换成具体Agent名称
2. 格式整理：保持原文含义不变，整理成清晰易读的表述

规则：
- 只做指代替换和格式整理，**不要归纳、不要总结、不要添加原文没有的信息**
- 原文没有明确写的内容，一律不要编造
- 输出每条注意事项一行，格式: [exc-XXXXX] :: [整理后的注意事项原文]

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
