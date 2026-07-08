"""Core #2: Generate micro-decision-trees for high-confusion agent pairs."""

import json
from typing import List, Dict, Tuple
from openai import OpenAI


MICRO_TREE_PROMPT = """你是一个消费金融智能客服路由系统的业务分析师。以下两个Agent在路由时经常被混淆。

**Agent A: {agent_a}**
职责: {agent_a_desc}

**Agent B: {agent_b}**
职责: {agent_b_desc}

**相关的badcase人工标注理由（human_note）：**
{human_notes}

请从这些human_note中归纳出区分这两个Agent的关键判断维度，并生成一个微决策流程（3-5步）。

要求：
1. 每一步必须有明确的判断条件
2. 每一步至少引用1条具体的human_note作为证据（格式: "human_note证据: 'xxx'"）
3. 步骤之间要有优先级顺序
4. 如果某一步无法判断，明确指向下一步
5. 如果human_note不足（<3条），标注"佐证不足"

输出格式（结构化文本）:

### 决策分支: {agent_a} vs {agent_b}

佐证human_note数量: X

STEP 1: [步骤名称]
  判断: [具体判断条件]
  → 指向{agent_a}的条件: [条件]
  → 指向{agent_b}的条件: [条件]
  → 无法判断 → STEP 2
  human_note证据: "[引用具体human_note]"

STEP 2: ...
"""

TOP_ROUTER_PROMPT = """你是一个消费金融智能客服路由系统。请生成一个顶层路由框架，用于决定query应该使用哪个微决策流程。

可用的微决策流程（每个针对一对高频混淆Agent）:
{micro_trees_summary}

所有Agent列表:
{agent_list}

排除规则（来自EXCLUSION THRESHOLDS）:
{exclusion_rules}

请生成一个顶层路由流程：
1. 首先执行排除规则，缩小候选范围
2. 判断query最可能涉及哪个混淆对的风险区域
3. 路由到对应的微决策流程
4. 如果不在任何已知高风险区域 → 使用通用匹配流程

输出格式（结构化文本，放入Playbook的DECISION FRAMEWORK章节）:

## DECISION FRAMEWORK

### 顶层路由

STEP 0.1: 排除检查
  [排除规则简述]

STEP 0.2: 风险区域判断
  [如何判断query属于哪个混淆对]

STEP 0.3: 通用匹配（无高风险区域时）
  [通用匹配策略]

### 微决策流程

[在此插入各混淆对的微决策树]
"""


def load_agent_info_text(agent_info_path: str) -> str:
    """Load and format agent info."""
    with open(agent_info_path, 'r', encoding='utf-8') as f:
        agents = json.load(f)
    lines = []
    for a in agents:
        lines.append(f"## {a['agentName']}\n{a['agentDescription']}\n")
    return "\n".join(lines)


def get_agent_description(agent_info_path: str, agent_name: str) -> str:
    """Get the description for a specific agent by name."""
    with open(agent_info_path, 'r', encoding='utf-8') as f:
        agents = json.load(f)
    for a in agents:
        if a['agentName'] == agent_name:
            return a['agentDescription']
    return ""


def generate_micro_tree(
    agent_a: str,
    agent_b: str,
    human_notes: List[str],
    agent_info_path: str,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro"
) -> str:
    """Generate a micro-decision-tree for a specific agent confusion pair."""
    desc_a = get_agent_description(agent_info_path, agent_a)
    desc_b = get_agent_description(agent_info_path, agent_b)
    notes_text = "\n".join(f"- {n}" for n in human_notes[:20])

    prompt = MICRO_TREE_PROMPT.format(
        agent_a=agent_a, agent_b=agent_b,
        agent_a_desc=desc_a, agent_b_desc=desc_b,
        human_notes=notes_text
    )

    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2,
    )

    result = response.choices[0].message.content.strip()
    if len(human_notes) < 3:
        result = "⚠️ 低置信度（佐证不足，不自动上线）\n\n" + result
    return result


def generate_decision_framework(
    conflict_pairs: List[Tuple[str, str, int]],
    human_notes_by_pair: Dict[Tuple[str, str], List[str]],
    agent_info_path: str,
    exclusion_rules_text: str,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro"
) -> str:
    """Generate the full DECISION FRAMEWORK section for the Playbook."""
    micro_trees = []
    micro_tree_summaries = []

    for agent_a, agent_b, count in conflict_pairs[:5]:
        pair_key = (agent_a, agent_b)
        notes = human_notes_by_pair.get(pair_key, [])
        if len(notes) == 0:
            continue
        print(f"  Generating micro-tree for: {agent_a} vs {agent_b} ({len(notes)} notes)")
        tree = generate_micro_tree(agent_a, agent_b, notes, agent_info_path, api_client, model)
        micro_trees.append(tree)
        micro_tree_summaries.append(
            f"- {agent_a} vs {agent_b} (混淆{count}次, {len(notes)}条human_note)"
        )

    agent_info_text = load_agent_info_text(agent_info_path)
    prompt = TOP_ROUTER_PROMPT.format(
        micro_trees_summary="\n".join(micro_tree_summaries),
        agent_list=agent_info_text,
        exclusion_rules=exclusion_rules_text or "(待生成)"
    )

    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2,
    )

    top_router = response.choices[0].message.content.strip()
    section = top_router + "\n\n" + "\n\n".join(micro_trees)
    return section


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, '.')
    from utils import initialize_clients

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_info', default='task_info/agent_info.json')
    parser.add_argument('--a_data', required=True, help='A-class JSONL with human_notes')
    parser.add_argument('--conflict_matrix_json', help='Pre-computed conflict matrix JSON')
    parser.add_argument('--exclusion_rules_file', default=None)
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

    human_notes_by_pair = {}
    for item in a_data:
        human_note = (item.get('human_note') or '').strip()
        human_label = (item.get('human_annotate_agent') or '').strip()
        router_label = (item.get('router_agent') or '').strip()
        if human_note and human_label and router_label and human_label != router_label:
            pair = tuple(sorted([human_label, router_label]))
            if pair not in human_notes_by_pair:
                human_notes_by_pair[pair] = []
            human_notes_by_pair[pair].append(human_note)

    if args.conflict_matrix_json:
        with open(args.conflict_matrix_json, 'r') as f:
            matrix_data = json.load(f)
        conflict_pairs = [(p[0], p[1], p[2]) for p in matrix_data.get('pairs', [])]
    else:
        conflict_pairs = [
            (a, b, len(notes))
            for (a, b), notes in
            sorted(human_notes_by_pair.items(), key=lambda x: len(x[1]), reverse=True)
        ]

    exclusion_rules = ""
    if args.exclusion_rules_file:
        with open(args.exclusion_rules_file, 'r', encoding='utf-8') as f:
            exclusion_rules = f.read()

    client, _, _ = initialize_clients(args.api_provider)
    section = generate_decision_framework(
        conflict_pairs, human_notes_by_pair,
        args.agent_info, exclusion_rules,
        client, args.model
    )
    print(section)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(section)
