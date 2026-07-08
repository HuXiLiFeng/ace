"""Aux #1: Compute agent-to-agent confusion matrix from annotated data."""

import json
from collections import defaultdict
from typing import List, Dict, Tuple


def compute_confusion_matrix(raw_data: List[Dict]) -> Dict:
    """Compute confusion matrix from raw annotated data.

    Returns dict with matrix, pairs (sorted by confusion count), and agent_names.
    """
    matrix = defaultdict(lambda: defaultdict(int))
    all_agents = set()

    for item in raw_data:
        pred = (item.get('router_agent') or '').strip()
        true = (item.get('human_annotate_agent') or '').strip()
        if not true:
            true = pred
        if pred and true:
            matrix[pred][true] += 1
            all_agents.add(pred)
            all_agents.add(true)

    pairs = []
    agent_list = sorted(all_agents)
    for i, a in enumerate(agent_list):
        for b in agent_list[i+1:]:
            confusion_count = matrix[a][b] + matrix[b][a]
            if confusion_count > 0:
                pairs.append((a, b, confusion_count))
    pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        'matrix': {k: dict(v) for k, v in matrix.items()},
        'pairs': pairs,
        'agent_names': agent_list,
    }


def format_matrix_report(results: Dict) -> str:
    """Generate a human-readable confusion matrix report."""
    lines = ["# Agent Confusion Matrix", ""]
    lines.append("## Top Confusion Pairs")
    lines.append("")
    for a, b, count in results['pairs'][:5]:
        lines.append(f"- **{a}** ↔ **{b}**: {count}次混淆")
    lines.append("")

    agents = results['agent_names']
    lines.append("## Full Matrix")
    lines.append("")
    header = "| 预测↓ / 真实→ | " + " | ".join(agents) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(agents) + 1))

    for pred in agents:
        row = f"| {pred} |"
        for true in agents:
            count = results['matrix'].get(pred, {}).get(true, 0)
            row += f" {count} |"
        lines.append(row)

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw JSONL data')
    parser.add_argument('--output', default=None, help='Save matrix report to file')
    args = parser.parse_args()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    results = compute_confusion_matrix(data)
    report = format_matrix_report(results)
    print(report)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSaved to {args.output}")
