"""Assemble a seeded Playbook from all pre-processing outputs."""

import os


def build_seed_playbook(
    exclusion_rules: str = "",
    decision_framework: str = "",
    evidence_reliability: str = "",
    routing_strategies: str = "",
) -> str:
    """Assemble all pre-generated sections into a complete seed playbook.

    Order follows the Generator's decision process:
      1. EXCLUSION THRESHOLDS
      2. DECISION FRAMEWORK
      3. EVIDENCE RELIABILITY
      4. ROUTING STRATEGIES
      5. COMMON MISTAKES (ACE fills during training)
    """
    sections = []

    sections.append(exclusion_rules or "## EXCLUSION THRESHOLDS\n\n(待训练生成)")
    sections.append("")
    sections.append(decision_framework or "## DECISION FRAMEWORK\n\n(待训练生成)")
    sections.append("")
    sections.append(evidence_reliability or "## EVIDENCE RELIABILITY\n\n(待训练生成)")
    sections.append("")
    sections.append(routing_strategies or "## ROUTING STRATEGIES\n\n(待训练生成)")
    sections.append("")
    sections.append("## COMMON MISTAKES\n")

    return "\n".join(sections)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build seed Playbook')
    parser.add_argument('--exclusion_rules', default=None)
    parser.add_argument('--decision_framework', default=None)
    parser.add_argument('--evidence_reliability', default=None)
    parser.add_argument('--routing_strategies', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    def load_if(path):
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    playbook = build_seed_playbook(
        exclusion_rules=load_if(args.exclusion_rules),
        decision_framework=load_if(args.decision_framework),
        evidence_reliability=load_if(args.evidence_reliability),
        routing_strategies=load_if(args.routing_strategies),
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(playbook)

    print(f"Seed playbook written to {args.output}")
    print(f"Total length: {len(playbook)} chars, ~{len(playbook)//4} tokens")


if __name__ == "__main__":
    main()
