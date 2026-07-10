"""Build a minimal seed Playbook with only exclusion rules pre-generated."""

import os


def build_seed_playbook(exclusion_rules: str = "") -> str:
    """Seed playbook: only EXCLUSION THRESHOLDS is pre-generated. Everything else ACE fills."""
    sections = []
    sections.append(exclusion_rules or "## EXCLUSION THRESHOLDS\n\n(待训练生成)")
    sections.append("")
    sections.append("## ROUTING STRATEGIES\n\n")
    sections.append("")
    sections.append("## COMMON MISTAKES\n")
    return "\n".join(sections)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build seed Playbook')
    parser.add_argument('--exclusion_rules', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    def load_if(path):
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    playbook = build_seed_playbook(exclusion_rules=load_if(args.exclusion_rules))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(playbook)

    print(f"Seed playbook written to {args.output}")
    print(f"Total length: {len(playbook)} chars, ~{len(playbook)//4} tokens")


if __name__ == "__main__":
    main()
