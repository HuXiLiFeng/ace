"""Core #5: Reverse calibration — ensure new rules don't break existing correct predictions."""

import json
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ReverseCalibrator:
    """Validates new Playbook rules by checking they don't regress on B-class data.

    Sample-size thresholds:
      - >=50 B-class samples: full forward-pass validation
      - 10-49: rules-conflict-only check
      - <10: low-confidence, do not auto-merge
    """

    def __init__(
        self,
        b_class_data: List[Dict],
        conflict_matrix: Dict,
        generator=None,
        playbook: str = "",
    ):
        self.b_class_data = b_class_data
        self.conflict_matrix = conflict_matrix
        self.generator = generator
        self.playbook = playbook
        self._build_index()

    def _build_index(self):
        """Index B-class data by router_agent for efficient sampling."""
        self.by_agent = defaultdict(list)
        for item in self.b_class_data:
            agent = (item.get('router_agent') or '').strip()
            if agent:
                self.by_agent[agent].append(item)

    def _get_affected_agents(self, rule_text: str) -> List[str]:
        """Extract agent names mentioned in a rule."""
        all_agents = list(self.by_agent.keys())
        return [agent for agent in all_agents if agent in rule_text]

    def _get_affected_samples(self, rule_text: str, max_samples: int = 100) -> List[Dict]:
        """Find B-class samples potentially affected by a rule using keyword overlap."""
        keywords = re.findall(r'「([^」]+)」|"([^"]+)"', rule_text)
        keywords = [k[0] or k[1] for k in keywords]

        if not keywords:
            affected_agents = self._get_affected_agents(rule_text)
            samples = []
            for agent in affected_agents:
                samples.extend(self.by_agent.get(agent, []))
            return samples[:max_samples]

        samples = []
        for item in self.b_class_data:
            query = item.get('query', '')
            reason = item.get('router_reason', '')
            text = query + ' ' + reason
            if any(kw in text for kw in keywords):
                samples.append(item)
            if len(samples) >= max_samples:
                break
        return samples

    def _get_sample_count_for_agents(self, agent_a: str, agent_b: str) -> int:
        """Get B-class sample count involving either of two agents."""
        count = 0
        for item in self.b_class_data:
            agent = (item.get('router_agent') or '').strip()
            if agent in (agent_a, agent_b):
                count += 1
        return count

    def check_rule_conflict(self, new_rule: str, existing_playbook: str) -> Tuple[bool, List[str]]:
        """Check if new rule contradicts existing playbook rules."""
        conflicts = []
        new_agents = set(self._get_affected_agents(new_rule))
        existing_rules = re.findall(
            r'\[([a-z]{3,}-\d{5})\]\s+helpful=\d+\s+harmful=\d+\s+::(.*?)(?=\n\[|$)',
            existing_playbook, re.DOTALL
        )
        for rule_id, rule_content in existing_rules:
            rule_agents = set(self._get_affected_agents(rule_content))
            if new_agents & rule_agents:
                conflicts.append(f"可能与 {rule_id} 冲突: {rule_content[:100]}...")
        return len(conflicts) > 0, conflicts

    def validate_rule(
        self,
        rule_text: str,
        target_agent_pair: Optional[Tuple[str, str]] = None
    ) -> Dict:
        """Validate a new rule candidate. Returns dict with status and recommendations."""
        result = {
            'status': 'rejected',
            'affected_samples_count': 0,
            'regressions': 0,
            'conflicts': [],
            'recommendation': '',
        }

        has_conflict, conflicts = self.check_rule_conflict(rule_text, self.playbook)
        result['conflicts'] = conflicts

        if target_agent_pair:
            sample_count = self._get_sample_count_for_agents(*target_agent_pair)
        else:
            affected = self._get_affected_samples(rule_text)
            sample_count = len(affected)

        result['affected_samples_count'] = sample_count

        if sample_count < 10:
            result['status'] = 'low_confidence'
            result['recommendation'] = (
                f'B类样本量不足 ({sample_count} < 10)。规则标记为低置信度，不自动上线。'
                f'建议人工审核后决定。'
            )
            return result

        if sample_count < 50:
            if has_conflict:
                result['status'] = 'rejected'
                result['recommendation'] = (
                    f'B类样本量 ({sample_count}) 不足以做完整反向校准，'
                    f'但检测到 {len(conflicts)} 处规则冲突，拒绝自动上线。'
                )
            else:
                result['status'] = 'low_confidence'
                result['recommendation'] = (
                    f'B类样本量 ({sample_count}) 在10-49之间，仅做规则冲突检测。'
                    f'未检测到冲突，但未做完整回归验证。建议人工审核。'
                )
            return result

        affected_samples = self._get_affected_samples(rule_text)
        result['status'] = 'pending_forward_pass'
        result['recommendation'] = (
            f'B类样本量充足 ({sample_count})。需要执行forward-pass验证。'
            f'受影响样本: {len(affected_samples)} 条。'
        )
        result['affected_samples_for_test'] = affected_samples
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_data', required=True, help='B-class JSONL data')
    parser.add_argument('--rule', required=True, help='New rule text to validate')
    parser.add_argument('--playbook', default='', help='Current playbook text')
    parser.add_argument('--conflict_matrix', default=None)
    args = parser.parse_args()

    def load_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    b_data = load_jsonl(args.b_data)
    conflict_matrix = {}
    if args.conflict_matrix:
        with open(args.conflict_matrix, 'r') as f:
            conflict_matrix = json.load(f)

    calibrator = ReverseCalibrator(b_data, conflict_matrix, playbook=args.playbook)
    result = calibrator.validate_rule(args.rule)
    print(json.dumps(result, ensure_ascii=False, indent=2))
