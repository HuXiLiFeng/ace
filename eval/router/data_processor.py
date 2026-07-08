# eval/router/data_processor.py
"""DataProcessor for the routing task — maps user query to correct agent."""

import json
from typing import List, Dict, Any


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data file."""
    import os
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def load_agent_info(agent_info_path: str) -> str:
    """Load agent_info.json and format it as a string for prompt injection."""
    with open(agent_info_path, 'r', encoding='utf-8') as f:
        agents = json.load(f)

    lines = []
    for agent in agents:
        lines.append(f"Agent名称: {agent['agentName']}")
        lines.append(f"Agent编码: {agent['agentCode']}")
        lines.append(f"职责描述: {agent['agentDescription']}")
        lines.append("---")
    return "\n".join(lines)


def format_history(history: List[str]) -> str:
    """Format dialogue history list into a single string."""
    if not history:
        return "(无对话历史)"
    return "\n".join(history)


class DataProcessor:
    """Processor for the intelligent routing task."""

    def __init__(self, task_name: str = "router", agent_info_path: str = None):
        self.task_name = task_name
        self.agent_info_path = agent_info_path or "task_info/agent_info.json"
        self._agent_info_str = None

    @property
    def agent_info_str(self) -> str:
        if self._agent_info_str is None:
            self._agent_info_str = load_agent_info(self.agent_info_path)
        return self._agent_info_str

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw annotated data to ACE standardized format.

        Raw fields:
          session_id, query, historyDialogue, current_agent,
          router_agent, router_reason, human_note, human_annotate_agent, final_agent

        Standard format:
          context: agent_info + dialogue history + current_agent
          question: the user's query (what agent should handle this?)
          target: the correct agent (human_annotate_agent if available, else router_agent)
          others: all additional metadata for Reflector enrichment
        """
        processed = []
        for item in raw_data:
            # Determine correct answer
            human_label = (item.get('human_annotate_agent') or '').strip()
            router_label = (item.get('router_agent') or '').strip()
            target = human_label if human_label else router_label

            # Build context: agent_info + history + current_agent
            history_str = format_history(item.get('historyDialogue', []))
            current_agent = item.get('current_agent', '')
            context = (
                f"可用Agent列表:\n{self.agent_info_str}\n\n"
                f"对话历史:\n{history_str}\n\n"
                f"当前Agent: {current_agent}"
            )

            # Question = the routing task
            # Embed human_note for Reflector's reasoning chain comparison (Core #1)
            query = item.get('query', '')
            human_note = (item.get('human_note') or '').strip()
            question = (
                f"用户最新问题: {query}\n"
                f"请从可用Agent列表中选择最匹配的Agent全名来处理此请求。"
                f"如果没有任何Agent符合，返回None。"
            )
            if human_note:
                question += f"\n\n【人工标注理由 — 仅在训练Reflector时参考】: {human_note}"

            # Determine data class
            is_badcase = bool(human_label and human_label != router_label)

            processed.append({
                "context": context,
                "question": question,
                "target": target,
                "others": {
                    "session_id": item.get('session_id', ''),
                    "query": query,
                    "history": history_str,
                    "current_agent": current_agent,
                    "router_agent": router_label,
                    "router_reason": item.get('router_reason', ''),
                    "human_note": item.get('human_note', ''),
                    "human_annotate_agent": human_label,
                    "final_agent": item.get('final_agent', ''),
                    "is_badcase": is_badcase,
                    "data_class": "A" if human_label else "B",
                }
            })
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted agent name with ground truth."""
        pred_clean = predicted.strip()
        gt_clean = ground_truth.strip()
        # Exact match
        if pred_clean == gt_clean:
            return True
        # Case-insensitive
        if pred_clean.lower() == gt_clean.lower():
            return True
        # None handling
        if pred_clean == "None" and gt_clean == "None":
            return True
        return False

    def evaluate_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate routing accuracy."""
        if not predictions:
            return 0.0
        correct = sum(
            1 for p, t in zip(predictions, targets)
            if self.answer_is_correct(p, t)
        )
        return correct / len(predictions)
