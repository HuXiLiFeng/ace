# ACE Routing Experience Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline pipeline that extracts routing business experience from annotated badcase data into an evolving ACE Playbook, then injects it into the existing routing prompt to improve qwen3-20b-a3b routing accuracy.

**Architecture:** New `eval/router/` task directory mirrors `eval/finance/` pattern. Five pre-processing modules (exclusion rules, evidence assessment, decision framework, conflict matrix, reverse calibration) run before/during ACE training to seed and validate the Playbook. Custom Generator/Reflector prompts enable reasoning-chain comparison learning. ACE core framework (`ace/`) is NOT modified.

**Tech Stack:** Python 3, ACE framework (existing), OpenAI-compatible API (DeepSeek V4 for offline), qwen3-20b-a3b (online target)

## Global Constraints

- ACE core framework (`ace/`, `utils.py`, `playbook_utils.py`, `llm.py`, `logger.py`) must NOT be modified
- All routing-specific code goes in `eval/router/`
- DeepSeek V4 used for offline knowledge extraction (via OpenAI-compatible API with custom base URL)
- Agent info loaded from `task_info/agent_info.json` at runtime
- Annotated data expected as JSONL with fields: session_id, query, historyDialogue, current_agent, router_agent, router_reason, human_note, human_annotate_agent, final_agent
- A类 (badcase): human_note and human_annotate_agent are non-empty
- B类 (准正确): human_note and human_annotate_agent are empty/null
- badcase < 100: strategies from #1 generate analysis reports only, not auto-rules
- badcase < 200: enable adversarial augmentation (#2 auxiliary)
- Decision framework updates only when ≥50 new badcases accumulated
- Reverse calibration sample thresholds: ≥50 full calibration, <50 rules-conflict-only, <10 no-go

---

## File Structure

```
eval/router/
├── __init__.py                          # Empty init
├── data_processor.py                    # Task 1: DataProcessor for routing task
├── run.py                               # Task 10: Training script entry point
├── data/
│   └── sample_config.json              # Task 2: Data path configuration
├── prompts/
│   ├── __init__.py                      # Empty init
│   ├── generator.py                     # Task 7: Custom Generator prompt
│   └── reflector.py                     # Task 6: Custom Reflector prompt
├── preprocess/
│   ├── __init__.py                      # Empty init
│   ├── data_prep.py                     # Task 2: Raw JSON → ACE JSONL conversion
│   ├── exclusion_rules.py              # Task 3: Core #4 — extract from agent_info
│   ├── conflict_matrix.py              # Task 4: Aux #1 — agent confusion stats
│   ├── evidence_assessment.py          # Task 5: Core #3 — evidence reliability patterns
│   └── decision_framework.py           # Task 8: Core #2 — micro-decision-trees
├── calibrate/
│   ├── __init__.py                      # Empty init
│   └── reverse_calibrate.py            # Task 9: Core #5 — regression testing
├── augment/
│   ├── __init__.py                      # Empty init
│   └── adversarial.py                  # Task 12: Aux #2 — adversarial sample generation
└── seed_playbook.py                     # Task 11: Assemble initial seeded Playbook
```

---

### Task 1: DataProcessor for Routing Task

**Files:**
- Create: `eval/router/__init__.py`
- Create: `eval/router/data_processor.py`

**Interfaces:**
- Consumes: Nothing (foundation task)
- Produces: `DataProcessor(task_name: str)` class with `process_task_data()`, `answer_is_correct()`, `evaluate_accuracy()`

- [ ] **Step 1: Create empty init**

```bash
touch eval/router/__init__.py
```

- [ ] **Step 2: Write DataProcessor**

```python
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
            query = item.get('query', '')
            question = (
                f"用户最新问题: {query}\n"
                f"请从可用Agent列表中选择最匹配的Agent全名来处理此请求。"
                f"如果没有任何Agent符合，返回None。"
            )

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
```

- [ ] **Step 3: Verify imports**

```bash
cd e:/github/ace && uv run python -c "from eval.router.data_processor import DataProcessor; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Test with data_sample.json**

```python
# Quick test in Python
from eval.router.data_processor import DataProcessor, load_data

dp = DataProcessor(agent_info_path="task_info/agent_info.json")
raw = load_data("task_info/data_sample.json")
processed = dp.process_task_data(raw)
assert len(processed) == 1
assert processed[0]["target"] == "逾期及APP使用"
assert processed[0]["others"]["is_badcase"] == True
assert processed[0]["others"]["data_class"] == "A"
print(f"Context length: {len(processed[0]['context'])} chars")
print(f"Target agent: {processed[0]['target']}")
print("All assertions passed.")
```

- [ ] **Step 5: Commit**

```bash
git add eval/router/__init__.py eval/router/data_processor.py
git commit -m "feat: add DataProcessor for routing task

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Data Preparation Script

**Files:**
- Create: `eval/router/preprocess/__init__.py`
- Create: `eval/router/preprocess/data_prep.py`
- Create: `eval/router/data/sample_config.json`

**Interfaces:**
- Consumes: `DataProcessor` from Task 1
- Produces: `prepare_splits(raw_path, output_dir, train_ratio, val_ratio)` function

- [ ] **Step 1: Create directories**

```bash
mkdir -p eval/router/preprocess eval/router/data
touch eval/router/preprocess/__init__.py
```

- [ ] **Step 2: Write data preparation script**

```python
# eval/router/preprocess/data_prep.py
"""Convert raw annotated JSONL to ACE-compatible train/val/test splits."""

import json
import os
import random
from typing import List, Dict, Tuple


def load_raw_data(path: str) -> List[Dict]:
    """Load raw JSONL data."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_data(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test, stratified by data_class (A/B)."""
    random.seed(seed)

    a_class = [d for d in data if (d.get('human_annotate_agent') or '').strip()]
    b_class = [d for d in data if not (d.get('human_annotate_agent') or '').strip()]

    random.shuffle(a_class)
    random.shuffle(b_class)

    def split_list(lst):
        n = len(lst)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    a_train, a_val, a_test = split_list(a_class)
    b_train, b_val, b_test = split_list(b_class)

    return (
        a_train + b_train,
        a_val + b_val,
        a_test + b_test
    )


def save_jsonl(data: List[Dict], path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {path}")


def prepare_splits(
    raw_data_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, str]:
    """
    Main entry point: load raw data, split, save.

    Returns dict with paths: train_data, val_data, test_data
    """
    raw_data = load_raw_data(raw_data_path)
    train, val, test = split_data(raw_data, train_ratio, val_ratio, seed)

    paths = {
        'train_data': os.path.join(output_dir, 'train.jsonl'),
        'val_data': os.path.join(output_dir, 'val.jsonl'),
        'test_data': os.path.join(output_dir, 'test.jsonl'),
    }

    save_jsonl(train, paths['train_data'])
    save_jsonl(val, paths['val_data'])
    save_jsonl(test, paths['test_data'])

    # Print stats
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        a_count = sum(1 for d in split_data if (d.get('human_annotate_agent') or '').strip())
        b_count = len(split_data) - a_count
        print(f"  {split_name}: {len(split_data)} total ({a_count} A-class, {b_count} B-class)")

    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw JSONL data')
    parser.add_argument('--output_dir', default='./eval/router/data', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    paths = prepare_splits(args.input, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
    print(f"\nOutput files:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
```

- [ ] **Step 3: Write sample config**

```json
# eval/router/data/sample_config.json
{
    "router": {
        "train_data": "./eval/router/data/train.jsonl",
        "val_data": "./eval/router/data/val.jsonl",
        "test_data": "./eval/router/data/test.jsonl"
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add eval/router/preprocess/__init__.py eval/router/preprocess/data_prep.py eval/router/data/sample_config.json
git commit -m "feat: add data preparation script for routing task

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Exclusion Rules Extractor (Core #4)

**Files:**
- Create: `eval/router/preprocess/exclusion_rules.py`

**Interfaces:**
- Consumes: `task_info/agent_info.json`
- Produces: `extract_exclusion_rules(agent_info_path, api_client, model) -> str` — returns Playbook-ready EXCLUSION THRESHOLDS section text

- [ ] **Step 1: Write exclusion rules extractor**

```python
# eval/router/preprocess/exclusion_rules.py
"""Core #4: Extract soft exclusion rules from agent_info.json [注意事项]."""

import json
import os
from openai import OpenAI


EXTRACTION_PROMPT = """你是一个消费金融客服系统的业务分析师。请分析以下6个Agent的【注意事项】，提取所有"不属于本Agent职责范围"的排除规则。

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
    """
    Extract soft exclusion rules from agent_info.json.

    Args:
        agent_info_path: Path to agent_info.json
        api_client: OpenAI-compatible client
        model: Model name for extraction
        max_tokens: Max tokens for response

    Returns:
        Formatted EXCLUSION THRESHOLDS section text ready for Playbook
    """
    with open(agent_info_path, 'r', encoding='utf-8') as f:
        agents = json.load(f)

    # Build agent info text with [注意事项] highlighted
    agent_text_parts = []
    for agent in agents:
        desc = agent['agentDescription']
        # Extract 注意事项 section
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

    # Wrap in section header
    section = "## EXCLUSION THRESHOLDS\n\n" + rules_text

    return section


if __name__ == "__main__":
    import argparse
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
```

- [ ] **Step 2: Test dry-run**

```bash
cd e:/github/ace && uv run python -c "
from eval.router.preprocess.exclusion_rules import extract_exclusion_rules
# Dry-run: just verify imports and function signature
print('Module loaded OK')
print('Function signature: extract_exclusion_rules(agent_info_path, api_client, model) -> str')
"
```

Expected: `Module loaded OK`

- [ ] **Step 3: Commit**

```bash
git add eval/router/preprocess/exclusion_rules.py
git commit -m "feat: add exclusion rules extractor (Core #4)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Conflict Matrix Calculator (Aux #1)

**Files:**
- Create: `eval/router/preprocess/conflict_matrix.py`

**Interfaces:**
- Consumes: Processed data from Task 1 (or raw JSONL)
- Produces: `compute_confusion_matrix(data, data_processor) -> Dict` with matrix + ranked confusion pairs

- [ ] **Step 1: Write conflict matrix calculator**

```python
# eval/router/preprocess/conflict_matrix.py
"""Aux #1: Compute agent-to-agent confusion matrix from annotated data."""

import json
from collections import defaultdict
from typing import List, Dict, Tuple


def compute_confusion_matrix(
    raw_data: List[Dict]
) -> Dict:
    """
    Compute confusion matrix from raw annotated data.

    Args:
        raw_data: List of raw data dicts with router_agent and human_annotate_agent fields

    Returns:
        Dict with:
          - matrix: {pred_agent: {true_agent: count}}
          - pairs: [(agent_a, agent_b, confusion_count)], sorted descending
          - agent_names: list of all unique agent names
    """
    matrix = defaultdict(lambda: defaultdict(int))
    all_agents = set()

    for item in raw_data:
        pred = (item.get('router_agent') or '').strip()
        true = (item.get('human_annotate_agent') or '').strip()

        # For B-class data (no human annotation), true = pred
        if not true:
            true = pred

        if pred and true:
            matrix[pred][true] += 1
            all_agents.add(pred)
            all_agents.add(true)

    # Build sorted confusion pairs (only off-diagonal)
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

    # Summary of top confusion pairs
    lines.append("## Top Confusion Pairs")
    lines.append("")
    for a, b, count in results['pairs'][:5]:
        lines.append(f"- **{a}** ↔ **{b}**: {count}次混淆")
    lines.append("")

    # Full matrix as markdown table
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

    # Load data
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
```

- [ ] **Step 2: Commit**

```bash
git add eval/router/preprocess/conflict_matrix.py
git commit -m "feat: add agent confusion matrix calculator (Aux #1)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Evidence Quality Assessment (Core #3)

**Files:**
- Create: `eval/router/preprocess/evidence_assessment.py`

**Interfaces:**
- Consumes: Processed A-class and B-class data
- Produces: `assess_evidence(router_reasons, api_client, model) -> str` — returns Playbook EVIDENCE RELIABILITY section

- [ ] **Step 1: Write evidence assessment module**

```python
# eval/router/preprocess/evidence_assessment.py
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
    result = response.choices[0].message.content.strip()
    return result


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

    # Parse JSON lines
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
    """
    Main entry point: analyze evidence patterns from A-class (badcase) and B-class (correct) data.

    Args:
        a_class_data: List of badcase items with router_reason
        b_class_data: List of correct items with router_reason
        api_client: OpenAI-compatible client
        model: Model name

    Returns:
        Playbook EVIDENCE RELIABILITY section text
    """
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
    for item in a_class_data[:100]:  # Cap at 100 for cost
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
    from utils import initialize_clients

    parser = argparse.ArgumentParser()
    parser.add_argument('--a_data', required=True, help='Path to A-class JSONL')
    parser.add_argument('--b_data', required=True, help='Path to B-class JSONL')
    parser.add_argument('--api_provider', default='openai')
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # Load data
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
```

- [ ] **Step 2: Commit**

```bash
git add eval/router/preprocess/evidence_assessment.py
git commit -m "feat: add evidence quality assessment (Core #3)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Custom Reflector Prompt (Core #1)

**Files:**
- Create: `eval/router/prompts/__init__.py`
- Create: `eval/router/prompts/reflector.py`

**Interfaces:**
- Consumes: Nothing
- Produces: `ROUTING_REFLECTOR_PROMPT` — string template with {human_note} and {router_reason} fields

- [ ] **Step 1: Create directories and init**

```bash
mkdir -p eval/router/prompts
touch eval/router/prompts/__init__.py
```

- [ ] **Step 2: Write custom Reflector prompt**

```python
# eval/router/prompts/reflector.py
"""Core #1: Custom Reflector prompt with reasoning chain comparison."""

ROUTING_REFLECTOR_PROMPT = """You are an expert analyst of intelligent customer service routing systems. Your job is to diagnose why the model's routing decision went wrong by comparing the model's reasoning chain with the human annotator's reasoning chain.

**Instructions:**
- Carefully analyze BOTH the model's reasoning trace AND the human annotator's note
- Identify the specific judgment strategy the model used vs. what the human used
- If the human_note is very short (<20 characters), only use explicitly stated information — do NOT over-infer the human's reasoning process
- If the human_note is too brief to determine the reasoning chain, note "信息不足" in your analysis
- Provide actionable insights about STRATEGY SELECTION — not just "what" went wrong, but "how" the model should have approached the decision
- You will receive bulletpoints from the playbook that the generator used. Tag each as helpful/harmful/neutral.

**Your output should be a json object with these fields:**
- reasoning: your chain of thought, detailed analysis comparing the two reasoning paths
- strategy_analysis: what strategy did the model use? what strategy did the human use? what's the essential difference?
- strategy_selection_rule: if the error was caused by using the wrong strategy, describe when to use which strategy (leave empty if human_note is too short to infer)
- error_identification: what specifically went wrong in the model's reasoning?
- root_cause_analysis: why did this error occur? what concept or priority was misunderstood?
- correct_approach: what should the model have done instead?
- key_insight: what strategy, principle, or rule should be remembered to avoid this error?
- human_note_sufficient: true/false — was the human_note detailed enough to extract strategy-level insights?
- bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator

**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer (Human Annotator):**
{}

**Human Annotator's Reasoning Note:**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Detailed analysis comparing model vs human reasoning paths]",
  "strategy_analysis": "[Model strategy vs human strategy — what's the essential difference?]",
  "strategy_selection_rule": "[When to use which strategy — only if human_note is sufficient]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy or principle should be remembered?]",
  "human_note_sufficient": true,
  "bullet_tags": [
    {{"id": "str-00001", "tag": "helpful"}},
    {{"id": "err-00002", "tag": "harmful"}}
  ]
}}

---
"""
```

- [ ] **Step 3: Commit**

```bash
git add eval/router/prompts/__init__.py eval/router/prompts/reflector.py
git commit -m "feat: add custom Reflector prompt with reasoning chain comparison (Core #1)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Custom Generator Prompt

**Files:**
- Create: `eval/router/prompts/generator.py`

**Interfaces:**
- Consumes: Nothing
- Produces: `ROUTING_GENERATOR_PROMPT` — string template

- [ ] **Step 1: Write custom Generator prompt**

```python
# eval/router/prompts/generator.py
"""Custom Generator prompt for the routing task with Playbook integration."""

ROUTING_GENERATOR_PROMPT = """You are an intelligent customer service router. Your task is to analyze the user's latest query and select the most appropriate agent from the available agent list.

**Playbook (Accumulated Business Experience):**
{}

**Reflection (Previous mistakes to avoid):**
{}

**Routing Task:**
{}

**Context (Agent info, dialogue history, current agent):**
{}

**Decision Process:**
1. First, check the EXCLUSION THRESHOLDS in the playbook — which agents are unlikely to handle this query?
2. Then, check the DECISION FRAMEWORK in the playbook — does this query fall into a known high-risk confusion pair? If so, follow the micro-decision-tree for that pair.
3. Check the EVIDENCE RELIABILITY section — what type of evidence should you trust? What type is unreliable?
4. Check ROUTING STRATEGIES for applicable strategy rules.
5. Check COMMON MISTAKES — have similar queries been misrouted before?
6. Make your final decision.

**Output format (strict JSON):**
{{
  "reasoning": "[Step-by-step analysis following the decision process above. Cite specific playbook rules you used.]",
  "bullet_ids": ["str-00001", "evi-00005"],
  "final_answer": "[Agent full name, or 'None' if no agent matches]"
}}

**IMPORTANT:**
- final_answer MUST be one of the exact agent names from the available agent list, or "None"
- If you find the query similar to a COMMON MISTAKE entry, explicitly address why you're not making the same mistake
- Prefer direct intent-keyword matching over conversational context inference

---
"""
```

- [ ] **Step 2: Commit**

```bash
git add eval/router/prompts/generator.py
git commit -m "feat: add custom Generator prompt for routing task

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Decision Framework Generator (Core #2)

**Files:**
- Create: `eval/router/preprocess/decision_framework.py`

**Interfaces:**
- Consumes: Conflict matrix (from Task 4), human_notes from A-class data, agent_info.json
- Produces: `generate_decision_framework(conflict_pairs, human_notes, agent_info_text, api_client, model) -> str` — returns Playbook DECISION FRAMEWORK section

- [ ] **Step 1: Write decision framework generator**

```python
# eval/router/preprocess/decision_framework.py
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
...
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

    notes_text = "\n".join(f"- {n}" for n in human_notes[:20])  # Cap at 20

    prompt = MICRO_TREE_PROMPT.format(
        agent_a=agent_a,
        agent_b=agent_b,
        agent_a_desc=desc_a,
        agent_b_desc=desc_b,
        human_notes=notes_text
    )

    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2,
    )

    result = response.choices[0].message.content.strip()

    # If human_notes < 3, prepend low-confidence marker
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
    """
    Generate the full DECISION FRAMEWORK section.

    Args:
        conflict_pairs: List of (agent_a, agent_b, confusion_count) sorted descending
        human_notes_by_pair: Dict mapping (agent_a, agent_b) -> list of human_note strings
        agent_info_path: Path to agent_info.json
        exclusion_rules_text: Pre-generated EXCLUSION THRESHOLDS text
        api_client: OpenAI-compatible client
        model: Model name

    Returns:
        Playbook DECISION FRAMEWORK section text
    """
    # Generate micro-trees for top-N confusion pairs
    micro_trees = []
    micro_tree_summaries = []

    for agent_a, agent_b, count in conflict_pairs[:5]:  # Top 5 pairs
        pair_key = (agent_a, agent_b)
        notes = human_notes_by_pair.get(pair_key, [])
        if len(notes) < 3:
            print(f"  Skipping {agent_a} vs {agent_b}: only {len(notes)} human_notes (<3)")
            # Still generate but mark low-confidence
            if len(notes) == 0:
                continue

        print(f"  Generating micro-tree for: {agent_a} vs {agent_b} ({len(notes)} notes)")
        tree = generate_micro_tree(agent_a, agent_b, notes, agent_info_path, api_client, model)
        micro_trees.append(tree)
        micro_tree_summaries.append(f"- {agent_a} vs {agent_b} (混淆{count}次, {len(notes)}条human_note)")

    # Generate top-level router
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

    # Assemble full section
    section = top_router + "\n\n" + "\n\n".join(micro_trees)
    return section


if __name__ == "__main__":
    import argparse
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

    # Load data
    def load_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    a_data = load_jsonl(args.a_data)

    # Build human_notes_by_pair from A-class data
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

    # Get conflict pairs (from pre-computed matrix or derive from data)
    if args.conflict_matrix_json:
        with open(args.conflict_matrix_json, 'r') as f:
            matrix_data = json.load(f)
        conflict_pairs = [(p[0], p[1], p[2]) for p in matrix_data.get('pairs', [])]
    else:
        # Derive from human_notes_by_pair
        conflict_pairs = [
            (a, b, len(notes))
            for (a, b), notes in
            sorted(human_notes_by_pair.items(), key=lambda x: len(x[1]), reverse=True)
        ]

    # Load exclusion rules
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
```

- [ ] **Step 2: Commit**

```bash
git add eval/router/preprocess/decision_framework.py
git commit -m "feat: add decision framework generator (Core #2)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Reverse Calibration Module (Core #5)

**Files:**
- Create: `eval/router/calibrate/__init__.py`
- Create: `eval/router/calibrate/reverse_calibrate.py`

**Interfaces:**
- Consumes: B-class data (regression set), newly generated rules
- Produces: `ReverseCalibrator` class with `validate_rule(rule_text, b_class_data, generator, playbook) -> Dict`

- [ ] **Step 1: Create directory**

```bash
mkdir -p eval/router/calibrate
touch eval/router/calibrate/__init__.py
```

- [ ] **Step 2: Write reverse calibration module**

```python
# eval/router/calibrate/reverse_calibrate.py
"""Core #5: Reverse calibration — ensure new rules don't break existing correct predictions."""

import json
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ReverseCalibrator:
    """
    Validates new Playbook rules by checking they don't regress on B-class data.

    Uses sample-size thresholds:
      - ≥50 B-class samples for the affected agent pair: full forward-pass validation
      - 10-49: rules-conflict-only check
      - <10: mark as low-confidence, do not auto-merge
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

        # Index B-class data by agent for fast retrieval
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
        # Simple approach: find agent names that appear in the rule text
        all_agents = list(self.by_agent.keys())
        mentioned = []
        for agent in all_agents:
            if agent in rule_text:
                mentioned.append(agent)
        return mentioned

    def _get_affected_samples(
        self, rule_text: str, max_samples: int = 100
    ) -> List[Dict]:
        """Find B-class samples potentially affected by a rule using keyword overlap."""
        # Extract key terms from rule
        keywords = re.findall(r'「([^」]+)」|"([^"]+)"', rule_text)
        keywords = [k[0] or k[1] for k in keywords]

        if not keywords:
            # Fallback: use agent-based filtering
            affected_agents = self._get_affected_agents(rule_text)
            samples = []
            for agent in affected_agents:
                samples.extend(self.by_agent.get(agent, []))
            return samples[:max_samples]

        # Keyword-based retrieval
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

    def check_rule_conflict(
        self, new_rule: str, existing_playbook: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if a new rule contradicts existing playbook rules.
        Uses simple heuristic: if two rules mention the same agent pair
        but suggest opposite routing, flag as conflict.

        Returns (has_conflict, conflict_descriptions).
        """
        conflicts = []
        new_agents = set(self._get_affected_agents(new_rule))

        # Extract existing rules
        existing_rules = re.findall(
            r'\[([a-z]{3,}-\d{5})\]\s+helpful=\d+\s+harmful=\d+\s+::(.*?)(?=\n\[|$)',
            existing_playbook, re.DOTALL
        )

        for rule_id, rule_content in existing_rules:
            rule_agents = set(self._get_affected_agents(rule_content))
            if new_agents & rule_agents:  # Overlap in mentioned agents
                # Simple heuristic: if new rule says "优先选A" and existing says "优先选B"
                # flag as potential conflict
                conflicts.append(f"可能与 {rule_id} 冲突: {rule_content[:100]}...")

        return len(conflicts) > 0, conflicts

    def validate_rule(
        self,
        rule_text: str,
        target_agent_pair: Optional[Tuple[str, str]] = None
    ) -> Dict:
        """
        Validate a new rule candidate.

        Returns dict with:
          - status: "approved" | "modified" | "rejected" | "low_confidence"
          - affected_samples_count: int
          - regressions: int (number of B-class samples broken)
          - conflicts: List[str]
          - recommendation: str
        """
        result = {
            'status': 'rejected',
            'affected_samples_count': 0,
            'regressions': 0,
            'conflicts': [],
            'recommendation': '',
        }

        # Step 1: Check for rule conflicts with existing playbook
        has_conflict, conflicts = self.check_rule_conflict(rule_text, self.playbook)
        result['conflicts'] = conflicts

        # Step 2: Determine sample adequacy
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
            # Rules-conflict-only mode
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

        # Step 3: Full forward-pass validation (sample_count >= 50)
        # NOTE: Full forward-pass requires the Generator and is done in Task 10
        # during ACE training. Here we prepare the validation set and return
        # a recommendation to proceed with forward-pass.

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

    # Load data
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
```

- [ ] **Step 3: Commit**

```bash
git add eval/router/calibrate/__init__.py eval/router/calibrate/reverse_calibrate.py
git commit -m "feat: add reverse calibration module (Core #5)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Training Script (run.py)

**Files:**
- Create: `eval/router/run.py`

**Interfaces:**
- Consumes: Tasks 1, 6, 7 (DataProcessor, Generator prompt, Reflector prompt)
- Produces: Runnable `python -m eval.router.run` CLI

- [ ] **Step 1: Write training script**

```python
# eval/router/run.py
"""ACE training script for the intelligent routing task."""

import os
import json
import argparse

from .data_processor import DataProcessor
from utils import initialize_clients


def parse_args():
    parser = argparse.ArgumentParser(description='ACE Router Training')

    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"])
    parser.add_argument("--api_provider", type=str, default="openai")
    parser.add_argument("--generator_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--reflector_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--curator_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--test_workers", type=int, default=20)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument("--agent_info_path", type=str, default="task_info/agent_info.json")
    parser.add_argument("--seed_playbook_path", type=str, default=None,
                        help="Path to pre-generated seeded playbook (from Task 11)")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--curator_batch_size", type=int, default=None)
    parser.add_argument("--augmented_shuffling", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def load_data(data_path: str):
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


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"ACE ROUTER SYSTEM — {args.mode.upper()}")
    print(f"{'='*60}")

    # Load config
    with open("./eval/router/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    router_config = task_config["router"]
    processor = DataProcessor(agent_info_path=args.agent_info_path)

    # Load data based on mode
    if args.mode in ["online", "eval_only"]:
        test_samples = load_data(router_config["test_data"])
        test_samples = processor.process_task_data(test_samples)
        train_samples = None
        val_samples = None
    else:
        train_samples = load_data(router_config["train_data"])
        val_samples = load_data(router_config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        if "test_data" in router_config:
            test_samples = load_data(router_config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

    # Load initial playbook (prefer seed playbook, fallback to initial_playbook_path)
    initial_playbook = None
    if args.seed_playbook_path and os.path.exists(args.seed_playbook_path):
        with open(args.seed_playbook_path, 'r', encoding='utf-8') as f:
            initial_playbook = f.read()
        print(f"Loaded seeded playbook from {args.seed_playbook_path}")
    elif args.initial_playbook_path and os.path.exists(args.initial_playbook_path):
        with open(args.initial_playbook_path, 'r', encoding='utf-8') as f:
            initial_playbook = f.read()
        print(f"Loaded initial playbook from {args.initial_playbook_path}")
    else:
        print("Starting with empty playbook")

    # CRITICAL: Override prompts BEFORE importing ACE.
    # The Generator and Reflector classes bind GENERATOR_PROMPT / REFLECTOR_PROMPT
    # at class definition time via module-level imports. Patching the module globals
    # must happen before ACE is imported, or the old references are captured.
    from .prompts.generator import ROUTING_GENERATOR_PROMPT
    from .prompts.reflector import ROUTING_REFLECTOR_PROMPT
    import ace.prompts.generator as gen_prompts
    import ace.prompts.reflector as ref_prompts
    gen_prompts.GENERATOR_PROMPT = ROUTING_GENERATOR_PROMPT
    ref_prompts.REFLECTOR_PROMPT = ROUTING_REFLECTOR_PROMPT

    # Now import ACE (Generator/Reflector will pick up the patched prompts)
    from ace import ACE, ACEBatch

    # Initialize ACE
    AceCls = ACEBatch if args.batch_size > 1 else ACE
    ace_system = AceCls(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
    )

    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': 'router',
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'api_provider': args.api_provider,
        'batch_size': args.batch_size,
        'curator_batch_size': args.curator_batch_size,
        'augmented_shuffling': args.augmented_shuffling,
    }

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=processor,
        config=config,
    )

    print(f"\nFinal results: {json.dumps({k: v for k, v in results.items() if k != 'training_results'}, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify module loads**

```bash
cd e:/github/ace && uv run python -c "from eval.router.run import main; print('Module loaded OK')"
```

Expected: `Module loaded OK`

- [ ] **Step 3: Commit**

```bash
git add eval/router/run.py
git commit -m "feat: add ACE training script for routing task

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Seed Playbook Builder

**Files:**
- Create: `eval/router/seed_playbook.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5, 8 (exclusion rules, conflict matrix, evidence patterns, decision framework)
- Produces: `build_seed_playbook(components: Dict) -> str` — assembled initial Playbook

- [ ] **Step 1: Write seed playbook builder**

```python
# eval/router/seed_playbook.py
"""Assemble a seeded Playbook from all pre-processing outputs."""

import json
import os


ACE_EMPTY_SECTIONS = """## COMMON MISTAKES

"""


def build_seed_playbook(
    exclusion_rules: str = "",
    decision_framework: str = "",
    evidence_reliability: str = "",
    routing_strategies: str = "",
) -> str:
    """
    Assemble all pre-generated sections into a complete seed playbook.

    Order follows the Generator's decision process:
      1. EXCLUSION THRESHOLDS — quick wins, eliminate impossible agents
      2. DECISION FRAMEWORK — structured decision process
      3. EVIDENCE RELIABILITY — what signals to trust/distrust
      4. ROUTING STRATEGIES — accumulated strategy rules
      5. COMMON MISTAKES — ACE will fill this during training
    """
    sections = []

    if exclusion_rules:
        sections.append(exclusion_rules)
    else:
        sections.append("## EXCLUSION THRESHOLDS\n\n(待训练生成)")

    sections.append("")

    if decision_framework:
        sections.append(decision_framework)
    else:
        sections.append("## DECISION FRAMEWORK\n\n(待训练生成)")

    sections.append("")

    if evidence_reliability:
        sections.append(evidence_reliability)
    else:
        sections.append("## EVIDENCE RELIABILITY\n\n(待训练生成)")

    sections.append("")

    if routing_strategies:
        sections.append(routing_strategies)
    else:
        sections.append("## ROUTING STRATEGIES\n\n(待训练生成)")

    sections.append("")
    sections.append(ACE_EMPTY_SECTIONS)

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
```

- [ ] **Step 2: Commit**

```bash
git add eval/router/seed_playbook.py
git commit -m "feat: add seed playbook builder

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Adversarial Sample Generator (Aux #2)

**Files:**
- Create: `eval/router/augment/__init__.py`
- Create: `eval/router/augment/adversarial.py`

**Interfaces:**
- Consumes: A-class badcase data
- Produces: `generate_adversarial_variants(badcase, api_client, model, n_variants=5) -> List[Dict]`

- [ ] **Step 1: Create directory**

```bash
mkdir -p eval/router/augment
touch eval/router/augment/__init__.py
```

- [ ] **Step 2: Write adversarial generator**

```python
# eval/router/augment/adversarial.py
"""Aux #2: Generate adversarial variants of badcases for data augmentation.

Enabled only when badcase count < 200.
Each variant keeps the same error pattern but changes surface form.
"""

import json
from typing import List, Dict
from openai import OpenAI


VARIANT_PROMPT = """你是一个数据增强专家。请为以下客服路由badcase生成变体。

原始query: {query}
对话历史: {history}
错误路由结果: {wrong_agent}
正确路由结果: {correct_agent}
错误原因: {router_reason}
人工判断理由: {human_note}

请生成 {n} 个变体query和对应的对话历史。要求：
1. 保持相同的错误模式（模型会被类似方式误导）
2. 正确的路由目标（{correct_agent}）保持不变
3. 只改变query的表面表达方式和对话历史的主题
4. 保持消费金融领域的语境
5. 每个变体应该有不同的对话历史主题

输出格式（每行一个JSON）:
{{"query": "变体query", "history": ["user: ...", "assistant: ..."], "correct_agent": "{correct_agent}"}}
"""


def generate_adversarial_variants(
    badcase: Dict,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro",
    n_variants: int = 5
) -> List[Dict]:
    """
    Generate adversarial variants for a single badcase.

    Args:
        badcase: Dict with query, historyDialogue, router_agent, human_annotate_agent,
                 router_reason, human_note
        api_client: OpenAI-compatible client
        model: Model name
        n_variants: Number of variants to generate

    Returns:
        List of variant dicts with query, history, correct_agent
    """
    history_str = "\n".join(badcase.get('historyDialogue', []))
    prompt = VARIANT_PROMPT.format(
        query=badcase.get('query', ''),
        history=history_str,
        wrong_agent=badcase.get('router_agent', ''),
        correct_agent=badcase.get('human_annotate_agent', ''),
        router_reason=badcase.get('router_reason', ''),
        human_note=badcase.get('human_note', ''),
        n=n_variants,
    )

    response = api_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON lines
    variants = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                variants.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return variants[:n_variants]


def validate_variant(
    variant: Dict,
    original_badcase: Dict,
    api_client: OpenAI,
    model: str = "deepseek-v4-pro"
) -> bool:
    """
    Validate that a variant preserves the original error pattern.

    Uses a simple check: does the variant's query still express the same
    routing intent as the original badcase?

    Returns True if variant preserves the error pattern.
    """
    original_query = original_badcase.get('query', '')
    variant_query = variant.get('query', '')
    correct_agent = variant.get('correct_agent', '')

    # Quick heuristic: variant query should contain at least one
    # meaningful word overlap with original beyond stopwords
    orig_words = set(original_query)
    var_words = set(variant_query)
    overlap = orig_words & var_words
    overlap_meaningful = {w for w in overlap if len(w) > 1}

    # If no meaningful overlap, the variant might have changed the intent
    if len(overlap_meaningful) < 2:
        print(f"  ⚠️ Low overlap variant: '{variant_query[:50]}...' — skipping")
        return False

    return True


def augment_badcases(
    a_class_data: List[Dict],
    api_client: OpenAI,
    model: str = "deepseek-v4-pro",
    n_variants_per_case: int = 5,
    output_path: str = None
) -> List[Dict]:
    """
    Generate and validate adversarial variants for all badcases.

    Args:
        a_class_data: List of A-class (badcase) dicts
        api_client: OpenAI-compatible client
        model: Model name
        n_variants_per_case: Variants per badcase
        output_path: Optional path to save augmented data

    Returns:
        List of validated variant dicts
    """
    badcase_count = len(a_class_data)
    if badcase_count >= 200:
        print(f"Badcase count ({badcase_count}) >= 200, skipping adversarial augmentation")
        return []

    print(f"Generating adversarial variants for {badcase_count} badcases...")
    all_variants = []

    for i, badcase in enumerate(a_class_data):
        print(f"  [{i+1}/{badcase_count}] Generating variants...")
        variants = generate_adversarial_variants(
            badcase, api_client, model, n_variants_per_case
        )

        for v in variants:
            if validate_variant(v, badcase, api_client, model):
                # Add metadata from original
                v['human_annotate_agent'] = v.pop('correct_agent')
                v['router_agent'] = badcase.get('router_agent', '')
                v['router_reason'] = ''
                v['human_note'] = f"对抗样本（原始: {badcase.get('human_note', '')[:50]}）"
                v['historyDialogue'] = v.pop('history', [])
                v['current_agent'] = badcase.get('current_agent', '')
                v['session_id'] = f"aug_{badcase.get('session_id', 'unknown')}_{len(all_variants)}"
                v['final_agent'] = badcase.get('final_agent', '')
                v['source'] = 'adversarial_augmentation'
                all_variants.append(v)

        print(f"    Generated {len(variants)} variants, {len([v for v in variants if v.get('query')])} validated")

    print(f"Total validated variants: {len(all_variants)}")

    if output_path and all_variants:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for v in all_variants:
                f.write(json.dumps(v, ensure_ascii=False) + '\n')
        print(f"Saved to {output_path}")

    return all_variants


if __name__ == "__main__":
    import argparse
    from utils import initialize_clients

    parser = argparse.ArgumentParser()
    parser.add_argument('--a_data', required=True, help='A-class JSONL data')
    parser.add_argument('--api_provider', default='openai')
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--n_variants', type=int, default=5)
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
    client, _, _ = initialize_clients(args.api_provider)
    augment_badcases(a_data, client, args.model, args.n_variants, args.output)
```

- [ ] **Step 3: Commit**

```bash
git add eval/router/augment/__init__.py eval/router/augment/adversarial.py
git commit -m "feat: add adversarial sample generator (Aux #2)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Implementation Order

| Phase | Task | Dependencies | Can parallelize? |
|---|---|---|---|
| Foundation | 1. DataProcessor | None | — |
| Foundation | 2. Data Prep | Task 1 | After Task 1 |
| Pre-process | 3. Exclusion Rules (#4) | None | Parallel with 1, 2 |
| Pre-process | 4. Conflict Matrix (#1) | Task 2 | After Task 2 |
| Pre-process | 5. Evidence Assessment (#3) | Task 2 | After Task 2 |
| Prompts | 6. Reflector Prompt (#1) | None | Parallel with all above |
| Prompts | 7. Generator Prompt | None | Parallel with all above |
| Pre-process | 8. Decision Framework (#2) | Tasks 3, 4 | After Tasks 3, 4 |
| Calibrate | 9. Reverse Calibration (#5) | Task 2 | After Task 2 |
| Integration | 10. Training Script | Tasks 1, 6, 7 | After Tasks 1, 6, 7 |
| Integration | 11. Seed Playbook | Tasks 3, 4, 5, 8 | After Task 8 |
| Augment | 12. Adversarial (#2) | Task 2 | After Task 2 |

**Recommended execution order**: 1 → 2, 3, 6, 7 (parallel) → 4, 5, 9, 12 (parallel) → 8 → 11 → 10

---

## End-to-End Pipeline Usage

```bash
# Step 0: Prepare data splits
uv run python -m eval.router.preprocess.data_prep \
    --input task_info/annotated_data.jsonl \
    --output_dir ./eval/router/data

# Step 1: Generate exclusion rules (Core #4)
uv run python -m eval.router.preprocess.exclusion_rules \
    --agent_info task_info/agent_info.json \
    --api_provider openai \
    --model deepseek-v4-pro \
    --output ./eval/router/data/exclusion_rules.txt

# Step 2: Generate conflict matrix (Aux #1)
uv run python -m eval.router.preprocess.conflict_matrix \
    --input ./eval/router/data/train.jsonl \
    --output ./eval/router/data/conflict_matrix.md

# Step 3: Evidence assessment (Core #3)
uv run python -m eval.router.preprocess.evidence_assessment \
    --a_data ./eval/router/data/train.jsonl \
    --b_data ./eval/router/data/train.jsonl \
    --api_provider openai \
    --model deepseek-v4-pro \
    --output ./eval/router/data/evidence_reliability.txt

# Step 4: Adversarial augmentation (Aux #2) — only if badcase < 200
uv run python -m eval.router.augment.adversarial \
    --a_data ./eval/router/data/train.jsonl \
    --api_provider openai \
    --model deepseek-v4-pro \
    --output ./eval/router/data/augmented.jsonl

# Step 5: Generate decision framework (Core #2)
uv run python -m eval.router.preprocess.decision_framework \
    --agent_info task_info/agent_info.json \
    --a_data ./eval/router/data/train.jsonl \
    --exclusion_rules_file ./eval/router/data/exclusion_rules.txt \
    --api_provider openai \
    --model deepseek-v4-pro \
    --output ./eval/router/data/decision_framework.txt

# Step 6: Build seed playbook
uv run python -m eval.router.seed_playbook \
    --exclusion_rules ./eval/router/data/exclusion_rules.txt \
    --decision_framework ./eval/router/data/decision_framework.txt \
    --evidence_reliability ./eval/router/data/evidence_reliability.txt \
    --output ./eval/router/data/seed_playbook.txt

# Step 7: Run ACE offline training
uv run python -m eval.router.run \
    --mode offline \
    --api_provider openai \
    --generator_model deepseek-v4-pro \
    --reflector_model deepseek-v4-pro \
    --curator_model deepseek-v4-pro \
    --num_epochs 3 \
    --eval_steps 50 \
    --seed_playbook_path ./eval/router/data/seed_playbook.txt \
    --agent_info_path task_info/agent_info.json \
    --save_path ./results/router

# Step 8: Evaluate with best playbook
uv run python -m eval.router.run \
    --mode eval_only \
    --api_provider openai \
    --generator_model deepseek-v4-pro \
    --initial_playbook_path ./results/router/ace_run_*/best_playbook.txt \
    --agent_info_path task_info/agent_info.json \
    --save_path ./results/router/eval
```
