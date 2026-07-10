# 种子 Playbook 清理 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 删除 evidence_assessment.py 和 decision_framework.py，种子 Playbook 只保留有 ground truth 的排除规则。

**Architecture:** 删 2 个文件，改 4 个文件的引用，种子 Playbook 从 5 段变 3 段。

**Tech Stack:** Python 3, ACE framework

## Global Constraints

- 种子 Playbook 只放有 ground truth 的内容
- LLM 没有业务知识就不该创造业务知识

---

### Task 1: 删除两个预处理模块

**Files:**
- Delete: `eval/router/preprocess/evidence_assessment.py`
- Delete: `eval/router/preprocess/decision_framework.py`

- [ ] **Step 1: 删除文件**

```bash
rm eval/router/preprocess/evidence_assessment.py
rm eval/router/preprocess/decision_framework.py
```

- [ ] **Step 2: 验证删除**

```bash
ls eval/router/preprocess/
```

Expected: 只剩 `__init__.py`, `data_prep.py`, `exclusion_rules.py`, `conflict_matrix.py`

- [ ] **Step 3: Commit**

```bash
git add -u eval/router/preprocess/
git commit -m "refactor: remove unreliable pre-generated content"
```

---

### Task 2: 精简 seed_playbook.py

**Files:**
- Modify: `eval/router/seed_playbook.py`

- [ ] **Step 1: 替换为只接受 exclusion_rules**

```python
"""Build a minimal seed Playbook with only exclusion rules pre-generated."""

import os


def build_seed_playbook(exclusion_rules: str = "") -> str:
    """Seed playbook: only EXCLUSION THRESHOLDS is pre-generated."""
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
```

- [ ] **Step 2: Commit**

```bash
git add eval/router/seed_playbook.py
git commit -m "refactor: simplify seed_playbook to only exclusion_rules"
```

---

### Task 3: 同步 run_all_prep.py

**Files:**
- Modify: `eval/router/run_all_prep.py`

- [ ] **Step 1: 去掉 evidence_assessment 和 decision_framework 步骤**

删除两个 STEPS 条目，seed playbook 调用去掉 `--decision_framework` 和 `--evidence_reliability`。

- [ ] **Step 2: Commit**

```bash
git add eval/router/run_all_prep.py
git commit -m "refactor: remove evidence & decision_framework from preprocess pipeline"
```

---

### Task 4: 同步 generator.py 和 README.md

**Files:**
- Modify: `eval/router/prompts/generator.py`
- Modify: `eval/router/README.md`

- [ ] **Step 1: generator.py 去掉 EVIDENCE RELIABILITY 和 DECISION FRAMEWORK 引用**

决策流程从 5 步变 3 步：
1. EXCLUSION THRESHOLDS
2. ROUTING STRATEGIES
3. COMMON MISTAKES

- [ ] **Step 2: README 更新文件结构、命令、Playbook 结构**

- [ ] **Step 3: Commit**

```bash
git add eval/router/prompts/generator.py eval/router/README.md
git commit -m "docs: remove evidence & decision_framework references"
```

---

## 实施顺序

Task 1 → 2,3,4 可并行

推荐：1 → 2 → 3 → 4（串行也很快，4 个 task 都小）
