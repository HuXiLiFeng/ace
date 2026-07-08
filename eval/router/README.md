# ACE 智能路由 — 业务经验提取系统

## 一句话概述

从标注数据中自动提取消费金融路由的业务经验，生成持续进化的 Playbook（策略手册），注入线上路由 prompt，提升 qwen3 模型的判断准确率。

## 核心思路

模型做错路由不是因为"知识不够"，而是因为**判断策略不对**——人被对话历史带偏还是按意图匹配，模型选错了策略。

系统从两类标注数据中学习：
- **A 类（badcase）**：人工标注了正确 agent 和判断理由 → 学"人类怎么判断"
- **B 类（准正确）**：人工默许模型判对了 → 作回归验证集

5 个核心机制：
| # | 机制 | 做什么 |
|---|---|---|
| 1 | 推理链对比 | 对比 human_note 和 router_reason，学判断策略差异 |
| 2 | 微决策流程 | 为每个高频混淆 Agent 对生成专用判断树 |
| 3 | 证据评估 | 分析 router_reason 中"可靠信号 vs 思维陷阱" |
| 4 | 排除规则 | 从 agent_info【注意事项】提取硬约束，缩小候选 |
| 5 | 反向校准 | 新规则上线前验证不破坏已有正确 case |

产出是一个 **Playbook（文本文件）**，包含决策框架、证据可靠性指南、策略规则、排除阈值、常见错误，直接注入路由 prompt。

## 文件结构

```
eval/router/
├── data_processor.py          # 数据格式转换 (raw → ACE standard)
├── run.py                     # ACE 训练入口
├── seed_playbook.py           # 组装初始 Playbook
├── prompts/
│   ├── generator.py           # Generator prompt (Playbook 驱动的路由判断)
│   └── reflector.py           # Reflector prompt (推理链对比)
├── preprocess/
│   ├── data_prep.py           # 数据切分 (train/val/test)
│   ├── exclusion_rules.py     # Core #4: 排除规则提取
│   ├── conflict_matrix.py     # Aux #1: Agent 混淆矩阵
│   ├── evidence_assessment.py # Core #3: 证据可靠性分析
│   └── decision_framework.py  # Core #2: 微决策流程生成
├── calibrate/
│   └── reverse_calibrate.py   # Core #5: 反向校准
├── augment/
│   └── adversarial.py         # Aux #2: 对抗样本增强
└── data/
    └── sample_config.json     # 数据路径配置
```

## 运行流程

### 前置条件

```bash
# 1. 确保 ACE 环境就绪
uv sync

# 2. 准备标注数据 (JSONL 格式)
# 每行一个 JSON，字段: session_id, query, historyDialogue,
#   current_agent, router_agent, router_reason,
#   human_note, human_annotate_agent, final_agent
# human_* 为空的 = B类(准正确)，有值的 = A类(badcase)
```

### Step 1: 数据切分

```bash
uv run python -m eval.router.preprocess.data_prep \
    --input path/to/annotated_data.jsonl \
    --output_dir ./eval/router/data
```

### Step 2: 生成种子 Playbook（4个预处理步骤）

```bash
# 2a. 排除规则 (Core #4)
uv run python -m eval.router.preprocess.exclusion_rules \
    --agent_info task_info/agent_info.json \
    --api_provider openai --model deepseek-v4-pro \
    --output ./eval/router/data/exclusion_rules.txt

# 2b. 冲突矩阵 (Aux #1)
uv run python -m eval.router.preprocess.conflict_matrix \
    --input ./eval/router/data/train.jsonl \
    --output ./eval/router/data/conflict_matrix.md

# 2c. 证据评估 (Core #3)
uv run python -m eval.router.preprocess.evidence_assessment \
    --a_data ./eval/router/data/train.jsonl \
    --b_data ./eval/router/data/train.jsonl \
    --api_provider openai --model deepseek-v4-pro \
    --output ./eval/router/data/evidence_reliability.txt

# 2d. 决策框架 (Core #2)
uv run python -m eval.router.preprocess.decision_framework \
    --agent_info task_info/agent_info.json \
    --a_data ./eval/router/data/train.jsonl \
    --exclusion_rules_file ./eval/router/data/exclusion_rules.txt \
    --api_provider openai --model deepseek-v4-pro \
    --output ./eval/router/data/decision_framework.txt
```

### Step 3: 组装种子 Playbook

```bash
uv run python -m eval.router.seed_playbook \
    --exclusion_rules ./eval/router/data/exclusion_rules.txt \
    --decision_framework ./eval/router/data/decision_framework.txt \
    --evidence_reliability ./eval/router/data/evidence_reliability.txt \
    --output ./eval/router/data/seed_playbook.txt
```

### Step 4: ACE 离线训练

```bash
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
```

### Step 5: 评估

```bash
uv run python -m eval.router.run \
    --mode eval_only \
    --api_provider openai \
    --generator_model deepseek-v4-pro \
    --initial_playbook_path ./results/router/<run>/best_playbook.txt \
    --agent_info_path task_info/agent_info.json \
    --save_path ./results/router/eval
```

### （可选）对抗增强

```bash
# 仅当 badcase < 200 时启用，生成训练数据变体
uv run python -m eval.router.augment.adversarial \
    --a_data ./eval/router/data/train.jsonl \
    --api_provider openai --model deepseek-v4-pro \
    --output ./eval/router/data/augmented.jsonl
```

## Playbook 结构

```
## DECISION FRAMEWORK          ← 微决策流程 (Core #2)
## EVIDENCE RELIABILITY        ← 可靠/不可靠证据模式 (Core #3)
## ROUTING STRATEGIES          ← 策略规则 (Core #1, ACE持续更新)
## EXCLUSION THRESHOLDS        ← 软排除约束 (Core #4)
## COMMON MISTAKES             ← ACE 自动积累
```

## 线上部署

将 `best_playbook.txt` 内容注入原路由 prompt 最前面：

```
{playbook_content}

# 可用Agent列表
{agent_info}
...
```

线上 qwen3-20b-a3b 按 Playbook 中的决策框架逐步判断。定期用新标注数据重复 Step 1-4 更新 Playbook。

## 关键阈值

| 条件 | 行为 |
|---|---|
| badcase < 100 | Core #1 不自动生成策略规则（只出分析报告） |
| badcase < 200 | 启用对抗样本增强 (Aux #2) |
| 新 badcase ≥ 50 | 重新生成决策框架 (Core #2) |
| B类样本 ≥ 50 | 完整反向校准 (Core #5) |
| B类样本 10-49 | 仅规则冲突检测 |
| B类样本 < 10 | 规则标记低置信度，不自动上线 |
