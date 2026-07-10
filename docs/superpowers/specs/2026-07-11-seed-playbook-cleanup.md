# 种子 Playbook 清理：去掉无 ground truth 的预生成

> 日期: 2026-07-11
> 状态: 待审批

---

## 1. 问题

当前种子 Playbook 有三个预生成模块：

| 模块 | 输入 | 有无 ground truth | LLM 角色 |
|---|---|---|---|
| exclusion_rules | agent_info 原文 | ✅ 有：注意事项原文明确写了排除条款 | 指代消解 + 格式整理 |
| evidence_assessment | router_reason | ❌ 无：LLM 不懂业务，无法判断推理靠谱性 | 凭空判断→ 胡说 |
| decision_framework | human_note | ❌ 无：3条短理由"展开"成5步决策树→ 多出来的步数全是脑补 | 创造→ 胡说 |

`evidence_assessment` 和 `decision_framework` 的产物注入种子 Playbook → ACE 带着错误经验训练 → 越训越偏。

## 2. 方案

**删除两个模块**，种子 Playbook 只保留排除规则。其余全交给 ACE 自己学。

```
之前（5段）:                        之后（3段）:
## DECISION FRAMEWORK               ## EXCLUSION THRESHOLDS
## EVIDENCE RELIABILITY             ## ROUTING STRATEGIES (空)
## ROUTING STRATEGIES               ## COMMON MISTAKES (空)
## EXCLUSION THRESHOLDS
## COMMON MISTAKES
```

**原则：种子 Playbook 里只放有 ground truth 的内容。LLM 没有业务知识就不该"创造"业务知识。**

## 3. 改动范围

| 文件 | 操作 |
|---|---|
| `eval/router/preprocess/evidence_assessment.py` | 删除 |
| `eval/router/preprocess/decision_framework.py` | 删除 |
| `eval/router/seed_playbook.py` | 精简为只接受 `--exclusion_rules` 一个输入 |
| `eval/router/run_all_prep.py` | 去掉 evidence 和 decision_framework 步骤 |
| `eval/router/prompts/generator.py` | 去掉 EVIDENCE RELIABILITY 和 DECISION FRAMEWORK 引用 |
| `eval/router/preprocess/exclusion_rules.py` | prompt 从"分析归纳"改为"指代消解+格式整理"，不创造不推断 |
| `eval/router/README.md` | 同步更新文件结构、命令、Playbook 结构 |

## 4. 影响

- 种子 Playbook 更短、更可信
- ACE 训练从更干净的起点开始，靠自己从 badcase 中学路由策略
- 预处理 pipeline 从 4 步减到 2 步（排除规则 + 冲突矩阵）
