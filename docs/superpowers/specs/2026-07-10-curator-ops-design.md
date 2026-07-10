# Curator 四操作（ADD/MODIFY/DELETE/KEEP）设计

> 日期: 2026-07-10
> 状态: 待审批

---

## 1. 背景

ACE 的 `apply_curator_operations` 当前只实现 ADD——Curator 永远追加新规则，Playbook 只涨不跌。训练后 Playbook 充斥着重复、矛盾、低质量规则。

借鉴 Training-Free GRPO 论文的经验管理机制：LLM 对每条经验自主决定 Add/Delete/Modify/Keep，代码只负责执行。

## 2. 与 Approach 1 对比

| 维度 | Approach 1（代码规则控制） | Approach 2（LLM 自主决策） |
|---|---|---|
| **处理逻辑** | 纯代码规则：embedding 相似度 > 阈值 → MERGE；helpful-harmful < 0 → PRUNE；计数器 > N → 升降权 | LLM 全权决策：Curator 看到 reflection + 当前 Playbook → 自己判断该增该改该删该留 |
| **可靠性的锚** | 数学公式和阈值——确定性强，但阈值本身是拍脑袋 | LLM 的语义理解——灵活，但判断质量依赖模型能力 |
| **决策者** | 三个独立机制：计数器（PRUNE）+ embedding（MERGE）+ Curator（UPDATE） | 只有 Curator LLM |
| **何时触发** | 不同时机：Curator 每步跑，MERGE 在 Curator 后，PRUNE 定期扫 | 统一在 Curator 这一步 |
| **冲突处理** | 三个机制可能互踢：MERGE 刚合完、PRUNE 又删了、UPDATE 又改回来 | LLM 自行消解并输出最终 plan，代码顺序执行 |
| **操作语义** | UPDATE/MERGE/DELETE/PRUNE，边界模糊（UPDATE vs MERGE 有什么区别？） | ADD/MODIFY/DELETE 三种代码分支，KEEP 是 prompt 约束（不产生代码分支） |
| **依赖** | sentence-transformers + faiss + embedding 阈值调参 | 无额外依赖 |
| **改动量** | 改 3 个函数 + 新增 1 个函数 + 调阈值 | 改 1 个函数 + 改 prompt |
| **论文依据** | 参考 Text2Mem 12 操作规范 | 照搬 Training-Free GRPO 的经验管理 |

## 3. 处理逻辑

**LLM 全权决策，代码纯执行。**

```
batch reflection → Curator LLM → 自冲突消解 → operations 列表
                                           ↓
                                    apply_curator_operations
                                    （代码层纯执行，不做任何判断）
```

- 多条 reflection 指向同一规则 → Curator 自行合并为一个 operation
- MODIFY 找不到目标 ID → 降级为 ADD
- DELETE 找不到目标 ID → 跳过（可能已被删）
- KEEP → 不产生 operation 条目

**和 Approach 1 的根本区别：代码不参与决策。没有阈值、没有计数器公式、没有 embedding 相似度。一切靠 Curator LLM。**

## 4. 改动范围

### 4.1 playbook_utils.py — `apply_curator_operations`

当前（第 127-161 行）只有一个 `if op_type == 'ADD'`。

新增两个分支：

```python
# MODIFY: 更新已有规则的 content，保留 id 和计数器
elif op_type == 'MODIFY':
    target_id = op.get('target_bullet_id', '')
    new_content = op.get('content', '')
    if target_id and new_content:
        bullets_to_modify[target_id] = new_content

# DELETE: 删除已有规则
elif op_type == 'DELETE':
    target_id = op.get('target_bullet_id', '')
    if target_id:
        bullets_to_delete.add(target_id)
```

重建逻辑调整——逐行遍历时：

```python
for line in new_lines:
    parsed = parse_playbook_line(line)
    if parsed:
        bid = parsed['id']
        if bid in bullets_to_delete:
            continue  # skip deleted
        if bid in bullets_to_modify:
            line = format_playbook_line(bid, parsed['helpful'], parsed['harmful'], bullets_to_modify[bid])
    final_lines.append(line)
```

**MODIFY 安全兜底**：操作完成后检查 `bullets_to_modify` 中未使用的 ID → 打印 warning 并降级为 ADD。

### 4.2 ace/prompts/curator.py — Curator prompt

`CURATOR_PROMPT` 和 `CURATOR_PROMPT_NO_GT` 的 Available Operations 从只有 ADD 扩展为：

```
**Available Operations:**
1. ADD: 追加一条全新规则
    - section, content

2. MODIFY: 更新已有规则的内容（已有规则不准确或不完整时使用）
    - target_bullet_id: 要修改的规则 [id]
    - section, content: 新的 section 和 content
    - 注意: 多条 reflection 指向同一规则时，自行合并为**一个** MODIFY

3. DELETE: 删除低质量规则（规则被证明有害或已被更好的规则替代）
    - target_bullet_id: 要删除的规则 [id]

4. KEEP: 不做任何操作（经验与已有规则等价时不产生 operation）
```

### 4.3 eval/router/prompts/curator.py — 路由专用 Curator prompt

同上，并保留已有的一句话规则约束（"不超过30字"）。

## 5. 执行流程

```
Curator LLM 收到 8 条 reflection + 当前 Playbook
    │
    ▼
逐条 reflection 判断:
  "对话历史词汇关联不等于话题延续"
    → 已有 [str-00001] 说了类似但更精准 → KEEP（不输出）
  "多步联想链不可靠，一步直达更可信"
    → 新洞察，Playbook 里没有 → ADD（输出一条 operation）
  "[err-00005] 提到'优先看上下文'，这是错误策略"
    → 旧规则有害 → DELETE err-00005
  "上下文关联策略的修正版"
    → 已有 [str-00007] 但对策过时 → MODIFY str-00007
    ...
    │
    ▼
输出 operations 列表: [{"type": "ADD", ...}, {"type": "DELETE", ...}, {"type": "MODIFY", ...}]
    │
    ▼
apply_curator_operations 顺序执行:
  1. 收集所有 DELETE → 从 lines 中移除
  2. 收集所有 MODIFY → 替换对应行的 content
  3. 收集所有 ADD → 按 section 插入
  4. MODIFY 未命中 → 降级 ADD（不丢规则）
  5. DELETE 未命中 → 跳过
```

## 6. Reflection 预分组

在 Curator LLM 调用前，将 reflection 按涉及的 playbook 规则分组，同一条规则的所有 feedback 放一起。

**分组依据**：Reflector 输出的 `bullet_tags`（每条 reflection 标注了它涉及哪些已有规则的 `bullet_id`）。

```
原始 reflection 列表:                    分组后:
  R1: bullet_tags=[str-00003] → 修正    ┌─ [str-00003] ─────────┐
  R2: bullet_tags=[] → 全新洞察          │ R1: 内容不够精准       │
  R3: bullet_tags=[str-00003] → 替换    │ R3: 建议完全替换       │
  R4: bullet_tags=[err-00007] → 删除    │ R5: 建议直接删除       │
  R5: bullet_tags=[str-00003] → 删除    └──────────────────────┘
                                        ┌─ [err-00007] ─────────┐
                                        │ R4: 规则有害，删除     │
                                        └──────────────────────┘
                                        ┌─ NEW ─────────────────┐
                                        │ R2: 新洞察，未涉及现有 │
                                        └──────────────────────┘
```

**效果**：每组 Curator 一眼看清"这条规则收到了哪些反馈"，直接决定最终操作。不再需要从 8 条无序 reflection 中自己找关联。

**代码层**：一个简单的 `defaultdict(list)` 预处理，按 `bullet_id` 分组 + 无关联的归入 `"new"` 组。在带入 Curator prompt 前把同组 reflection 拼在一起。

## 8. 关键设计决策

| 决策 | 原因 |
|---|---|
| 代码零判断 | 不设阈值、不查计数器、不调 embedding——LLM 说了算 |
| MODIFY 降级 ADD | 不丢信息，Curator 引用了不存在的 ID 大概率是自己搞错了 |
| DELETE 找不到跳过 | 安全优先，不误删（可能已经被删过了） |
| KEEP 不产生 operation | 减少 noise，Curator 只输出实际操作的条目 |
| 操作顺序: DELETE > MODIFY > ADD | DELETE 先跑，避免 MODIFY 白改；MODIFY 再替换 content；ADD 最后插入 |

## 9. Prompt 关键约束

在 Curator prompt 的 Instructions 中新增：

```
- 如果多条 reflection 指向同一条已有规则，自行合并为**一个** MODIFY 或 DELETE 操作
- MODIFY 必须指定 target_bullet_id，引用当前 Playbook 中已有规则的 [id]
- DELETE 必须指定 target_bullet_id，只在规则被证明有害或有更好替代时使用
- 不确定时优先 KEEP——不要为了"做点什么"而乱改
```
