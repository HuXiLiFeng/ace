# Curator 四操作（ADD/MODIFY/DELETE/KEEP）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ACE Curator 中加入 ADD/MODIFY/DELETE/KEEP 四操作，LLM 全权决策每条经验的操作类型，代码层纯执行。

**Architecture:** 改 `playbook_utils.py` 的 `apply_curator_operations` 函数（加 MODIFY/DELETE 分支 + 重建逻辑） + 改两个 Curator prompt（`ace/prompts/curator.py` 和 `eval/router/prompts/curator.py`，加四种操作说明 + reflection 分组指令）。

**Tech Stack:** Python 3, ACE framework

## Global Constraints

- 不改 `ace/core/` 目录
- 代码不参与决策，不设阈值、不查计数器、不调 embedding
- DELETE > MODIFY > ADD 执行顺序（DELETE 先于 MODIFY 先于 ADD）
- MODIFY 的 `target_bullet_id` 不存在时降级为 ADD
- DELETE 的 `target_bullet_id` 不存在时跳过（可能已被删）
- KEEP 不产生 operation 条目

---

### Task 1: 改 `apply_curator_operations` 加 MODIFY/DELETE 分支

**Files:**
- Modify: `playbook_utils.py:96-216`

**Interfaces:**
- Consumes: 现有 ADD 逻辑 + `parse_playbook_line()`, `format_playbook_line()`, `get_section_slug()`
- Produces: `apply_curator_operations(playbook_text, operations, next_id) -> (str, int)` 返回更新后的 playbook 和新 next_id

- [ ] **Step 1: 替换整个函数**

`playbook_utils.py` 第 96-216 行全部替换为：

```python
def apply_curator_operations(playbook_text, operations, next_id):
    """
    Apply curator operations to playbook.

    Supported operations: ADD, MODIFY, DELETE
    KEEP is a no-op (no operation entry produced by Curator).
    Execution order: DELETE first, then MODIFY, then ADD.
    """
    lines = playbook_text.strip().split('\n')

    # Build section map
    sections = {}
    current_section = "general"
    section_line_map = {}

    for i, line in enumerate(lines):
        if line.strip().startswith('##'):
            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(' ', '_').replace('&', 'and')
            section_line_map[current_section] = i
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections[current_section].append((i, line))

    # ---- Pass 1: collect operations by type ----
    bullets_to_add = []          # [(section, new_line_str)]
    bullets_to_modify = {}       # {bullet_id: new_content}
    bullets_to_delete = set()    # {bullet_id}

    for op in operations:
        op_type = op.get('type', '').upper()

        if op_type == 'DELETE':
            target_id = op.get('target_bullet_id', '')
            if target_id:
                bullets_to_delete.add(target_id)
                print(f"  DELETE target: {target_id}")

        elif op_type == 'MODIFY':
            target_id = op.get('target_bullet_id', '')
            new_content = op.get('content', '')
            if target_id and new_content:
                # If also deleted, DELETE wins
                if target_id not in bullets_to_delete:
                    bullets_to_modify[target_id] = new_content
                    print(f"  MODIFY target: {target_id}")

        elif op_type == 'ADD':
            section_raw = op.get('section', 'general')
            section = section_raw.lower().replace(' ', '_').replace('&', 'and')
            if section not in sections and section != 'general':
                print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
                section = 'others'

            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1

            content = op.get('content', '')
            new_line = format_playbook_line(new_id, 0, 0, content)
            bullets_to_add.append((section, new_line))
            print(f"  ADD {new_id} to section {section}")

    # ---- Pass 2: rebuild playbook line by line ----
    # First pass through original lines to handle DELETE and MODIFY
    new_lines = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            bid = parsed['id']
            if bid in bullets_to_delete:
                continue  # DELETE: skip this line
            if bid in bullets_to_modify:
                # MODIFY: replace content, keep id and counters
                new_line = format_playbook_line(
                    bid, parsed['helpful'], parsed['harmful'],
                    bullets_to_modify[bid]
                )
                del bullets_to_modify[bid]  # mark as consumed
                new_lines.append(new_line)
                continue
        new_lines.append(line)

    # Any MODIFY target not found → downgrade to ADD
    for target_id, content in bullets_to_modify.items():
        print(f"Warning: MODIFY target '{target_id}' not found, downgrading to ADD")
        slug = "misc"
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        new_line = format_playbook_line(new_id, 0, 0, content)
        bullets_to_add.append(('others', new_line))

    # ---- Pass 3: insert ADD bullets into appropriate sections ----
    final_lines = []
    current_section = None

    for line in new_lines:
        if line.strip().startswith('##'):
            # Before moving to new section, add pending bullets for current section
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(' ', '_').replace('&', 'and')
        final_lines.append(line)

    # Add remaining bullets
    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

    # Unmatched bullets → OTHERS
    if bullets_to_add:
        others_bullets = [b for _, b in bullets_to_add]
        others_idx = -1
        for i, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_idx = i
                break
        if others_idx >= 0:
            for i, bullet in enumerate(others_bullets):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            final_lines.extend(others_bullets)

    return '\n'.join(final_lines), next_id
```

- [ ] **Step 2: 冒烟测试 — 在 Python 中验证函数签名**

创建 `_test_ops.py`:

```python
from playbook_utils import apply_curator_operations

playbook = """## COMMON MISTAKES

[err-00001] helpful=2 harmful=0 :: 旧规则内容
"""

# Test ADD
new_pb, nid = apply_curator_operations(
    playbook,
    [{"type": "ADD", "section": "common_mistakes", "content": "新规则一句话"}],
    2
)
assert "新规则一句话" in new_pb
assert "[err-00001]" in new_pb  # 旧规则还在

# Test MODIFY
new_pb2, nid2 = apply_curator_operations(
    new_pb,
    [{"type": "MODIFY", "target_bullet_id": "err-00001", "content": "改过的规则"}],
    nid
)
assert "改过的规则" in new_pb2
assert "旧规则内容" not in new_pb2
assert "[err-00001]" in new_pb2  # ID 不变

# Test DELETE
new_pb3, nid3 = apply_curator_operations(
    new_pb2,
    [{"type": "DELETE", "target_bullet_id": "err-00001"}],
    nid2
)
assert "[err-00001]" not in new_pb3

# Test DELETE > MODIFY (DELETE wins when both target same rule)
new_pb4, _ = apply_curator_operations(
    playbook,
    [
        {"type": "MODIFY", "target_bullet_id": "err-00001", "content": "改的"},
        {"type": "DELETE", "target_bullet_id": "err-00001"},
    ],
    2
)
assert "[err-00001]" not in new_pb4  # DELETE won

# Test MODIFY downgrade to ADD (target not found)
new_pb5, nid5 = apply_curator_operations(
    playbook,
    [{"type": "MODIFY", "target_bullet_id": "not-exist", "content": "找不到目标的规则"}],
    2
)
assert "找不到目标的规则" in new_pb5

print("All assertions passed!")
```

- [ ] **Step 3: Commit**

```bash
git add playbook_utils.py _test_ops.py
git commit -m "feat: ADD/MODIFY/DELETE operations in apply_curator_operations

Execution order: DELETE > MODIFY > ADD.
MODIFY downgrades to ADD when target_bullet_id not found.
DELETE silently skips when target not found.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 改 ACE 通用 Curator prompt

**Files:**
- Modify: `ace/prompts/curator.py`

**Interfaces:**
- Consumes: 现有 `CURATOR_PROMPT` 和 `CURATOR_PROMPT_NO_GT`
- Produces: 更新后的 prompt，LLM 输出 ADD/MODIFY/DELETE/KEEP 操作

- [ ] **Step 1: 替换 Available Operations 段落**

在 `ace/prompts/curator.py` 的两个 prompt（`CURATOR_PROMPT` 和 `CURATOR_PROMPT_NO_GT`）中，将 `**Available Operations:**` 段落替换为：

```
**Available Operations:**
1. ADD: Append a brand-new rule that does not overlap with any existing bullet.
    - section: the target section name
    - content: the new rule as ONE concise sentence

2. MODIFY: Update an existing rule that is inaccurate or incomplete.
    Use when the new insight refines an existing rule rather than replacing it entirely.
    - target_bullet_id: the [id] of the existing bullet to modify (e.g. "str-00042")
    - section: the target section name
    - content: the revised rule content (ONE concise sentence)
    IMPORTANT: If multiple reflections reference the same target_bullet_id,
    merge them into a SINGLE MODIFY operation.

3. DELETE: Remove a low-quality rule that has been proven harmful or
    superseded by a better rule. Only use when the reflection provides
    clear evidence that the existing rule causes errors.
    - target_bullet_id: the [id] of the bullet to delete

4. KEEP: Take no action. Use when the new insight is already covered by
    an existing rule. Produces NO operation entry.
```

并在 Instructions 中加入：

```
- If multiple reflections point to the same existing rule, merge all feedback
  into a SINGLE MODIFY or DELETE operation — never output multiple operations
  for the same target_bullet_id.
- When in doubt between MODIFY and DELETE, prefer MODIFY (refine rather than discard).
- When in doubt between ADD and KEEP, prefer KEEP (avoid redundancy).
```

- [ ] **Step 2: 更新 JSON 示例**

把两个 prompt 的 `**RESPONSE FORMAT**` JSON 示例扩展：

```
**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "common_mistakes_to_avoid",
      "content": "[One concise rule]"
    }},
    {{
      "type": "MODIFY",
      "target_bullet_id": "str-00003",
      "section": "routing_strategies",
      "content": "[Revised concise rule]"
    }},
    {{
      "type": "DELETE",
      "target_bullet_id": "err-00012"
    }}
  ]
}}
```

- [ ] **Step 3: Commit**

```bash
git add ace/prompts/curator.py
git commit -m "feat: ADD/MODIFY/DELETE/KEEP operations in Curator prompt

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 改路由专用 Curator prompt

**Files:**
- Modify: `eval/router/prompts/curator.py`

**Interfaces:**
- Consumes: Task 2 的 prompt 改动
- Produces: 路由专用的 Curator prompt，保留一句话规则约束

- [ ] **Step 1: 同步 Available Operations + Instructions**

对 `eval/router/prompts/curator.py` 中 `ROUTING_CURATOR_PROMPT` 和 `ROUTING_CURATOR_PROMPT_NO_GT` 做同样的改动：

- 替换 `**Available Operations:**` 段落（同 Task 2 Step 1）
- 加入 multi-reflection 合并指令
- 更新 JSON 示例（同 Task 2 Step 2）
- 保留已有的一句话规则约束（"不超过 30 字"）

- [ ] **Step 2: Commit**

```bash
git add eval/router/prompts/curator.py
git commit -m "feat: ADD/MODIFY/DELETE/KEEP in routing Curator prompt

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Reflection 预分组逻辑

**Files:**
- Modify: `ace/ace_batch.py` — 在 Phase 2 给 Curator 传分组后的 reflection

**接口:**
- Consumes: Phase 1 产出的 `sample_results`（每条含 `bullet_tags` + `reflection_content`）
- Produces: 给 Curator 传入按 `bullet_id` 分组的 formatted reflection text

**注意:** 这条只影响 batch 模式（`batch_size > 1`）。单条模式（`ACE` 基类）每步只一个 reflection，无需分组。

- [ ] **Step 1: 在 ace_batch.py 加分组函数**

在 `ace_batch.py` 中新增：

```python
def _group_reflections(sample_results):
    """
    Group reflections by the bullet_ids they reference.
    
    Each sample_result has:
      - reflection_content: the Reflector's analysis
      - all_bullet_tags: list of {"id": "str-00003", "tag": "harmful"}, ...

    Returns formatted text with reflections grouped by bullet_id.
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    new_reflections = []
    
    for res in sample_results:
        reflection = res.get('reflection_content', '')
        tags = res.get('all_bullet_tags', [])
        if not reflection or reflection == '(empty)':
            continue
        
        ref_ids = {t.get('id', '') for t in tags}
        if ref_ids:
            for rid in ref_ids:
                groups[rid].append(reflection)
        else:
            new_reflections.append(reflection)
    
    # Format output
    parts = []
    for bid, reflections in groups.items():
        parts.append(f"--- 针对现有规则 [{bid}] 的反馈 ---")
        for r in reflections:
            parts.append(f"  - {r[:500]}")
        parts.append("")
    
    if new_reflections:
        parts.append("--- 新洞察（未涉及现有规则） ---")
        for r in new_reflections:
            parts.append(f"  - {r[:500]}")
        parts.append("")
    
    return "\n".join(parts)
```

- [ ] **Step 2: 集成到 `_train_batch`**

在 `ace_batch.py` 的 Phase 2 前调用分组函数，将分组后的 reflection 传入 Curator。找到 `_train_batch` 中调用 Curator 的位置，将 `all_reflections` 替换为分组后的版本。

- [ ] **Step 3: Commit**

```bash
git add ace/ace_batch.py
git commit -m "feat: group reflections by bullet_id before Curator call

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 实施顺序

```
Task 1 (apply_curator_operations)  ← 独立
Task 2 (ACE Curator prompt)       ← 独立，可在 1 之后
Task 3 (路由 Curator prompt)       ← 独立，与 2 并行
Task 4 (reflection 分组)          ← 依赖 1，增强 Curator 效果
```

推荐顺序：1 → 2,3（并行）→ 4
