"""Aux #2: Generate adversarial variants of badcases for data augmentation.

Enabled only when badcase count < 200.
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
    """Generate adversarial variants for a single badcase."""
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
    variants = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                variants.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return variants[:n_variants]


def validate_variant(variant: Dict, original_badcase: Dict) -> bool:
    """Validate variant preserves original error pattern."""
    original_query = original_badcase.get('query', '')
    variant_query = variant.get('query', '')
    orig_words = set(original_query)
    var_words = set(variant_query)
    overlap = orig_words & var_words
    overlap_meaningful = {w for w in overlap if len(w) > 1}
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
    """Generate and validate adversarial variants for all badcases."""
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
            if validate_variant(v, badcase):
                # Copy correct_agent as human_annotate_agent before removal
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

        print(f"    {len(all_variants)} validated so far")

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
    import sys
    sys.path.insert(0, '.')
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
