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
    """Main entry point: load raw data, split, save. Returns dict with paths."""
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

    for split_name, split_batch in [('train', train), ('val', val), ('test', test)]:
        a_count = sum(1 for d in split_batch if (d.get('human_annotate_agent') or '').strip())
        b_count = len(split_batch) - a_count
        print(f"  {split_name}: {len(split_batch)} total ({a_count} A-class, {b_count} B-class)")

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
