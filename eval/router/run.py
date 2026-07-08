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
                        help="Path to pre-generated seeded playbook")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--curator_batch_size", type=int, default=None)
    parser.add_argument("--augmented_shuffling",
                        action=argparse.BooleanOptionalAction, default=True)
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

    with open("./eval/router/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    router_config = task_config["router"]
    processor = DataProcessor(agent_info_path=args.agent_info_path)

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

    # Load initial playbook
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
    # at class definition time via module-level imports.
    from .prompts.generator import ROUTING_GENERATOR_PROMPT
    from .prompts.reflector import ROUTING_REFLECTOR_PROMPT
    import ace.prompts.generator as gen_prompts
    import ace.prompts.reflector as ref_prompts
    gen_prompts.GENERATOR_PROMPT = ROUTING_GENERATOR_PROMPT
    ref_prompts.REFLECTOR_PROMPT = ROUTING_REFLECTOR_PROMPT

    from ace import ACE, ACEBatch

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

    summary = {k: v for k, v in results.items() if k != 'training_results'}
    print(f"\nFinal results: {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
