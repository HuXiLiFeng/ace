"""Run ALL preprocessing steps in one go. Just run this once:

    python eval/router/run_all_prep.py
"""

import subprocess

STEPS = [
    ("Core #4: Exclusion Rules",
     "python -m eval.router.preprocess.exclusion_rules "
     "--agent_info task_info/agent_info.json --api_provider openai "
     "--model deepseek-v4-pro --output eval/router/data/exclusion_rules.txt"),

    ("Aux #1: Conflict Matrix",
     "python -m eval.router.preprocess.conflict_matrix "
     "--input eval/router/data/train.jsonl "
     "--output eval/router/data/conflict_matrix.md"),

    ("Seed Playbook",
     "python -m eval.router.seed_playbook "
     "--exclusion_rules eval/router/data/exclusion_rules.txt "
     "--output eval/router/data/seed_playbook.txt"),
]

print("ACE Router — Full Preprocessing Pipeline")

for name, cmd in STEPS:
    print(f"{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ❌ FAILED (exit {result.returncode})")
        exit(1)
    print(f"  ✅ DONE")

print(f"\n{'='*50}")
print(f"  ALL PREPROCESSING COMPLETE")
print(f"  Seed playbook: eval/router/data/seed_playbook.txt")
print(f"{'='*50}")
