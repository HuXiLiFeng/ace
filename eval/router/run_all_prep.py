"""Run ALL preprocessing steps in one go. Just run this once:

    python eval/router/run_all_prep.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PYTHON = sys.executable
STEPS = [
    ("Core #4: Exclusion Rules",
     f"{PYTHON} -m eval.router.preprocess.exclusion_rules "
     "--agent_info task_info/agent_info.json --api_provider openai "
     "--model deepseek-v4-pro --output eval/router/data/exclusion_rules.txt"),

    ("Aux #1: Conflict Matrix",
     f"{PYTHON} -m eval.router.preprocess.conflict_matrix "
     "--input eval/router/data/train.jsonl "
     "--output eval/router/data/conflict_matrix.md"),

    ("Core #3: Evidence Assessment",
     f"{PYTHON} -m eval.router.preprocess.evidence_assessment "
     "--a_data eval/router/data/train.jsonl --b_data eval/router/data/train.jsonl "
     "--api_provider openai --model deepseek-v4-pro "
     "--output eval/router/data/evidence_reliability.txt"),

    ("Core #2: Decision Framework",
     f"{PYTHON} -m eval.router.preprocess.decision_framework "
     "--agent_info task_info/agent_info.json "
     "--a_data eval/router/data/train.jsonl "
     "--exclusion_rules_file eval/router/data/exclusion_rules.txt "
     "--api_provider openai --model deepseek-v4-pro "
     "--output eval/router/data/decision_framework.txt"),

    ("Seed Playbook",
     f"{PYTHON} -m eval.router.seed_playbook "
     "--exclusion_rules eval/router/data/exclusion_rules.txt "
     "--decision_framework eval/router/data/decision_framework.txt "
     "--evidence_reliability eval/router/data/evidence_reliability.txt "
     "--output eval/router/data/seed_playbook.txt"),
]

print(f"ACE Router — Full Preprocessing Pipeline")
print(f"Working dir: {os.getcwd()}")
print(f"Python: {PYTHON}")
print(f"Already done: mock data generation, data split, exclusion rules, conflict matrix")
print()

for name, cmd in STEPS:
    print(f"{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ❌ FAILED (exit {result.returncode})")
        sys.exit(1)
    print(f"  ✅ DONE")

print(f"\n{'='*50}")
print(f"  ALL PREPROCESSING COMPLETE")
print(f"  Seed playbook: eval/router/data/seed_playbook.txt")
print(f"{'='*50}")
