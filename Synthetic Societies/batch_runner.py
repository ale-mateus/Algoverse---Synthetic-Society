import os
import subprocess

def load_prompts(path="prompts.txt"):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    prompts = [p.strip() for p in content.split("\n\n") if p.strip()]
    return prompts

societies = [
    "dev_society.json",
    "ER_society.json",
    "noRoles.json"
]

providers = ["openai", "openrouter"]
models = ["gpt-4.1-nano"]
seeds = [1, 2, 3]

def run_single(json_path, task, provider, model, seed):
    cmd = [
        "python", "run.py",
        "--json", json_path,
        "--task", task,
        "--provider", provider,
        "--model", model,
        "--seed", str(seed)
    ]
    subprocess.run(cmd)

def main():
    prompts = load_prompts("prompts.txt")
    for json_file in societies:
        json_path = os.path.join("societies", json_file)
        for task in prompts:
            for provider in providers:
                for model in models:
                    for seed in seeds:
                        run_single(json_path, task, provider, model, seed)

if __name__ == "__main__":
    main()
