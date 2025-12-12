import os
import subprocess
import re

def load_prompts(path="prompts.txt"):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return [p.strip() for p in content.split("\n\n") if p.strip()]

def make_prompt_label(prompt):
    first_line = prompt.strip().split("\n")[0]
    words = re.findall(r"[A-Za-z0-9]+", first_line.lower())
    if not words:
        return "prompt"
    label = "_".join(words[:8])
    if not label[0].isalpha():
        label = "p_" + label
    return label[:60]

json_dir = "/Users/prishapriyadashini/Downloads/Algoverse---Synthetic-Society/Synthetic Societies/jsonFiles"

societies = ["dev_society.json", "ER_society.json", "noRoles.json"]

jobs = [
    ("openai", "gpt-4.1-nano"),
    #("openrouter", "qwen/qwen-2.5-72b-instruct")
]

seeds = [1, 2, 3]

def run_single(json_path, task, provider, model, seed, label):
    print(f"\nRunning society={json_path}, provider={provider}, model={model}, seed={seed}")
    print(f"Prompt label: {label}\n")
    cmd = [
        "python3", "run.py",
        "--json", json_path,
        "--task", task,
        "--provider", provider,
        "--model", model,
        "--seed", str(seed),
        "--label", label
    ]
    subprocess.run(cmd)

def main():
    prompts = load_prompts("prompts.txt")
    for json_file in societies:
        json_path = os.path.join(json_dir, json_file)
        for task in prompts:
            label = make_prompt_label(task)
            for provider, model in jobs:
                for seed in seeds:
                    run_single(json_path, task, provider, model, seed, label)

if __name__ == "__main__":
    main()
