import os
import json
import random
import asyncio
from mcq_eval import run_mcq_eval

SOCIETY_DIR = "jsonFiles"
TRAIN_JSON = "datasets/MEDMCQA/train.json"
RESULTS_DIR = "results"

MODEL = "gpt-4.1-nano"
SEEDS = [1, 2, 3]
SAMPLE_SIZE = 100


def load_sampled_train():
    with open(TRAIN_JSON, "r") as f:
        lines = f.readlines()

    lines = [json.loads(l) for l in lines]
    sample = random.sample(lines, SAMPLE_SIZE)

    print(f"\nLoaded {SAMPLE_SIZE} sampled training questions.\n")
    return sample


def save_accuracy(split, society, model, seed, acc):
    out_dir = os.path.join(RESULTS_DIR, split, society, model, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "accuracy.txt"), "w") as f:
        f.write(str(acc))


def main():
    examples = load_sampled_train()
    societies = sorted([f for f in os.listdir(SOCIETY_DIR) if f.endswith(".json")])

    all_results = {}

    for soc_file in societies:
        society_name = soc_file.replace(".json", "")
        json_path = os.path.join(SOCIETY_DIR, soc_file)

        print(f"\n--- Society: {society_name} ---")

        soc_results = []

        for seed in SEEDS:
            print(f"Running model={MODEL}, seed={seed}...")
            acc = asyncio.run(
                run_mcq_eval(json_path, MODEL, seed, examples, society_name)
            )
            print(f"Accuracy: {acc:.4f}")

            save_accuracy("train_sample", society_name, MODEL, seed, acc)
            soc_results.append(acc)

        all_results[society_name] = sum(soc_results) / len(soc_results)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "accuracy_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved accuracy summary to {summary_path}")


if __name__ == "__main__":
    main()
