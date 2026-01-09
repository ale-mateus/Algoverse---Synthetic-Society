import os
import json
import random
import asyncio

from mcq_eval import run_mcq_eval


SOCIETY_DIR = "jsonMCQ"
TRAIN_JSON = "datasets/MEDMCQA/train.json"
RESULTS_DIR = "results"

MODEL = "gpt-4.1-nano"
SEEDS = [1]
SAMPLE_SIZE = 100
SPLIT_NAME = "train_sample"


def load_sampled_train(seed):
    """
    Load and subsample MEDMCQA training data for quick experimentation.
    """
    random.seed(seed)
    with open(TRAIN_JSON, "r") as f:
        lines = [json.loads(line) for line in f]

    return random.sample(lines, SAMPLE_SIZE)


def save_accuracy(split, society, model, seed, acc):
    """
    Save accuracy using a directory structure
    aligned with experimental variables.
    """
    out_dir = os.path.join(RESULTS_DIR, split, society, model, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "accuracy.txt"), "w") as f:
        f.write(str(acc))


def main():
    """
    Batch evaluation across all society JSONs and seeds.
    Produces both per-society results and an aggregate summary.
    """
    societies = sorted(
        f for f in os.listdir(SOCIETY_DIR) if f.endswith(".json")
    )

    all_results = {}

    for soc_file in societies:
        society_name = soc_file.replace(".json", "")
        json_path = os.path.join(SOCIETY_DIR, soc_file)

        print(f"\n=== Society: {society_name} ===\n")

        soc_results = []

        for seed in SEEDS:
            print(f"[Seed {seed}] Sampling data")
            examples = load_sampled_train(seed)

            acc = asyncio.run(
                run_mcq_eval(
                    json_path=json_path,
                    model=MODEL,
                    seed=seed,
                    examples=examples,
                    society_name=society_name,
                )
            )

            save_accuracy(SPLIT_NAME, society_name, MODEL, seed, acc)
            soc_results.append(acc)

        # Average accuracy across seeds for this society
        all_results[society_name] = sum(soc_results) / len(soc_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, SPLIT_NAME, "accuracy_summary.json")

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[SYSTEM] Accuracy summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()