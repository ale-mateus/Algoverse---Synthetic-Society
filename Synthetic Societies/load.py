import json
import random

TRAIN_JSON = "datasets/MEDMCQA/train.json"
SAMPLE_SIZE = 100


def load_train_sample():
    with open(TRAIN_JSON, "r") as f:
        lines = f.readlines()

    items = [json.loads(l) for l in lines]
    sample = random.sample(items, SAMPLE_SIZE)
    return sample


if __name__ == "__main__":
    sample = load_train_sample()
    print(f"Loaded {len(sample)} sampled examples.")
