from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/medmcqa/train.json",
        "validation": "datasets/medmcqa/dev.json",
        "test": "datasets/medmcqa/test.json"
    }
)
