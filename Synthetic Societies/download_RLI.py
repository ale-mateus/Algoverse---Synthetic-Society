from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="cais/rli-public-set",
    repo_type="dataset",
    local_dir="datasets/RLI",
    local_dir_use_symlinks=False
)

print("RLI dataset downloaded to:", dataset_path)
