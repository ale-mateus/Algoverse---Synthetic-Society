import os
from collections import Counter, defaultdict
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# IMPORTANT: only these are valid speakers
ALLOWED_SPEAKERS = {
    "architect",
    "engineer",
    "analyst",
    "finalizer",
    "er_physician",
    "er_nurse",
    "er_tech",
    "emt",
    "consult_surgeon",
    "agent1",
    "agent2",
    "agent3"
}


def parse_convo(filepath):
    convo = []
    with open(filepath, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        header = lines[0].strip()

        if not header.endswith(":"):
            continue

        speaker = header[:-1].strip()

        if speaker not in ALLOWED_SPEAKERS:
            continue

        message = "\n".join(lines[1:]).strip()
        if not message:
            continue

        convo.append((speaker, message))

    return convo


def compute_metrics(convo):
    turn_counts = Counter(s for s, _ in convo)
    total_turns = sum(turn_counts.values())
    if total_turns == 0:
        return None

    turn_dom = {s: turn_counts[s] / total_turns for s in turn_counts}

    token_counts = defaultdict(int)
    total_tokens = 0
    for speaker, msg in convo:
        n = len(tokenizer.encode(msg))
        token_counts[speaker] += n
        total_tokens += n

    if total_tokens == 0:
        return None

    token_dom = {s: token_counts[s] / total_tokens for s in token_counts}

    return {
        "turn_dominance": turn_dom,
        "token_dominance": token_dom,
        "conversation_length": total_tokens
    }


def save_metrics(metrics, out_dir, fname="dataset_metrics.txt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)

    with open(path, "w") as f:
        f.write("Turn Dominance:\n")
        for k, v in metrics["turn_dominance"].items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write("\nToken Dominance:\n")
        for k, v in metrics["token_dominance"].items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write(f"\nConversation Length (tokens): {metrics['conversation_length']}\n")


def plot_metric(metric, title, out_path):
    if not metric:
        return

    plt.figure(figsize=(6, 4))
    plt.bar(metric.keys(), metric.values())
    plt.title(title)
    plt.ylabel("Proportion")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path)
    plt.close()


def analyze_convo_file(txt_path):
    convo = parse_convo(txt_path)
    if not convo:
        return None

    metrics = compute_metrics(convo)
    if metrics is None:
        return None

    out_dir = txt_path.replace("results", "dataset_metrics").rsplit(".", 1)[0]
    os.makedirs(out_dir, exist_ok=True)

    save_metrics(metrics, out_dir)

    plot_metric(
        metrics["turn_dominance"],
        "Turn Dominance",
        os.path.join(out_dir, "turn_dominance.png")
    )

    plot_metric(
        metrics["token_dominance"],
        "Token Dominance",
        os.path.join(out_dir, "token_dominance.png")
    )

    return metrics


def walk_all_mcq_txt(base_dir="results"):
    txts = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".txt"):
                txts.append(os.path.join(root, f))
    return txts


def main():
    txt_files = walk_all_mcq_txt()

    for txt in txt_files:
        analyze_convo_file(txt)

    print("\nAll MCQ conversations processed (single seed).\n")


if __name__ == "__main__":
    main()
