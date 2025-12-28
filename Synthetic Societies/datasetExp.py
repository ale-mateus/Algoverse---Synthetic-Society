import os
from collections import Counter, defaultdict
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def parse_convo(filepath):
    convo = []
    with open(filepath, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        if not lines[0].endswith(":"):
            continue
        speaker = lines[0][:-1].strip()
        message = "\n".join(lines[1:]).strip()
        convo.append((speaker, message))
    return convo


def compute_metrics(convo):
    turn_counts = Counter(s for s, _ in convo)
    total_turns = sum(turn_counts.values())

    turn_dom = {s: turn_counts[s] / total_turns for s in turn_counts}

    token_counts = defaultdict(int)
    total_tokens = 0
    for speaker, msg in convo:
        n = len(tokenizer.encode(msg))
        token_counts[speaker] += n
        total_tokens += n

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
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\nToken Dominance:\n")
        for k, v in metrics["token_dominance"].items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write(f"\nConversation Length (tokens): {metrics['conversation_length']}\n")


def plot_metric(metric, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.bar(metric.keys(), metric.values())
    plt.title(title)
    plt.ylabel("Proportion")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_convo_file(txt_path):
    convo = parse_convo(txt_path)
    if not convo:
        return None

    metrics = compute_metrics(convo)

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


def average_seed_metrics(seed_dir):
    aggregate_turn = defaultdict(float)
    aggregate_token = defaultdict(float)
    lengths = []
    count = 0

    for root, _, files in os.walk(seed_dir):
        if "dataset_metrics.txt" not in files:
            continue

        path = os.path.join(root, "dataset_metrics.txt")
        with open(path) as f:
            lines = f.read().splitlines()

        i = 1
        while i < len(lines) and lines[i].startswith("  "):
            k, v = lines[i].split(":")
            aggregate_turn[k.strip()] += float(v)
            i += 1

        i += 2
        while i < len(lines) and lines[i].startswith("  "):
            k, v = lines[i].split(":")
            aggregate_token[k.strip()] += float(v)
            i += 1

        for line in lines:
            if "Conversation Length" in line:
                lengths.append(float(line.split(":")[1]))

        count += 1

    if count == 0:
        return

    avg = {
        "turn_dominance": {k: v / count for k, v in aggregate_turn.items()},
        "token_dominance": {k: v / count for k, v in aggregate_token.items()},
        "conversation_length": sum(lengths) / count
    }

    out_dir = os.path.join(seed_dir, "AVERAGED")
    save_metrics(avg, out_dir, "averaged_dataset_metrics.txt")

    plot_metric(
        avg["turn_dominance"],
        "Averaged Turn Dominance",
        os.path.join(out_dir, "turn_dominance_avg.png")
    )

    plot_metric(
        avg["token_dominance"],
        "Averaged Token Dominance",
        os.path.join(out_dir, "token_dominance_avg.png")
    )


def main():
    txt_files = walk_all_mcq_txt()

    for txt in txt_files:
        analyze_convo_file(txt)

    for root, dirs, _ in os.walk("dataset_metrics"):
        for d in dirs:
            if d.startswith("seed_"):
                average_seed_metrics(os.path.join(root, d))


if __name__ == "__main__":
    main()
    print("\nAll MCQ conversations processed and averaged.\n")