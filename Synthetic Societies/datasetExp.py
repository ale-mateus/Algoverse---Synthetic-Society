import os
from collections import Counter, defaultdict
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

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
    current_speaker = None
    current_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line.endswith(":"):
                speaker = line[:-1].strip()
                if speaker in ALLOWED_SPEAKERS:
                    if current_speaker and current_lines:
                        convo.append((current_speaker, "\n".join(current_lines).strip()))
                    current_speaker = speaker
                    current_lines = []
                else:
                    if current_speaker:
                        current_lines.append(line)
            else:
                if current_speaker:
                    current_lines.append(line)

    if current_speaker and current_lines:
        convo.append((current_speaker, "\n".join(current_lines).strip()))

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


def save_metrics(metrics, out_dir, fname):
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

    save_metrics(metrics, out_dir, "dataset_metrics.txt")

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


def average_dicts(dicts):
    avg = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            avg[k] += v
    n = len(dicts)
    return {k: v / n for k, v in avg.items()}


def main():
    txt_files = walk_all_mcq_txt()
    society_metrics = defaultdict(list)

    for txt in txt_files:
        metrics = analyze_convo_file(txt)
        if metrics is None:
            continue
        society = txt.split(os.sep)[1]
        society_metrics[society].append(metrics)

    for society, metrics_list in society_metrics.items():
        avg_turn = average_dicts([m["turn_dominance"] for m in metrics_list])
        avg_token = average_dicts([m["token_dominance"] for m in metrics_list])
        avg_len = sum(m["conversation_length"] for m in metrics_list) / len(metrics_list)

        out_dir = os.path.join("dataset_metrics", society, "averaged")
        os.makedirs(out_dir, exist_ok=True)

        save_metrics(
            {
                "turn_dominance": avg_turn,
                "token_dominance": avg_token,
                "conversation_length": avg_len
            },
            out_dir,
            "averaged_metrics.txt"
        )

        plot_metric(
            avg_turn,
            f"{society} Avg Turn Dominance",
            os.path.join(out_dir, "avg_turn_dominance.png")
        )

        plot_metric(
            avg_token,
            f"{society} Avg Token Dominance",
            os.path.join(out_dir, "avg_token_dominance.png")
        )

    print("\nAll MCQ conversations processed (averaged per society).\n")


if __name__ == "__main__":
    main()