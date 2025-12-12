import os
import re
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def parse_convo(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or not lines[0].endswith(":"):
            continue
        speaker = lines[0][:-1].strip()
        message = "\n".join(lines[1:]).strip()
        if speaker and message:
            data.append((speaker, message))
    return data

def compute_metrics(convo):
    agent_counts = Counter([speaker for speaker, _ in convo])
    total_turns = len(convo)
    turn_dominance = {a: agent_counts[a] / total_turns for a in agent_counts}

    agent_token_counts = {}
    total_tokens = 0
    for speaker, message in convo:
        num_tokens = len(tokenizer.encode(message))
        total_tokens += num_tokens
        agent_token_counts[speaker] = agent_token_counts.get(speaker, 0) + num_tokens

    token_dominance = {a: agent_token_counts[a] / total_tokens for a in agent_token_counts}
    conversation_length = total_tokens

    return {
        "turn_dominance": turn_dominance,
        "token_dominance": token_dominance,
        "conversation_length": conversation_length
    }

def plot_metric(metric_dict, title, ylabel, savepath):
    agents = list(metric_dict.keys())
    values = list(metric_dict.values())
    plt.figure(figsize=(6, 4))
    plt.bar(agents, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def save_metrics_txt(metrics, save_folder, name="metrics.txt"):
    filepath = os.path.join(save_folder, name)
    with open(filepath, "w") as f:
        f.write("Turn Dominance:\n")
        for agent, val in metrics["turn_dominance"].items():
            f.write(f"  {agent}: {val:.4f}\n")
        f.write("\nToken Dominance:\n")
        for agent, val in metrics["token_dominance"].items():
            f.write(f"  {agent}: {val:.4f}\n")
        f.write(f"\nConversation Length (tokens): {metrics['conversation_length']}\n")

def analyze_convo(filepath, save_folder):
    convo = parse_convo(filepath)
    metrics = compute_metrics(convo)

    os.makedirs(save_folder, exist_ok=True)

    plot_metric(
        metrics["turn_dominance"],
        "Turn Dominance",
        "Proportion",
        os.path.join(save_folder, "turn_dominance.png")
    )

    plot_metric(
        metrics["token_dominance"],
        "Token Dominance",
        "Proportion",
        os.path.join(save_folder, "token_dominance.png")
    )

    save_metrics_txt(metrics, save_folder)
    return metrics

def walk_all_convos(base_dir="conversations"):
    convo_files = []
    for root, dirs, files in os.walk(base_dir):
        if "conversation.txt" in files:
            convo_files.append(os.path.join(root, "conversation.txt"))
    return convo_files

def average_across_seeds(root_path):
    seed_dirs = [d for d in os.listdir(root_path) if d.startswith("seed_")]
    if not seed_dirs:
        return

    sum_turn = defaultdict(float)
    sum_token = defaultdict(float)
    lengths = []
    count = 0

    for seed in seed_dirs:
        metrics_path = os.path.join(root_path, seed, "metrics.txt")
        if not os.path.exists(metrics_path):
            continue

        turn_dom = {}
        token_dom = {}
        length = None

        with open(metrics_path, "r") as f:
            lines = f.read().splitlines()

        i = 1
        while i < len(lines) and lines[i].startswith("  "):
            agent, val = lines[i].split(":")
            turn_dom[agent.strip()] = float(val)
            i += 1

        i += 2
        while i < len(lines) and lines[i].startswith("  "):
            agent, val = lines[i].split(":")
            token_dom[agent.strip()] = float(val)
            i += 1

        for line in lines:
            if "Conversation Length" in line:
                length = float(line.split(":")[1])

        for a, v in turn_dom.items():
            sum_turn[a] += v
        for a, v in token_dom.items():
            sum_token[a] += v
        lengths.append(length)

        count += 1

    if count == 0:
        return

    avg_turn = {a: sum_turn[a] / count for a in sum_turn}
    avg_token = {a: sum_token[a] / count for a in sum_token}
    avg_length = sum(lengths) / count

    averaged = {
        "turn_dominance": avg_turn,
        "token_dominance": avg_token,
        "conversation_length": avg_length
    }

    out_dir = os.path.join(root_path, "AVERAGED")
    os.makedirs(out_dir, exist_ok=True)

    save_metrics_txt(averaged, out_dir, name="averaged_metrics.txt")

    plot_metric(avg_turn, "Averaged Turn Dominance", "Proportion",
                os.path.join(out_dir, "turn_dominance_avg.png"))
    plot_metric(avg_token, "Averaged Token Dominance", "Proportion",
                os.path.join(out_dir, "token_dominance_avg.png"))

def process_and_average():
    convo_files = walk_all_convos()

    for convo_path in convo_files:
        relative = convo_path.replace("conversations", "").strip("/")
        save_folder = os.path.join("metrics", os.path.dirname(relative))

        analyze_convo(convo_path, save_folder)

    for root, dirs, _ in os.walk("metrics"):
        if any(d.startswith("seed_") for d in dirs):
            average_across_seeds(root)

if __name__ == "__main__":
    process_and_average()
    print("\nAll conversations processed and averaged.\n")
