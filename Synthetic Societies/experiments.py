import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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

    turn_dominance = {
        a: agent_counts[a] / total_turns for a in agent_counts
    }

    agent_token_counts = {}
    total_tokens = 0

    for speaker, message in convo:
        tokens = tokenizer.encode(message)
        num_tokens = len(tokens)
        total_tokens += num_tokens
        agent_token_counts[speaker] = agent_token_counts.get(speaker, 0) + num_tokens

    token_dominance = {
        a: agent_token_counts[a] / total_tokens for a in agent_token_counts
    }

    return {
        "turn_dominance": turn_dominance,
        "token_dominance": token_dominance,
        "conversation_length": total_tokens
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

def save_metrics_txt(metrics, save_folder):
    filepath = os.path.join(save_folder, "metrics.txt")
    with open(filepath, "w") as f:
        f.write("Turn Dominance:\n")
        for agent, val in metrics["turn_dominance"].items():
            f.write(f"  {agent}: {val:.4f}\n")
        f.write("\nToken Dominance:\n")
        for agent, val in metrics["token_dominance"].items():
            f.write(f"  {agent}: {val:.4f}\n")
        f.write(f"\nConversation Length (tokens): {metrics['conversation_length']}\n")

def analyze_convo(convo_path, save_folder):
    convo = parse_convo(convo_path)
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

def walk_all_convos(base_dir="conversations"):
    convo_files = []
    for root, _, files in os.walk(base_dir):
        if "conversation.txt" in files:
            convo_files.append(os.path.join(root, "conversation.txt"))
    return convo_files

def process_all():
    convo_files = walk_all_convos()
    for convo_path in convo_files:
        relative = convo_path.replace("conversations", "").strip("/")
        save_folder = os.path.join("metrics", os.path.dirname(relative))
        analyze_convo(convo_path, save_folder)

if __name__ == "__main__":
    process_all()
    print("\nAll conversations processed.\n")