import os
import re
import numpy as np
from collections import Counter
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
        if not lines:
            continue
        header = lines[0].strip()
        if not header.endswith(":"):
            continue
        speaker = header[:-1].strip()
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
        "conversation_length": conversation_length,
        "token_counts": agent_token_counts
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

def analyze_convo(filepath):
    convo = parse_convo(filepath)
    metrics = compute_metrics(convo)

    filename = os.path.basename(filepath)
    convo_name = os.path.splitext(filename)[0]
    save_folder = os.path.join("metrics", convo_name)
    os.makedirs(save_folder, exist_ok=True)

    print("Turn Dominance:", metrics["turn_dominance"])
    print("Token Dominance:", metrics["token_dominance"])
    print("Conversation Length:", metrics["conversation_length"])

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--convo", required=True)
    args = parser.parse_args()
    analyze_convo(args.convo)
