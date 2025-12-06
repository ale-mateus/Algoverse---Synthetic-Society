import os
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    text = text.replace("\n", " ")
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(emb.data[0].embedding)

def parse_convo(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        if ":" not in block:
            continue
        try:
            name, msg = block.split(":", 1)
            data.append((name.strip(), msg.strip()))
        except:
            continue
    return data

def compute_semantic_drift(messages):
    if len(messages) < 2:
        return 0.0
    embeddings = [get_embedding(m) for m in messages]
    drifts = []
    for i in range(1, len(embeddings)):
        v1 = embeddings[i - 1]
        v2 = embeddings[i]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        drifts.append(1 - cos_sim)
    return float(np.mean(drifts))

def compute_metrics(convo):
    agent_counts = Counter([speaker for speaker, _ in convo])
    total_turns = len(convo)
    dominance = {a: agent_counts[a] / total_turns for a in agent_counts}
    probs = np.array(list(dominance.values()))
    entropy = -(probs * np.log2(probs + 1e-10)).sum()
    messages = [msg for _, msg in convo]
    repetition_ratio = len(messages) / len(set(messages))
    final_present = any(re.search(r"^\s*FINAL\b", msg, re.I) for _, msg in convo)
    semantic_drift = compute_semantic_drift(messages)
    return {
        "dominance": dominance,
        "entropy": float(entropy),
        "semantic_drift": float(semantic_drift),
        "repetition_ratio": repetition_ratio,
        "final_present": final_present
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
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, "metrics.txt")
    with open(filepath, "w") as f:
        f.write("Dominance:\n")
        for agent, val in metrics["dominance"].items():
            f.write(f"  {agent}: {val}\n")
        f.write(f"\nEntropy: {metrics['entropy']}\n")
        f.write(f"Semantic drift: {metrics['semantic_drift']}\n")
        f.write(f"Repetition ratio: {metrics['repetition_ratio']}\n")
        f.write(f"Final present: {metrics['final_present']}\n")

def analyze_convo(filepath):
    convo = parse_convo(filepath)
    metrics = compute_metrics(convo)

    filename = os.path.basename(filepath)
    convo_name = os.path.splitext(filename)[0]
    save_folder = os.path.join("metrics", convo_name)
    os.makedirs(save_folder, exist_ok=True)

    print("Dominance:", metrics["dominance"])
    print("Entropy:", metrics["entropy"])
    print("Semantic drift:", metrics["semantic_drift"])
    print("Repetition ratio:", metrics["repetition_ratio"])
    print("Final present:", metrics["final_present"])

    if metrics["dominance"]:
        plot_metric(
            metrics["dominance"],
            "Agent Dominance",
            "Proportion",
            os.path.join(save_folder, "dominance_plot.png")
        )

    save_metrics_txt(metrics, save_folder)

    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--convo", required=True)
    args = parser.parse_args()
    analyze_convo(args.convo)
