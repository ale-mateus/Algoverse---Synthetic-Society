import json
import asyncio
import os
import re
from agents.societies import create_society_from_json

LETTER_PATTERN = re.compile(r"\b([A-D])\b", re.IGNORECASE)

def extract_letters(text, multi=False):
    text = text.upper()
    if multi:
        return sorted(list(set(LETTER_PATTERN.findall(text))))
    else:
        m = LETTER_PATTERN.search(text)
        return m.group(1) if m else None

def build_prompt(ex):
    return f"""
You are the finalizer agent. Answer the following multiple-choice question.

Question:
{ex["question"]}

Options:
A: {ex["opa"]}
B: {ex["opb"]}
C: {ex["opc"]}
D: {ex["opd"]}

Respond ONLY with the letter(s) of the correct option(s).
""".strip()

async def run_single_mcq_with_logs(agent, prompt):
    messages = []
    last = ""

    async for event in agent.run_stream(task=prompt):
        if hasattr(event, "content") and event.content:
            last = event.content
            messages.append(event.content)

    return last.strip(), messages


async def run_mcq_eval(json_path, model, seed, examples, society_name):
    agents, settings, entry_point, edges = create_society_from_json(
        json_path, model_name=model, provider="openai"
    )

    if "finalizer" in agents:
        finalizer = agents["finalizer"]
    else:
        print(f"[WARNING] Society {society_name} has no finalizer. Choosing a random agent.")
        finalizer = list(agents.values())[-1]

    correct = 0
    total = len(examples)

    print(f"Evaluating {total} questions...")

    log_dir = f"results/logs/{society_name}/{model}/seed_{seed}"
    os.makedirs(log_dir, exist_ok=True)

    for i, ex in enumerate(examples, 1):
        prompt = build_prompt(ex)
        pred_text, msg_log = await run_single_mcq_with_logs(finalizer, prompt)

        if ex["choice_type"] == "single":
            pred = extract_letters(pred_text, multi=False)
            gold = chr(ord("A") + (ex["cop"] - 1))
            is_correct = (pred == gold)

        elif ex["choice_type"] == "multi":
            pred = extract_letters(pred_text, multi=True)
            gold = sorted([
                chr(ord("A") + idx)
                for idx in range(4)
                if ex["cop"] == idx + 1
            ])
            is_correct = (pred == gold)

        else:
            is_correct = False

        qid = ex.get("id", f"q_{i}")
        out_path = os.path.join(log_dir, f"{qid}.json")
        with open(out_path, "w") as f:
            json.dump({
                "id": qid,
                "question": ex["question"],
                "options": {
                    "A": ex["opa"],
                    "B": ex["opb"],
                    "C": ex["opc"],
                    "D": ex["opd"]
                },
                "gold": ex["cop"],
                "choice_type": ex["choice_type"],
                "pred_raw": pred_text,
                "pred_parsed": pred,
                "correct": is_correct,
                "messages": msg_log
            }, f, indent=2)

        if is_correct:
            correct += 1

        if i % 20 == 0:
            print(f"Processed {i}/{total}...")

    return correct / total
