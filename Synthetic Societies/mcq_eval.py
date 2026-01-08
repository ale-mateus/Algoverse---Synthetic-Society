import os
import random
from agents.societies import create_society_from_json
from run import autonomous_loop


def get_model_answer(conversation_log):
    speaker, msg = conversation_log[-1]
    if speaker.lower() != "finalizer":
        raise ValueError(f"Last speaker is not finalizer: {speaker}")
    ans = msg.strip().upper()
    if ans not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid finalizer output: {ans}")
    return ans


def build_prompt(ex):
    return f"""
This is a multiple-choice medical question.

Question:
{ex["question"]}

Options:
A: {ex["opa"]}
B: {ex["opb"]}
C: {ex["opc"]}
D: {ex["opd"]}

Instructions:
- Agents may discuss freely.
- Reach a consensus on the correct answer.
- Do NOT finalize yet.

Begin discussion.
""".strip()


async def run_finalizer(finalizer, convo):
    transcript = ""
    for speaker, msg in convo:
        transcript += f"{speaker}:\n{msg}\n\n"

    final_msg = ""
    async for event in finalizer.run_stream(
        task=(
            "You are the finalizer.\n"
            "Based ONLY on the discussion below, choose exactly ONE best answer.\n"
            "Even if multiple options seem partially correct, select the single "
            "most standard answer expected in medical entrance exams.\n\n"
            "Discussion:\n"
            f"{transcript}\n"
            "Output ONLY the letter A, B, C, or D."
        )
    ):
        if hasattr(event, "content") and event.content:
            final_msg = event.content.strip()

    return final_msg


async def run_mcq_eval(json_path, model, seed, examples, society_name):
    random.seed(seed)

    agents, settings, entry_point, edges = create_society_from_json(
        json_path,
        model_name=model,
    )

    finalizer = None
    discussion_agents = {}

    for name, agent in agents.items():
        if name.lower() == "finalizer":
            finalizer = agent
        else:
            discussion_agents[name] = agent

    if finalizer is None:
        raise RuntimeError("Finalizer agent not found in society JSON")

    total = len(examples)
    correct = 0

    print("\n========================================")
    print("Running MCQ Evaluation (Autonomous Loop)")
    print(f"Society: {society_name}")
    print(f"Model: {model}")
    print(f"Seed: {seed}")
    print(f"Num questions: {total}")
    print("========================================\n")

    base_dir = f"results/mcq/{society_name}/{model}/seed_{seed}"
    os.makedirs(base_dir, exist_ok=True)

    for i, ex in enumerate(examples, 1):
        qid = ex.get("id", f"q_{i}")
        print(f"[Q {i}/{total}] ID={qid}")

        prompt = build_prompt(ex)

        convo = await autonomous_loop(
            agents=discussion_agents,
            settings=settings,
            entry_point=entry_point,
            edges=edges,
            task=prompt
        )

        final_msg = await run_finalizer(finalizer, convo)
        convo.append(("finalizer", final_msg))

        pred = get_model_answer(convo)
        gold = chr(ord("A") + (ex["cop"] - 1))

        is_correct = pred == gold
        if is_correct:
            correct += 1

        output_file = os.path.join(base_dir, f"{qid}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for speaker, msg in convo:
                f.write(f"{speaker}:\n{msg}\n\n")

        print(f"Predicted: {pred} | Gold: {gold} | Correct: {is_correct}")
        print(f"[SYSTEM] Conversation saved to: {output_file}\n")

    acc = correct / total
    print(f"\nFinal Accuracy: {acc:.4f}")
    return acc
