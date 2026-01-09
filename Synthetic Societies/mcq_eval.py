import os
import random
import asyncio

from agents.societies import create_society_from_json
from mcq_run import autonomous_loop


def get_model_answer(conversation_log):
    """
    Extract and validate the final answer produced by the finalizer agent.
    Assume the finalizer is always appended last.
    """
    speaker, msg = conversation_log[-1]

    # Sanity check: the last turn must come from the finalizer
    if speaker.lower() != "finalizer":
        raise ValueError(f"Last speaker is not finalizer: {speaker}")

    ans = msg.strip().upper()

    # MEDMCQA answers are constrained to these options
    if ans not in {"A", "B", "C", "D", "N/A"}:
        raise ValueError(f"Invalid finalizer output: {ans}")

    return ans


def build_prompt(ex):
    """
    Construct the discussion prompt shown to the non-finalizer agents.
    Agents are instructed to reason freely but not commit to a final answer.
    """
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
    """
    Run the finalizer agent over the full discussion transcript.
    The finalizer's job is to infer the consensus decision, not re-solve the question.
    """
    transcript = ""
    for speaker, msg in convo:
        transcript += f"{speaker}:\n{msg}\n\n"

    final_msg = ""

    # Stream the finalizer response to handle long outputs correctly
    async for event in finalizer.run_stream(
        task=(
            "You are the finalizer.\n"
            "Please read the discussion below and identify the answer option that the agents converged on.\n\n"
            "Guidelines:\n"
            "- Select the option (A, B, C, or D) that is clearly supported in the discussion.\n"
            "- Do not select an option that is described as incorrect or eliminated.\n"
            "- If the discussion eliminates all but one option, select the remaining option.\n"
            "- For EXCEPT or NOT questions, select the option identified as incorrect.\n"
            "- If no clear conclusion is present, respond with N/A.\n\n"
            "Discussion:\n"
            f"{transcript}\n\n"
            "Respond with one of the following: A, B, C, D, or N/A."
        )
    ):
        if hasattr(event, "content") and event.content:
            final_msg = event.content.strip()

    return final_msg


async def run_mcq_eval(json_path, model, seed, examples, society_name):
    """
    Run MCQ evaluation for a single society configuration and seed.
    Handles agent creation, discussion, finalization, logging, and accuracy computation.
    """
    random.seed(seed)

    # Load agents and settings from the society JSON
    agents, settings, entry_point, edges = create_society_from_json(
        json_path,
        model_name=model,
    )

    finalizer = None
    discussion_agents = {}

    # Separate the finalizer from discussion agents explicitly
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

    # Per-question convos are saved for later analysis
    base_dir = f"results/mcq/{society_name}/{model}/seed_{seed}"
    os.makedirs(base_dir, exist_ok=True)

    for i, ex in enumerate(examples, 1):
        qid = ex.get("id", f"q_{i}")
        print(f"[Q {i}/{total}] ID={qid}")

        prompt = build_prompt(ex)

        # Retry loop to handle API failures 
        while True:
            try:
                convo = await autonomous_loop(
                    agents=discussion_agents,
                    settings=settings,
                    task=prompt
                )

                final_msg = await run_finalizer(finalizer, convo)
                convo.append(("finalizer", final_msg))
                break

            except Exception as e:
                if "RateLimit" in str(e) or "429" in str(e):
                    print("[SYSTEM] Rate limit hit — sleeping 65s and retrying...")
                    await asyncio.sleep(65)
                else:
                    raise

        pred = get_model_answer(convo)
        gold = chr(ord("A") + (ex["cop"] - 1))

        is_correct = pred == gold
        if is_correct:
            correct += 1

        # Save the full convo 
        output_file = os.path.join(base_dir, f"{qid}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for speaker, msg in convo:
                f.write(f"{speaker}:\n{msg}\n\n")

        print(f"Predicted: {pred} | Gold: {gold} | Correct: {is_correct}")
        print(f"[SYSTEM] Conversation saved to: {output_file}\n")

        # Small delay to reduce API pressure
        await asyncio.sleep(0.5)

    acc = correct / total
    print(f"\nFinal Accuracy: {acc:.4f}")
    return acc