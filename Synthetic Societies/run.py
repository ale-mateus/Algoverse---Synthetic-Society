import asyncio
import argparse
import re
import random
import os
from agents.societies import create_society_from_json

async def run_agent(agent, message):
    content = ""
    async for event in agent.run_stream(task=message):
        if hasattr(event, "content") and event.content:
            content = event.content
    return content

def regex_final(agent_name, response):
    if agent_name != "finalizer":
        if re.search(r"^\s*FINAL\s*$", response, flags=re.IGNORECASE | re.MULTILINE):
            return re.sub(
                r"^\s*FINAL\s*$",
                "ERROR: Only finalizer may say FINAL.",
                response,
                flags=re.IGNORECASE | re.MULTILINE
            )
    return response

async def autonomous_loop(agents, settings, entry_point, edges, task):
    print("[SYSTEM] Free-Talk Mode Activated\n")
    conversation_log = []
    max_round = settings.get("max_round", 20)
    global_context = f"Conversation begins:\nUser task: {task}\n"
    current_message = task
    agent_names = list(agents.keys())

    for round_idx in range(1, max_round + 1):
        print(f"\n[SYSTEM] ROUND {round_idx}\n")

        speaker_name = random.choice(agent_names)
        speaker = agents[speaker_name]
        last_speaker = conversation_log[-1][0] if conversation_log else "user"

        prompt = (
            f"{global_context}\n"
            f"The previous message is from {last_speaker}:\n{current_message}\n\n"
            f"Please respond naturally in the conversation.\n"
            f"Do NOT choose who speaks next.\n"
            f"Do NOT output instructions or meta-information.\n"
            f"ONLY continue the discussion."
        )

        response = await run_agent(speaker, prompt)
        response = regex_final(speaker_name, response)

        print(f"[{speaker_name.upper()}]: {response}")

        conversation_log.append((speaker_name, response))
        global_context += f"\n{speaker_name}: {response}\n"

        if speaker_name == "finalizer" and re.search(r"^\s*FINAL\s*$", response, flags=re.IGNORECASE | re.MULTILINE):
            print("\n[SYSTEM] FINAL detected — terminating free-talk.")
            return conversation_log

        current_message = response

    print("[SYSTEM] Max rounds reached — ending free-talk.")
    return conversation_log

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--task", default="Build a short plan for a mobile app.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", required=False, default="default")
    parser.add_argument("--mcq", default="False")   # <<< NEW
    args = parser.parse_args()

    json_path = args.json
    society_name = os.path.splitext(os.path.basename(json_path))[0]

    if args.mcq == "True":
        output_dir = os.path.join(
            "results",
            society_name,
            args.model,
            f"seed_{args.seed}"
        )
    else:
        output_dir = os.path.join(
            "conversations",
            society_name,
            args.label,
            args.provider,
            args.model,
            f"seed_{args.seed}"
        )

    os.makedirs(output_dir, exist_ok=True)

    agents, settings, entry_point, edges = create_society_from_json(
        json_path,
        model_name=args.model,
        provider=args.provider
    )

    convo = await autonomous_loop(
        agents=agents,
        settings=settings,
        entry_point=entry_point,
        edges=edges,
        task=args.task
    )
    if args.mcq == "True":
        filename = f"{args.label}.txt" 
    else:
        filename = "conversation.txt"

    output_file = os.path.join(output_dir, filename)

    with open(output_file, "w", encoding="utf-8") as f:
        for speaker, msg in convo:
            f.write(f"{speaker}:\n{msg}\n\n")

    print(f"\n[SYSTEM] Conversation saved to: {output_file}\n")

if __name__ == "__main__":
    asyncio.run(main())
