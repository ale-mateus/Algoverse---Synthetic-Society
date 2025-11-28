import asyncio
import argparse
import re
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
            response = re.sub(r"^\s*FINAL\s*$", "ERROR: Only finalizer may say FINAL.", response, flags=re.IGNORECASE | re.MULTILINE)
    return response

async def autonomous_loop(agents, settings, entry_point, edges, task):
    adjacency = {name: [] for name in agents.keys()}
    for e in edges:
        adjacency[e["from"]].append(e["to"])

    max_round = settings.get("max_round", 20)
    free_talk = settings.get("free_talk", False)

    current_agent = agents[entry_point]
    last_message = task
    round_count = 0
    conversation_log = []

    print("[SYSTEM]COLLABORATION STARTED")

    while round_count < max_round:
        round_count += 1
        print(f"\n[SYSTEM] Round {round_count}\n")

        response = await run_agent(current_agent, last_message)
        response = regex_final(current_agent.name, response)

        print(f"[{current_agent.name.upper()}]: {response}")
        conversation_log.append((current_agent.name, response))

        if current_agent.name == "finalizer" and re.search(r"^\s*FINAL\s*$", response, flags=re.IGNORECASE | re.MULTILINE):
            print("\n[SYSTEM] FINAL detected — conversation terminated.")
            return conversation_log

        if free_talk:
            possible_next = list(agents.keys())
        else:
            possible_next = adjacency[current_agent.name]

        if not possible_next:
            print("[SYSTEM] No next agent available. Ending.")
            return conversation_log

        selector_prompt = (
            f"Choose who speaks next from this list: {possible_next}. "
            f"Respond ONLY with the agent name."
        )

        next_choice = await run_agent(current_agent, selector_prompt)
        next_choice = next_choice.strip().lower()

        if next_choice not in possible_next:
            next_choice = possible_next[0]

        current_agent = agents[next_choice]
        last_message = response

    print("[SYSTEM] Max rounds reached. Ending.")
    return conversation_log

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--task", required=False, default="Build a short plan for a mobile app.")
    args = parser.parse_args()

    agents, settings, entry_point, edges = create_society_from_json(args.json)

    convo = await autonomous_loop(
        agents=agents,
        settings=settings,
        entry_point=entry_point,
        edges=edges,
        task=args.task
    )

    with open("devSociety.txt", "w", encoding="utf-8") as f:
        for speaker, msg in convo:
            f.write(f"{speaker}:\n{msg}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
