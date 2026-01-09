import asyncio
import argparse
import re
import random
import os

from run import run_agent, regex_final


async def autonomous_loop(agents, settings, task):
    """
    Multi-agent discussion loop.
    Agents speak in randomized order to avoid fixed turn biases.
    """
    print("[SYSTEM] MCQ Discussion Started\n")

    conversation_log = []
    max_round = settings.get("max_round", 3)
    agent_names = list(agents.keys())
    current_message = task

    for round_idx in range(1, max_round + 1):
        print(f"\n[SYSTEM] ROUND {round_idx}\n")

        speaker_name = random.choice(agent_names)
        speaker = agents[speaker_name]

        # Retry logic API timeouts
        while True:
            try:
                response = await run_agent(speaker, current_message)
                break
            except Exception as e:
                if "Timeout" in str(e) or "ConnectTimeout" in str(e):
                    print("[SYSTEM] Timeout — sleeping 15s and retrying")
                    await asyncio.sleep(15)
                else:
                    raise

        response = regex_final(speaker_name, response)
        print(f"[{speaker_name.upper()}]: {response}")

        conversation_log.append((speaker_name, response))

        # Early stopping condition if a FINAL signal appears (Never happens)
        if speaker_name == "finalizer" and re.search(
            r"^\s*FINAL\s*$",
            response,
            flags=re.IGNORECASE | re.MULTILINE
        ):
            print("\n[SYSTEM] FINAL detected — terminating.")
            return conversation_log

        current_message = response

    print("[SYSTEM] Max rounds reached — ending.")
    return conversation_log