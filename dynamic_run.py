import sys
import os
from autogen import GroupChat, GroupChatManager
from agents.agentFactory import load_society
from agents.baseAgents import create_LLM_config

def run(config_file_path, task):
    agents, settings = load_society(config_file_path)

    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=settings.get("max_round", 10),
        speaker_selection_method=settings.get("speaker_selection_method", "auto")
    )

    controller = GroupChatManager(
        groupchat=groupchat,
        llm_config=create_LLM_config(), # Uses Gemini by default, but could be made dynamic as well
    )

    initiator = agents[0] # Set to the first agent in the list, but could be made dynamic as well

    controller.initiate_chat(
        recipient=initiator,
        message=task
    )

    return groupchat

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join("configs", "dev_team.json")

    task = "Develop a Python script that prints the first 10 Fibonacci numbers."
    
    x = run(config_path, task)