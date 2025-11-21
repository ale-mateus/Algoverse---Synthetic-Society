from autogen import GroupChat, GroupChatManager
from agents.societies import createSociety
from agents.baseAgents import create_LLM_config

def run(task):
    manager, developer, tester = createSociety()

    groupchat = GroupChat(
        agents=[manager, developer, tester],
        messages=[],
        max_round=2
    )

    controller = GroupChatManager(
        groupchat=groupchat,
        llm_config=create_LLM_config(),
    )

    controller.initiate_chat(
        recipient=manager,
        message=task
    )

    return groupchat

if __name__ == "__main__":
    task = "Develop a program that prints \"Hello, World!\""
    x = run(task)

    print("\n=== CONVERSATION ===\n")
    for msg in x.messages:
        print(f"{msg['role']} -> {msg.get('to','group')}:\n{msg['content']}\n{'-'*40}")
