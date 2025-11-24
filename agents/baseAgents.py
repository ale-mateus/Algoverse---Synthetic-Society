from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

def create_agent(name, system_message):
    client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=system_message,
    )
