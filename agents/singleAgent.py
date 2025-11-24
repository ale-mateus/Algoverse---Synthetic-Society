from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

def create_single_agent():
    client = OpenAIChatCompletionClient(model="gpt-4.1-nano")

    return AssistantAgent(
        name="SingleAgent",
        model_client=client,
        system_message=(
            "You are a single autonomous AI agent. "
            "You must complete the entire task alone without communicating with any other agents. "
            "Produce the clearest, most complete answer you can."
        ),
    )
