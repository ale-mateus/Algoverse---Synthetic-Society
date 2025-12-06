from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

def get_model_client(model_name: str, provider:str = "openai"):
    if provider == "openai":
        return OpenAIChatCompletionClient(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
            
        )
    elif provider == "openrouter":
        return OpenAIChatCompletionClient(
            model=model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
def create_agent(name, system_message, model_name="gpt-4.1-nano" , provider="openai"):
    client = get_model_client(model_name, provider)
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=system_message,
    )
